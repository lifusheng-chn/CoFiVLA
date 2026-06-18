import torch.distributed as dist
import torch
import os
from typing import Optional, List
from torch import Tensor
import torchvision
import numpy as np
import random
from tqdm import tqdm
import time
import datetime
from timm.utils import AverageMeter # 用于跟踪和计算一些指标的平均值
import torchvision.transforms as T
import sys
from torch import optim
from sklearn.metrics import accuracy_score
import clip
import math
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from .IAAloss import EMDLoss
from .IAAloss import EMDLoss1
from .IAAloss import IAALoss 
import torch.nn.functional as F
from models.longclip import tokenize
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler



try:
    from apex import amp
except:
    amp = None
    
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# NEW metrics
# metrics_printed = ['ACC', 'SROCC', 'PLCC', 'MSE', 'RMSE']
metrics_printed = ['ACC', 'SROCC', 'PLCC', 'MSE']


def writer_add_scalar(writer, status, dataset, scalars, iter):
    for metric_print in metrics_printed:
        writer.add_scalar('{}/{}/{}'.format(status, dataset,
                                            metric_print), scalars[metric_print], iter)


def train_one_epoch(args, model, data_loader, optimizer, epoch, lr_scheduler, device, writer, logger):
    torch.cuda.empty_cache()
    model.train()
    optimizer.zero_grad()
    scaler = GradScaler()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    promptloss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    # 创建损失函数
    # 定义损失函数 (Euclidean distance)
    clsEMDLoss = EMDLoss().to(device)
    mse_loss = nn.MSELoss().to(device)
    # 创建 L1 损失函数
    l1_loss = nn.L1Loss().to(device)
    js_loss = IAALoss(loss_type="js")


    total_loss = 0.0
    losses = []
    pred_score_list = []
    
    # optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    # ori:
    for idx, (im, label) in tqdm(enumerate(data_loader)):
        with torch.cuda.amp.autocast():
            im = im.to(device, non_blocking=True)
            # texts = texts.to(device, non_blocking=True)


            # key_texts = key_texts.to(device, non_blocking=True)
            # key_texts = key_texts.squeeze(1)

            label = torch.tensor(label, dtype=torch.float32).to(
                device, non_blocking=True).view(-1, 10)
            tscore, tscore_np = get_score(device, label)
            true_labels = score_to_label(tscore)
            
            # gt_dis = score_to_one_hot(pscore_np)
            # gt_dis = torch.tensor(gt_dis, dtype=torch.float32).to(
            #     device, non_blocking=True)
            
        
            outputs = model(im)


            
            pre_score = outputs['pre_score']
            # pre_score = pre_score.squeeze(1)

            probabilities = outputs['probabilities']
            prompt_loss = - torch.mean(torch.sum(true_labels * torch.log(probabilities + 1e-8), dim=1))  # 添加一个小值防止 log(0)


            class_loss = clsEMDLoss(p_target=label, p_estimate=pre_score)
            loss = class_loss + 0.01 * prompt_loss
            
             
            # 使用 cross_entropy 计算交叉熵损失
            # dis_loss = F.cross_entropy(probs, gt_dis)
            # class_loss = clsEDMLoss(p_target=label, p_estimate=outputs["pred_logits"]

            

        if args.train_accumulation_steps > 1:
            loss_backward = loss / args.train_accumulation_steps

            if args.opt_level != 'O0' and amp is not None:
                # Apex AMP 混合精度反向传播
                with amp.scale_loss(loss_backward, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # 普通 FP32 反向传播
                loss_backward.backward()

           
            should_update = (
                ((idx + 1) % args.train_accumulation_steps == 0)
                or ((idx + 1) == num_steps)
            )

            if should_update:
                if args.opt_level != 'O0' and amp is not None:
                    # Apex AMP 下使用 master parameters
                    params_for_grad = amp.master_params(optimizer)

                    if args.clip_grad:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            params_for_grad,
                            args.clip_grad
                        )
                    else:
                        grad_norm = get_grad_norm(params_for_grad)

                else:
                    params_for_grad = [
                        p for p in model.parameters()
                        if p.requires_grad and p.grad is not None
                    ]

                    if len(params_for_grad) > 0:
                        if args.clip_grad:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                params_for_grad,
                                args.clip_grad
                            )
                        else:
                            grad_norm = get_grad_norm(params_for_grad)
                    else:
                        grad_norm = 0.0

    
                optimizer.step()
                optimizer.zero_grad()
                if args.lr_scheduler_name != 'iaatrstep':
                    lr_scheduler.step_update(epoch * num_steps + idx)

            else:
                grad_norm = 0.0



        else:
            if args.opt_level != 'O0' and amp is not None:
                # Apex AMP 混合精度反向传播
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # 普通 FP32 反向传播
                loss.backward()

            if args.opt_level != 'O0' and amp is not None:
                # Apex AMP 下使用 master parameters
                params_for_grad = amp.master_params(optimizer)

                if args.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params_for_grad,
                        args.clip_grad
                    )
                else:
                    grad_norm = get_grad_norm(params_for_grad)

            else:
                # 普通 FP32 下使用 model.parameters()
                params_for_grad = [
                    p for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]

                if len(params_for_grad) > 0:
                    if args.clip_grad:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            params_for_grad,
                            args.clip_grad
                        )
                    else:
                        grad_norm = get_grad_norm(params_for_grad)
                else:
                    grad_norm = 0.0


            optimizer.step()
            optimizer.zero_grad()
            if args.lr_scheduler_name != 'iaatrstep':
                lr_scheduler.step_update(epoch * num_steps + idx)

    

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if writer is not None:
            writer.add_scalar("train/loss", loss_meter.val,
                              epoch * num_steps + idx)

        loss_meter.update(class_loss.item(), label.size(0))
        promptloss_meter.update(prompt_loss.item(), label.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        # 记录到日志中
        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = 0.0
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            if get_rank() == 0:
                logger.info(
                    f'Train: [{epoch}/{args.epochs}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'promptloss_meter {0.01 * promptloss_meter.val:.4f} ({0.01 * promptloss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB\t')

    epoch_time = time.time() - start
    if get_rank() == 0:
        logger.info(
            f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
        


@torch.no_grad()
def validate(args, data_loader, model, device, logger, non_blocking=True):
    model.eval()
    batch_time = AverageMeter() # 实例化用于跟踪和计算批处理时间平均值的AverageMeter类
    num_steps = len(data_loader)
    losstest_meter = AverageMeter()
    clsEDMLoss = EDMLoss1().to(device)

    end = time.time()
    true_score = []
    pred_score = []

    total_inference_time = 0.0
    
    # ori:
    for idx, (images, labels) in tqdm(enumerate(data_loader)):
        with torch.cuda.amp.autocast():
            images = images.to(device, non_blocking=True)
            # 真实美学分数分布列表
            labels = torch.tensor(labels, dtype=torch.float32).clone().to(device, non_blocking=True).view(-1, 10)
            tscore, tscore_np = get_score(device, labels)
            true_score += tscore_np.tolist()
            B, C, H, W = images.shape

            
            with torch.no_grad():
                    pre_score = model(images)["pre_score"]
                    # pre_score = pre_score.squeeze(1)
                    class_loss = clsEMDLoss1(p_target=labels, p_estimate=pre_score)
                    losstest_meter.update(class_loss.item(), labels.size(0))
                    # output = model(images)
                    # pre_score = model(images)["score"]
            pscore, pscore_np = get_score(device, pre_score)
            # score_np = pre_score.data.cpu().numpy()
            pred_score += pscore_np.tolist()
            # pred_score.append(pscore_np)

            # output = {'pred_logits': output}
            # metrics.update((output, labels))

            # 记录推理开始时间
            start_time = time.time()
            output = model(images)
            # 记录推理结束时间
            end_time = time.time()
            # 计算本次推理时间
            inference_time = end_time - start_time
            total_inference_time += inference_time

            # 计算每张图片的推理时间
            per_image_time = inference_time / B  # B 是批次大小
            # logger.info(f'Batch {idx}: Per-image inference time = {per_image_time:.7f} seconds')


        batch_time.update(time.time() - end)
        end = time.time()
        if idx % args.print_freq == 0:
            memory_used = 0.0
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if get_rank() == 0:
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

    # 计算平均每张图片的推理时间
    avg_per_image_time = total_inference_time / (len(data_loader) * args.test_batch_size)
    if get_rank() == 0:
        logger.info(f'* Average Per-Image Inference Time: {avg_per_image_time:.7f} seconds')
    print(f'* Average Per-Image Inference Time: {avg_per_image_time:.7f} seconds')

    #-----------------------指标计算------------------------
    # 计算ACC
    true_score = np.array(true_score)
    true_score_label = np.where(true_score < 5, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_label = np.where(pred_score < 5, 0, 1)
    acc = accuracy_score(true_score_label, pred_score_label)
    
    # 计算PLCC
    plcc = pearsonr(pred_score, true_score)[0]
    print(plcc)
    # 计算SRCC
    srocc = spearmanr(pred_score, true_score)[0]
    # 计算MSE
    mse = mean_squared_error(pred_score, true_score)
    

    if is_dist_avail_and_initialized():
        acc = reduce_tensor(acc)
        srocc = reduce_tensor(srocc)
        plcc = reduce_tensor(plcc)
        # rmse = reduce_tensor(rmse)
        mse = reduce_tensor(mse)
    if get_rank() == 0:
        logger.info(
            f'* ACC {acc:.4f} SROCC {srocc:.4f} PLCC {plcc:.4f} MSE {mse:.4f} EMD {losstest_meter.avg:.4f}')
    return acc, srocc, plcc, mse






def get_score(device, y):
    distance = torch.arange(1, 11).float().to(device)
    score = (y.view(-1, 10) * distance).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def score_to_label(scores, num_classes=5):
    # 初始化标签为 0
    labels = torch.zeros_like(scores, dtype=torch.long)

     # 按照条件划分标签
    labels[(scores >= 1) & (scores < 3)] = 0
    labels[(scores >= 3) & (scores < 5)] = 1
    labels[(scores >= 5) & (scores < 6)] = 2
    labels[(scores >= 6) & (scores < 8)] = 3
    labels[(scores >= 8) & (scores <= 10)] = 4

    # 将标签转换为 one-hot 编码，假设 num_classes 为 5
    one_hot_labels = F.one_hot(labels, num_classes=num_classes)
    
    return one_hot_labels.float()  # 返回浮点型 one-hot 编码

def savediff(predicted_distribution, ground_truth_distribution, ID, logger):

    # 将张量从 GPU 移动到 CPU（如果必要），并转换为 NumPy 数组
    predicted_distribution = predicted_distribution.detach().cpu().numpy()
    ground_truth_distribution = ground_truth_distribution.detach().cpu().numpy()
    # 计算labels与pre之间的差异
    diff = np.abs(ground_truth_distribution - predicted_distribution)  # 计算每个评分位置的差值
    logger.info(
                    f'ID: {ID}\t'
                    f'diff: {diff}\t')

    

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def load_checkpoints(args, model, optimizer, lr_scheduler, logger):
    logger.info(
        f"==============> Resuming form {args.trained_model_file}....................")
    checkpoints = torch.load(args.trained_model_file, map_location='cpu')
    model.load_state_dict(checkpoints['model'], strict=True)
    if not args.evaluate and 'optimizer' in checkpoints and 'lr_scheduler' in checkpoints and 'epoch' in checkpoints:
        optimizer.load_state_dict(checkpoints['optimizer'])
        lr_scheduler.load_state_dict(checkpoints['lr_scheduler'])
        args.start_epoch = checkpoints['epoch'] + 1
        if 'amp' in checkpoints and args.opt_level != 'O0':
            amp.load_state_dict(checkpoints['amp'])
    logger.info(
        f"=> loaded successfully '{args.trained_model_file}' (epoch {checkpoints['epoch']})")
    best_val_criterion, best_epoch = -100, -1
    if 'best_epoch' in checkpoints and 'best_val_criterion' in checkpoints:
        best_val_criterion, best_epoch = checkpoints['best_val_criterion'], checkpoints['best_epoch']
    del checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_val_criterion, best_epoch


def save_checkpoints(args, epoch, model, optimizer, lr_scheduler, logger, best_val_criterion=-1, update=False):
    if args.test_in_train:
        checkpoints = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_epoch': epoch,
            'best_val_criterion': best_val_criterion
        }
    else:
        checkpoints = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch
        }
    if args.opt_level != 'O0' and amp is not None:
        checkpoints['amp'] = amp.state_dict()
    if args.test_in_train and update:
        logger.info(f"{args.trained_model_file} saving......")
        torch.save(checkpoints, args.trained_model_file)
        logger.info(f"{args.trained_model_file} saved !!!")
    # else:
    #     logger.info(f"{args.trained_model_file + '-epoch_' + str(epoch)} saving......")
    #     torch.save(checkpoints, args.trained_model_file + '-epoch_' + str(epoch))
    #     logger.info(f"{args.trained_model_file + '-epoch_' + str(epoch)} saved !!!")
    print(f"{args.trained_model_file + '-epoch_' + str(epoch)} saving......")
    torch.save(checkpoints, args.trained_model_file + '-epoch' + str(epoch))
    print(f"{args.trained_model_file + '-epoch_' + str(epoch)} saved !!!")



def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def init_seed(args): # 初始化随机种子，确保在模型训练中的可重现性
    if not args.randomness:
        torch.manual_seed(args.seed)
        # 设置PyTorch的cuDNN算法为确定性模式，以确保每次运行时卷积算法的输出始终相同。
        torch.backends.cudnn.deterministic = True
        # 关闭cuDNN的自动调整功能，以确保每次运行时卷积算法的性能保持一致。
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


# 根据训练的分布式设置和批量大小来动态调整学习率，以确保在分布式训练中学习率的合理性和稳定性。
def init_lr(args):
    if is_dist_avail_and_initialized() and args.model_type != 'iaatr':
        linear_scaled_lr = args.lr * args.train_batch_size * dist.get_world_size() / 512.0
        linear_scaled_warmup_lr = args.warmup_lr * \
                                  args.train_batch_size * dist.get_world_size() / 512.0
        linear_scaled_min_lr = args.min_lr * args.train_batch_size * dist.get_world_size() / \
                               512.0
        if args.train_accumulation_steps > 1:
            linear_scaled_lr = linear_scaled_lr * args.train_accumulation_steps
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * args.train_accumulation_steps
            linear_scaled_min_lr = linear_scaled_min_lr * args.train_accumulation_steps
        args.lr = linear_scaled_lr
        args.warmup_lr = linear_scaled_warmup_lr
        args.min_lr = linear_scaled_min_lr


def init_dir(args): # 初始化目录
    if get_rank() == 0: # 如果当前进程的等级是 0（即主进程）
        if not os.path.exists(args.checkpoints_dir):
            os.makedirs(args.checkpoints_dir)
        if not os.path.exists(args.tb_dir):
            os.makedirs(args.tb_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# 初始化分布式训练模式的函数
def init_distributed_mode(args):
    # world-size代表全局进程个数(一般一个GPU上一个进程)
    # rank代表进程的优先级也是这个进程的编号，rank=0的主机就是主要节点
    if not args.distributed: # 是否启用了分布式模式
        print('Not using distributed mode')
        return
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])  # 当前进程的等级（优先级）
        args.world_size = int(os.environ['WORLD_SIZE'])  # 表示全局进程数
        args.gpu = int(os.environ['LOCAL_RANK'])  # 当前进程在本地的GPU编号
        print("RANK and WORLD_SIZE in environ: {rank}/{wofrld_size}")
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    """
    backend str/Backend 是通信所用的后端，可以是"ncll" "gloo"或者是一个torch.distributed.Backend类（Backend.GLOO）
    init_method str 这个URL指定了如何初始化互相通信的进程
    world_size int 执行训练的所有的进程数
    rank int this进程的编号，也是其优先级
    """
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank,
        timeout=datetime.timedelta(0, 180))
    # 不同进程之间的数据同步
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        # tensor=(b,c,max_h,max_w),mask=(b,c,max_h,max_w)
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=True):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def is_dist_avail_and_initialized(): # 分布式通信是否可用和初始化
    if not dist.is_available(): 
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    # return get_rank() == 0
    return False


def collate_fn(batch):
    # 将每个样本中的数据与标签分离，将数据和标签分别组成两个列表。
    # 例如，假设一个样本是 (data, label)，其中 data 是图像数据，label 是标签，
    # 那么这一步将把所有的 data 组成一个列表，所有的 label 组成一个列表。
    batch = list(zip(*batch))
    # batch_size=3,tuple,batch[0]中为8张im，batch[1]中为8个label，batch[2]中为8个label_std
    batch[0] = nested_tensor_from_tensor_list(batch[0]) # 对批次中的图像数据进行处理

    # saliency:
    # batch[1] = nested_tensor_from_tensor_list(batch[1])
    # 将处理后的数据和标签重新组成一个元组，并返回。返回的元组中的第一个元素是处理后的图像数据，第二个元素是原始的标签数据。
    return tuple(batch) 


def _max_by_axis(the_list):
    # 找到一个batch中c，w，h最大的值
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    # tenosr_list->Tuple(im)->size=batch_size
    # 确定处理的是否为图像数据
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        # 
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            # img=(c,h,w),pad_img=(c,max_h,max_w),m=(max_h,max_w)
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack(
            [img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def load_pretrained(config, model):
    checkpoint = torch.load(config.pretrained, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            print("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            print(f"Error in loading classifier head, re-init classifier head to 0")

    model.load_state_dict(state_dict, strict=False)

    print(f"=> loaded successfully '{config.pretrained}'")

    del checkpoint
    torch.cuda.empty_cache()
