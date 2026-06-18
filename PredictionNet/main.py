import time
import datetime
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from models import build_model
from data import build_loader
from util import build_scheduler, build_optimizer, create_logger, IAALoss
import util.misc as utils
import sys
import clip
from models import models
from models import model_longclip

from models.aesmodel import AESMODEL


try:
    from apex import amp 
except:
    amp = None


def run(args):
    utils.init_distributed_mode(args) # 初始化分布式训练模式的函数
    utils.init_dir(args) # 初始化目录
    utils.init_seed(args) # 设置随机种子
    utils.init_lr(args) # 初始化学习率
    torch.utils.backcompat.broadcast_warning.enabled = True

   
    logger = create_logger(output_dir=args.log_dir, dist_rank=utils.get_rank(), name=f"{args.model_name}")

    torch.cuda.set_device(0)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Creating model:{args.model_type}/{args.model_name}") 
    
    state_dict = torch.load("Pretrained.pt", map_location="cpu")
    # clip_model, __ = clip.load("ViT-B/16", device=device)
    # clip_state_dict = clip_model.state_dict()
    clip_model = model_longclip.build_model(state_dict or model.state_dict(), load_from_clip = False).to(device)
    model = AESMODEL(clip_model, device)
    model.lock_clip()
    model.to(device)
   
    
    train_loader, test_loader = build_loader(args)

    # 确保模型的参数都是“float”类型!!!
    for name, parm in model.named_parameters():
        if parm.type() == 'torch.cuda.DoubleTensor':
            parm.data = parm.float()  

    optimizer = build_optimizer(args, model)

    if args.opt_level != 'O0' and amp != None:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], broadcast_buffers=False, output_device=args.gpu)
        model_without_ddp = model.module
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad) # 统计需要梯度更新的参数的数量
    logger.info(f"number of params: {n_parameters}")

    lr_scheduler = build_scheduler(args, optimizer, len(train_loader)) # 学习率调度器，动态调整学习率
    # lr_scheduler = scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=1e-6)

    best_val_criterion, best_epoch = -100, -1
    # 从先前保存的检查点或断点处继续模型的训练
    if args.resume:
        best_val_criterion, best_epoch = utils.load_checkpoints(
            args, model_without_ddp, optimizer, lr_scheduler, logger)

    if args.evaluate:
        best_val_criterion, best_epoch = utils.load_checkpoints(
            args, model_without_ddp, optimizer, lr_scheduler, logger)


        utils.validate(args, test_loader, model, device, logger)



        if args.distributed:
            dist.destroy_process_group()
        return

    # criterion = IAALoss(loss_type=args.loss_type)

    current_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    writer = None
    if utils.get_rank() == 0:
        writer = SummaryWriter(
            log_dir='{}/{}-{}'.format(args.tb_dir, args.format_str, current_time))

    logger.info('Start Training')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        

       
        # 开始训练
        utils.train_one_epoch(args, model, train_loader,
                              optimizer, epoch, lr_scheduler, device, writer, logger)  # train
        

        # if epoch == 25:
        #     # 训完保存当前轮数的模型
        #     utils.save_checkpoints(
        #                 args, epoch, model_without_ddp, optimizer, lr_scheduler, logger)
        if args.test_in_train:
            acc, srocc, plcc, mse = utils.validate(
                args, test_loader, model, device, logger)
            val_criterions = {'ACC': acc, 'SROCC': srocc, 'PLCC': plcc, 'MSE': mse}


            if writer is not None:
                utils.writer_add_scalar(writer, 'val', args.dataset,
                                        val_criterions, epoch)
            val_criterion = abs(val_criterions[args.val_criterion])

            if val_criterion > best_val_criterion:
                # if utils.get_rank() == 0:
                #     utils.save_checkpoints(
                #         args, epoch, model_without_ddp, optimizer, lr_scheduler, logger, val_criterion, update=True)
                best_val_criterion = val_criterion
                best_epoch = epoch
                logger.info(
                    f'Save current best model @best_val_criterion ({args.val_criterion}): {best_val_criterion:.3f} @epoch: {best_epoch}')
            else:
                logger.info(
                    f'Model is not updated @val_criterion ({args.val_criterion}): {val_criterion:.3f} @epoch: {epoch}')
        else:
            if utils.get_rank() == 0:
                utils.save_checkpoints(
                    args, epoch, model_without_ddp, optimizer, lr_scheduler, logger)
        

        if args.lr_scheduler_name == 'iaatrstep':
            lr_scheduler.step(epoch)
        #     utils.save_checkpoints(
        #             args, epoch, model_without_ddp, optimizer, lr_scheduler, logger)



    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # logger.info('Training time {}'.format(total_time_str))

    # start_time = time.time()
    # if writer is not None:
    #     writer.close()
    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # logger.info('Testing time {}'.format(total_time_str))
    # if args.distributed:
    #     dist.destroy_process_group()


if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--distributed', type=bool, default=False)
    # CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node 2 main.py


    # GPU2:
    parser.add_argument('--train_img_path', type=str, default='/data/images')
    parser.add_argument('--test_img_path', type=str, default='/data/images')

    parser.add_argument('--train_csv_file', type=str,
                        default="/data/train_dis.txt")
    parser.add_argument('--test_csv_file', type=str,
                        default="/data/test_dis.txt") 
    

    parser.add_argument('--warmup_lr', type=float, default=1e-7) #1e-7
    parser.add_argument('--min_lr', type=float, default=1e-6) #1e-6
    parser.add_argument('--lr', type=float, default=1e-6) #1e-5
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=15)
    parser.add_argument('--train_batch_size', type=int, default=16)  # 64   128
    parser.add_argument('--val_batch_size', type=int, default=16)  # 64   128
    parser.add_argument('--test_batch_size', type=int, default=1)  # 32
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)

    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--decay_epochs', type=int, default=2)

    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default="0,1")  # None
    parser.add_argument('--warm_start', type=bool, default=False)  # False
    parser.add_argument('--warm_start_epoch', type=int, default=1)  # 0
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--save_fig', type=bool, default=True)

    # save_path:
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/train1")
    parser.add_argument("--tb_dir", type=str, default="runs/train1")
    parser.add_argument("--log_dir", type=str, default="logs/train1")

    parser.add_argument('--randomness', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=88888)

    parser.add_argument("--model_name", type=str, default='M2')  # logger
    parser.add_argument("--model", type=str, default='CLIP_VITB16')  # logger

    parser.add_argument("--model_type", type=str, default='clip')

    parser.add_argument("--lr_scheduler_name", type=str, default="iaatrstep")
    parser.add_argument('--lr_drop', type=int, default=10)  # 10
    parser.add_argument('--gamma', type=float, default=0.8) # 0.9


    parser.add_argument("--opt_level", type=str, default="O1")
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--test_in_train', type=bool, default=True)
    parser.add_argument("--loss_type", type=str, default="js")
    parser.add_argument("--trained_model_file", type=str, default="")
    parser.add_argument("--format_str", type=str, default="")
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--train_accumulation_steps', type=int, default=0)
    parser.add_argument('--clip_grad', type=bool, default=False)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--print_freq_con', type=int, default=20)
    parser.add_argument("--dataset", type=str, default="AVA")
    parser.add_argument("--val_criterion", type=str, default="ACC")  #ACC
    parser.add_argument('--world_size', type=int, default=-1)
    parser.add_argument('--rank', type=int, default=-1)
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--dist_url", type=str, default="env://")
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0001) # 0.00001
    parser.add_argument("--local_rank", type=int, required=False,
                        help='local rank for DistributedDataParallel')

    config = parser.parse_args()
    config.format_str = '{}-{}'.format(config.model_name, config.model_type)
    config.trained_model_file = os.path.join(config.checkpoints_dir, config.format_str)


    run(config)
