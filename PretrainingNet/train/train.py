import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
#from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
#the original concat_all_gather is abandoned because of no gradient backward
from utils import is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

import sys
sys.path.append("..")

from torch.utils.data import DataLoader
from data_loader import AVADataset
from model import aesclip

from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr
import argparse
import subprocess
import collections
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from torch.cuda.amp import GradScaler
from thop import profile
import time
# import warnings
# warnings.filterwarnings("ignore")


class CLIP_Clean_Train():
    def __init__(self, rank,local_rank,args):
        self.rank=rank
        self.local_rank = local_rank
        self.base_model = args.base_model
        self.model, _ = aesclip.load_from_clip("ViT-B/16")
        # add
        self.model.train()
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)  # 可学习的参数
        self.model = self.model.cuda()
        
        self.batch_size = args.batch_size
        self.num_epoch = args.epochs
        self.lr = args.lr # 1e-6
        self.weight_decay = args.weight_decay # 1e-2
        self.warmup_length = args.warmup_length # 200
        # log设置
        if args.exp_name == "auto":
            self.logdir = f"aesclip/lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_logs={args.log_scale}_64xb"
        else:
            self.logdir = args.exp_name
        self.ckptdir = self.logdir + "/ckpt/"
        os.makedirs(self.ckptdir, exist_ok=True)
        # self.writer = SummaryWriter(self.logdir)

        

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
           
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler =GradScaler()


    def train_epoch(self, dataloader, epoch, start_iter=0):

        # 计算训练时间
        # 记录epoch开始时间
        epoch_start_time = time.time()

        # 用于累积长文本和短文本的损失
        running_loss = 0.0
        running_loss_short = 0.0
        #rank = torch.distributed.get_rank() 
        num_batches_per_epoch = len(dataloader)

        # 用于记录批次处理时间
        batch_times = []

        for i, (images, texts, short_text) in enumerate(tqdm(dataloader, disable=(self.rank != 0))):
            # 记录批次开始时间
            batch_start_time = time.time()

            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue
            #images = images.cuda()
            images_short = images.clone()
            texts = aesclip.tokenize(texts, truncate=True).cuda()
            short_text = aesclip.tokenize(short_text, truncate=True).cuda()
            
            self.scheduler(step)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss_long,loss_short = self.model(images, short_text, texts, self.rank)


                loss = loss_long + loss_short

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()


            # 计算批次处理时间
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            batch_times.append(batch_duration)
        
        # ToDo: revise the report part
            running_loss += loss.item()
            running_loss_short += loss_short.item()
            batch_num = i
            
            loss = running_loss
            running_loss = 0.0

            loss_short = running_loss_short
            running_loss_short = 0.0

            loss = torch.tensor(loss).cuda()
            dist.all_reduce(loss)
            loss = loss.item() / torch.distributed.get_world_size()

            loss_short = torch.tensor(loss_short).cuda()
            dist.all_reduce(loss_short)
            loss_short = loss_short.item() / torch.distributed.get_world_size()

            rank = torch.distributed.get_rank()
            if step % 100 == 0:
                if rank == 0:
                    # self.writer.add_scalar("hyper/lr", self.optimizer.param_groups[0]['lr'], step)
                    # self.writer.add_scalar("logit_scale/train", self.model.logit_scale.item(), step)
                    print("=====================================")
                    print(f"train lr step {step}: {self.optimizer.param_groups[0]['lr']}")
                    # print(f"train logit_scale step {step}: {self.model.logit_scale.item()}")
                    print(f"train loss step {step}: {loss}")
                    print(f"train loss short step {step}: {loss_short}")
                    print("=====================================")
                    # self.writer.add_scalar("Loss/train", loss + loss_short, step)
                    
                    # with torch.no_grad():
                    #     self.model.eval()
                    #     self.test(epoch = epoch)
                    #     self.model.train()
                    # 重置批次时间记录
                    batch_times = []

            # 计算epoch总时间
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # 转换为分钟和秒
        mins, secs = divmod(epoch_duration, 60)
        hrs, mins = divmod(mins, 60)
        
        rank = torch.distributed.get_rank()
        if rank == 0:
            print(f"Epoch {epoch} completed in {int(hrs):02d}h:{int(mins):02d}m:{secs:05.2f}s")

        # return running_loss / batch_num

    
    def train(self, resume=False, warmup_length=200):

        AVAtrainset = AVADataset()
        AVAtrain_sampler = DistributedSampler(dataset=AVAtrainset, shuffle=True)
        AVAtrain_loader = torch.utils.data.DataLoader(AVAtrainset, batch_size=self.batch_size, sampler=AVAtrain_sampler, num_workers=32, pin_memory=True)

        # 返回一个内部函数 _lr_adjuster，用于根据当前的训练步数调整学习率（全局函数）
        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=self.num_epoch * len(AVAtrain_loader))
        start_epoch = 0
        resume_iter = 0
        
        for epoch in range(start_epoch, self.num_epoch):
            self.train_epoch(AVAtrain_loader, epoch, start_iter=resume_iter)
            
           
            if self.rank == 0:
                name = "aesclip.pt"
                now = datetime.now()
                formatted_date = now.strftime("%m-%d--%H_%M_%S_")
                #torch.distributed.barrier()
                # torch.save(self.model.module.state_dict(), "/data/csl/aes-CLIP-main/checkpoints/rebuttal/abtion_normal/"+str(epoch)+name)
            # print("=====================================")
            # print(f"loss after training epoch: {epoch}")
            # print("=====================================")

            # if epoch == self.num_epoch - 1:
            #     if self.base_model == "ViT-B/16":
            #         name = 'aesclip-B.pt'
            #     elif self.base_model == "ViT-L/14":
            #         name = 'aesclip-L.pt'
            #     else:
            #         name = "aesclip-others.pt"

            #     torch.save(self.model.module.state_dict(), name)

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "28522"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(device=f'cuda:{rank % num_gpus}')
    return rank, rank % num_gpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=1e-6, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--base_model", default="ViT-B/16", help="CLIP Base Model")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size per gpu."#112
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--resume",
        default=False,
        action='store_true',
        help="resume training from checkpoint."
    )
    parser.add_argument("--download-root", default=None, help="CLIP Base Model download root")
    args = parser.parse_args()
    rank,local_rank = setup_distributed(port=28501)
    print("DDP Done")

    trainer = CLIP_Clean_Train(
        rank=rank,
        local_rank=local_rank, 
        args=args
        )
    trainer.train(resume=args.resume, warmup_length=args.warmup_length)
