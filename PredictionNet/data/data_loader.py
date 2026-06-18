"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
from PIL import Image
from PIL import ImageFile
import pandas as pd
import csv
import util.misc as utils
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from torchvision import transforms as T
from nltk.corpus import stopwords
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler, BatchSampler
from tqdm import tqdm
import cv2
import clip
from models import longclip
import scipy.io
from torch.utils.data import Subset, DataLoader
import random



stoplist = stopwords.words('english')
dist = torch.arange(1, 11).float()

def transform(status):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if status == 'train':
        return T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            normalize
        ])
    elif status == 'test':
        return T.Compose([
            T.Resize((224,224)),
            normalize
        ])
        
    elif status == 'cam':
        Normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return T.Compose([
            T.Resize((224,224)),
            normalize
        ])
    else:
        Normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return T.Compose([
            T.Resize(256),
            normalize
        ])


def getstr(status):
    if status == 'train':
        annotations_str = "trainScore"
        imgname_str = "trainNameList"
    elif status == 'test':
        annotations_str = "testScore"
        imgname_str = "testNameList"
    return annotations_str, imgname_str


class MyDataset(Dataset):

    def __init__(self, csv_file, root_dir, status):
        super(MyDataset, self).__init__()
        _ , self.preprocess = clip.load("ViT-B/16")
        self.annotations = np.loadtxt(csv_file, 'str')  # 'int'
        # print(self.annotations)
        self.root_dir = root_dir
        self.transform = transform(status)
        self.keyword_path = '/data/csl/llama-main/text/target.txt'
        self.status = status
        self.stoplist = stoplist

        with open(self.keyword_path) as f:
            key_texts = f.readlines()
        self.key_texts = key_texts
       
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        im_name = str(int(self.annotations[index, 0])) + '.jpg'
        img_name = os.path.join(self.root_dir, im_name)

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(img_name)
        image = image.convert("RGB")
        img = self.transform(image)
        # img = self.preprocess(image)


        annotations = self.annotations[index, 1:11]  # 2-11
        annotations = annotations.astype('float').reshape(-1, 1)  # 1lie
        # text = self.texts[index]
        # text = text[:20]
        # text_token = longclip.tokenize(text)

        # if self.status == 'train':
        #     key_texts = self.key_texts[index]
        #     key_texts_token = longclip.tokenize(key_texts)
        #     return img, annotations, text_token, key_texts_token

        # elif self.status == 'test':
        #     return img, annotations, text_token
        return img, annotations


    
    
# ------------------------------------------------------AADB-dataset--------------------------------------------------------
class AADBDataset(Dataset):
   

    def __init__(self, csv_file, root_dir, status):
        super(AADBDataset, self).__init__()
        self.file = scipy.io.loadmat(csv_file) # label文件
        self.annotations_str, self.imgname_str = getstr(status)
        self.annotations_data = self.file[self.annotations_str]
        self.imgname_data = self.file[self.imgname_str]
        self.root_dir = root_dir # 图片文件
        self.transform = transform(status)

     

    def __len__(self):
        return self.annotations_data.shape[1]
    

    def __getitem__(self, index):
        # 获取图片 
        im_name = self.imgname_data[:, index]
        img_name = os.path.join(self.root_dir, im_name[0].item())

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(img_name)
        image = image.convert("RGB")
        img = self.transform(image)


        # 获取分数标签
        annotations = self.annotations_data[:, index]  # 2-11


        return img, annotations 




def build_loader(config):
    train_dataset = MyDataset(csv_file=config.train_csv_file, root_dir=config.train_img_path, status='train')
    test_dataset = MyDataset(csv_file=config.test_csv_file, root_dir=config.train_img_path, status='test')


    if config.distributed:
        sampler_train = DistributedSampler(train_dataset) # 用于训练集的分布式采样器
        sampler_test = DistributedSampler(test_dataset, shuffle=False) # 用于测试集的分布式采样器
    else:  # 每个进程都会独立处理数据集的一个子集，而不需要进行数据集的分割和分配。
        sampler_train = RandomSampler(train_dataset)
        sampler_test = SequentialSampler(test_dataset)

    batch_sampler_train = BatchSampler(
        sampler_train, config.train_batch_size, drop_last=True)
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler_train,  num_workers=config.num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=config.test_batch_size,  sampler=sampler_test,
        drop_last=False)


    return train_loader, test_loader





