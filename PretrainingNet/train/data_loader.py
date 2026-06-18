"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
from PIL import Image
from PIL import ImageFile
import pandas as pd
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from torchvision import transforms as T
from nltk.corpus import stopwords
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler, BatchSampler
from tqdm import tqdm
import cv2
import clip
from model import aesclip
import scipy.io



# torch.multiprocessing.set_sharing_strategy('file_system')
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


class AVADataset(Dataset):
    def __init__(self):
        super(AVADataset, self).__init__()

        self.root_dir = '/data/images'
        self.keyword_path = "/data/Aessumary.txt"
        self.text_path = "/data/AVA-comments.txt"
        self.HaveID = "/data/Aessumary_haveID.txt"
        
        with open(self.text_path) as f:
            long_texts = f.readlines()
        self.long_texts = long_texts

        with open(self.keyword_path) as f:
            short_texts = f.readlines()
        self.short_texts = short_texts

        # 读取图片名
        with open(self.HaveID, 'r') as f:
            lines = f.readlines()
            self.image_names = [line.split(',', 1)[0].strip() for line in lines if line.strip()]  # 取逗号前的jpg名
            
        
       
        _ , self.preprocess = clip.load("ViT-B/16")
       
    def __len__(self):
        return len(self.long_texts)

    def __getitem__(self, index):
        im_name = self.image_names[index]
        img_name = os.path.join(self.root_dir, im_name)

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(img_name)
        image = image.convert("RGB")
        image_tensor = self.preprocess(image)

        
        long_texts = self.long_texts[index][1:]
        long_texts_word = long_texts.split()
        if len(long_texts_word) > 77:
            long_texts = long_texts[:76]
        
        short_texts = self.short_texts[index]
        
        return image_tensor, long_texts, short_texts


