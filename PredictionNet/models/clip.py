import torch
import clip
from PIL import Image


def Clip(args, device):
    model, preprocess = clip.load("ViT-B/16", device=device, download_root="/data/csl/CLIP1/clipmodel")
    return model, preprocess
    