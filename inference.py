# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import argparse
import os
import numpy as np
# from torchsummaryX import summary


from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from model.Transformers.VIT.mae import VisionTransfromers as VIT
from scipy.special import softmax

from torch.cuda.amp import autocast as autocast

import glob


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='pretrained_mae_ckpts/vit-mae_losses_0.20102281799793242.pth', type=str)
parser.add_argument('--test_file', default='/workspace/data/food/classification/KFOOD201.classification/val', type=str)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--patch_size', default=16, type=int)
args = parser.parse_args()


class TestDataset(Dataset):
    def __init__(self, data_file):
        super(TestDataset, self).__init__()
        # self.data_list = [x.strip() for x in open(data_file).readlines()]
        self.image_file = glob.glob(data_file + '/*/*.jpg')
        self.data_list = [i+','+i.split('/')[-2] for i in self.image_file]
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.data_aug = transforms.Compose([
            transforms.Resize(args.crop_size, interpolation=3),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])

    def __getitem__(self, item):
        line = self.data_list[item]
        image_path = line.split(',')[0]
        image_label = int(line.split(',')[1])
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.data_aug(image)
        return image, image_label 

    def __len__(self):
        return len(self.data_list)
    

if __name__ == '__main__':
    args = parser.parse_args()
    test_file = args.test_file
    dataset = TestDataset(test_file)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=False,
        num_workers=32,
        drop_last=False
    )
    # vit-base./16 
    model = VIT(
        img_size = args.crop_size,
        patch_size = args.patch_size,
        embed_dim = 768,
        depth = 12,
        num_heads = 12,
        num_classes = 1000
    )

    ckpt_file = args.ckpt

    # state_dict = torch.load(ckpt_file, map_location="cpu")
    # if "state_dict" in state_dict:
    #     state_dict = state_dict["state_dict"]
    # model.load_state_dict(state_dict, strict=True)
    model._load_mae_pretrain(args.ckpt)
    print("Load the imagenet pretrain!!")
    # model = convert_syncbn_model(model)
    
    print(model)
    print(len(dataset))
    # model.half()
    model.eval()
    model.cuda()
    
    num = 0
    start_time = time.time()
    count = 0
    for batch_idx, data in enumerate(dataloader):
        count += 1
        image, label = data[0], data[1]
        
        image = image.cuda()
        label = label.cuda()
        with torch.no_grad():
            with autocast():
                output = model(image)
            prob = torch.softmax(output, 1)
            _, pred = prob.topk(k=1, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(label.view(1, -1).expand_as(pred))
            correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)

            num += correct_k 
            
            print(f"processing {batch_idx}/{len(dataloader)}!!!!")
    
    total_time = time.time() - start_time
    print("time is ", total_time)
    print("batch time average is : ", total_time / count)
    print(num.data.item() / len(dataset))            