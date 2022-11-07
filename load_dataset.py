#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms

from PIL import Image

def load_dataset():
    
    transform = transforms.ToTensor()
    
    def load_images(path):
        ten = []
        for i in range(386):
            img = Image.open(path + f"{i+1}.png").convert('RGB')
            img = transform(img)
            ten.append(img)
        
        return torch.stack(ten)
        
    emerald = load_images("./dataset/emerald/")
    frlg = load_images("./dataset/frlg/")
    rs = load_images("./dataset/rs/")
    
    return torch.stack([emerald, frlg, rs])