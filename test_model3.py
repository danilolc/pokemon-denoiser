#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:47:15 2022

@author: danilo
"""

import torch
from torchvision import transforms
from math import exp
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from load_dataset import load_types

STEP = 0.25
VALS = np.arange(-3, 3.25, STEP)

transform = transforms.ToTensor()

types = load_types()

models = []
noises = []
for i in VALS:
    models.append( torch.load(f"model{i}-{i+0.5}.pth").cpu() ) # Trocar para STEP
    noises.append( torch.randn(3, 64, 64) / exp(i) )

def show_images(img, typ):
    
    for i, v in enumerate(VALS): # enumerate?
        img = img + noises[i]
        plt.imshow(img.detach().permute(1, 2, 0))
        plt.show()
        img = models[i](img.unsqueeze(0), types[typ].unsqueeze(0))[0]
        
    plt.imshow(img.detach().permute(1, 2, 0))
    plt.show()
    

# In[]:

img = Image.open("4.png").convert('RGB')
img = transform(img)
#img = torch.zeros(3,64,64)

show_images(img)