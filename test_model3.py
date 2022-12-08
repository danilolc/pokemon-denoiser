#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:47:15 2022

@author: danilo
"""

import torch
from torchvision import transforms
from random import randint
from math import exp
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from load_dataset import load_types

STEP = 0.5
VALS = np.arange(-1, 3, STEP)
transform = transforms.ToTensor()

types = load_types()

models = []
noises = []
for i in VALS:
    models.append( torch.load(f"model{i}-{i+STEP}.pth").cpu() )
    noises.append( torch.randn(3, 64, 64) / exp(i) )

def show_images(img, typ=0):
    
    #for i in [8,9,10]:
    #    noises[i] = torch.randn(3, 64, 64) / exp(VALS[i])
    
    for i, v in enumerate(VALS): # enumerate?
        img = img + noises[i]
        #plt.imshow(img.detach().permute(1, 2, 0))
        #plt.show()
        img = models[i](img.unsqueeze(0), types[typ].unsqueeze(0))[0]
        
    plt.imshow(img.detach().permute(1, 2, 0))
    plt.show()
    
# In[]:

img = Image.open("pkm.png").convert('RGB')
img = transform(img)
#img = torch.zeros(3,64,64)

plt.imshow(img.detach().permute(1, 2, 0))
plt.show()

typ = randint(0,350)

show_images(img, typ=typ)