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

STEP = 0.25
VALS = np.arange(-1.0, 4, STEP)

transform = transforms.ToTensor()

models = []
noises = []
for i in VALS:
    models.append( torch.load(f"model{i}-{i+0.5}.pth").cpu() )
    noises.append( torch.randn(1, 64, 64) / exp(i) )

def show_images(img):
    
    for i in range(len(VALS)): # enumerate?
        img = img + noises[i]
        plt.imshow(img.detach().permute(1, 2, 0))
        plt.show()
        img = models[i](img.unsqueeze(0))[0]
        
        
    plt.imshow(img.detach().permute(1, 2, 0))
    plt.show()
    

# In[]:

#img = Image.open("fenk2.png").convert('RGB')
#img = transform(img)
img = torch.zeros(1,64,64) + 0.2

show_images(img)