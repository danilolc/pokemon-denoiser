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

#VALS = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
VALS = np.arange(-3, 3, 0.25)

transform = transforms.ToTensor()

models = []
noises = []
for i in VALS:
    models.append( torch.load(f"model{i}-{i+0.5}.pth").cpu() )
    noises.append( torch.randn(3, 64, 64) / exp(i) )

def show_images(img):
    
    for i in range(len(VALS)): # enumerate?
        img = img + noises[i]
        plt.imshow(img.detach().permute(1, 2, 0))
        plt.show()
        img = models[i](img.unsqueeze(0)).squeeze()

        
    plt.imshow(img.detach().permute(1, 2, 0))
    plt.show()
    

# In[]:

#img = Image.open("fenk2.png").convert('RGB')
#img = transform(img)
img = torch.zeros(3,64,64)

show_images(img)