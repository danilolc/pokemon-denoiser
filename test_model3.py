#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:47:15 2022

@author: danilo
"""

import torch
from torchvision import transforms, utils
from random import randint
from math import exp
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import time

from load_dataset import load_types, load_dataset

pimages = load_dataset()
pimages[:,:,0] *= pimages[:,:,3]
pimages[:,:,1] *= pimages[:,:,3]
pimages[:,:,2] *= pimages[:,:,3]

STEP = 0.25
#VALS = np.arange(0.25, 3, STEP)
VALS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
transform = transforms.ToTensor()

types = load_types()

models = []
noises = []
for i in VALS:
    models.append( torch.load(f"model{i}-{i+STEP}.pth").cpu() )
    noises.append( torch.randn(4, 64, 64) / exp(i) )

def show_images(img, typ=0):
    
    #for i in [8,9,10]:
    #    noises[i] = torch.randn(3, 64, 64) / exp(VALS[i])
    
    for i, v in enumerate(VALS): # enumerate?
        img = img + noises[i]
        img = models[i](img.unsqueeze(0), types[typ].unsqueeze(0))[0]
        #plt.imshow(img.detach().permute(1, 2, 0))
        #plt.show() 
        
    utils.save_image(img, f"{time.time()}.png")
    plt.imshow(img.detach().permute(1, 2, 0))
    plt.show()
    
# In[]:

#img = Image.open("fenk.png").convert('RGB')
#img = transform(img)
img = pimages[1,260]

#plt.imshow(img.detach().permute(1, 2, 0))
#plt.show()

typ = randint(0,350)

show_images(img, typ=typ)