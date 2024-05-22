#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:47:15 2022

@author: danilo
"""

import torch
from torchvision import utils
from random import randint, random
from math import exp
import numpy as np

import matplotlib.pyplot as plt
import gc
import time

from load_dataset import *

STEP = 0.25
VALS = np.arange(-2.5, 3, STEP)

device = "cuda"

#pimages = load_dataset().to(device)
#PATH = "./"
PATH = "./models/alpha3/"

files = [PATH+f"model{i}-{i+STEP}.pt" for i in VALS]
models = [torch.jit.load(n).eval().to(device) for n in files]

def show_images(img):
    
    t = randint(0,17)
    
    for i, v in enumerate(VALS):
        gc.collect()
        noise = torch.randn(img.size(), device=device) / exp(v)
        img = img + noise
        #plt.imshow(1 - img.cpu().detach().repeat(3,1,1).permute(1,2,0).clamp(0,1))
        #plt.show()
        img = models[i](img[None], torch.tensor([t], device=device))[0]
        
    #plt.imshow(1 - img.cpu().detach().repeat(3,1,1).permute(1,2,0).clamp(0,1))
    #plt.show()
    
    #plot_image(img)
    save_image(img, f"{time.time()}.png", h=random())
    
# In[]:    

for i in range(100):
    print(i)
    #img = load_image_RGB("fenk.png").to("cuda")
    #img = img[1:] # Remove H
    img = torch.zeros(4,64,64, device="cuda")
    #img = pimages[1,randint(0,385)]
    show_images(img)
    
# Export to ONNX

# In[]

import os

for l in files:
    os.rename(l, "./models/" + ? + "/" + l)
    