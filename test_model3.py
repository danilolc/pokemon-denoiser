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

import time

from load_dataset import *

STEP = 0.25
VALS = np.arange(-1.5, 3, STEP)

pimages = load_dataset().to("cuda") # HSV
pimages = pimages[:,:,1:] # Remove H

PATH = "."
PATH = "./models/hsv1/"

files = [PATH+f"model{i}-{i+STEP}.pt" for i in VALS]
models = [torch.jit.load(n).eval() for n in files]

def show_images(img):
    
    for i, v in enumerate(VALS):
        noise = torch.randn(img.size(), device="cuda") / exp(v)
        img = img + noise
        img = models[i](img.unsqueeze(0), 0)[0]
        #plot_image(img, h=0.3)
        
    plot_image(img, h=random())
        
    #save_image(img, f"{time.time()}.png", h=random())
    
# In[]:

for i in range(50):
    img = load_image("fenk.png").to("cuda")
    img = img[1:] # Remove H
    #img = torch.ones(3,64,64, device="cuda")
    img = pimages[1,randint(0,385)]
    show_images(img)
    
# In[]

import os

for l in files:
    os.rename(l, "./models/" + ? + "/" + l)
    