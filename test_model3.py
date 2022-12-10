#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:47:15 2022

@author: danilo
"""

import torch
from random import randint
from tqdm import tqdm
from math import exp
import numpy as np

from random import randint, random


from PIL import Image

import time

from load_dataset import load_types, load_dataset, plot_image, save_image

pimages = load_dataset().to("cuda") # HSV
pimages = pimages[:,:,1:] # Remove H

STEP = 0.25
VALS = np.arange(-2.75, 3, STEP)

models = []
for i in VALS:
    models.append( torch.load(f"model{i}-{i+STEP}.pth") )


def show_images(img):
    
    for i, v in enumerate(VALS):
        noise = torch.randn(3, 64, 64, device="cuda") / exp(v)
        img = img + noise
        img = models[i](img.unsqueeze(0), 0)[0]
        #plot_image(img)
        
    save_image(img, f"testHSV/{time.time()}.png")
    plot_image(img)
    
# In[]:

#img = Image.open("fenk.png").convert('RGBA')
#img = transform(img)
#img[0] *= img[3]
#img[1] *= img[3]
#img[2] *= img[3]


#img = torch.zeros(4,64,64)

for i in tqdm(range(1000)):
    img = pimages[1,randint(0,385)]
    #plot_image(img)
    show_images(img)
    
    
    
    
    
# In[]

import os

L = [f"model{i}-{i+STEP}.pth" for i in VALS]

for l in L:
    os.rename(l, "./models/" + ? + "/" + l)
    