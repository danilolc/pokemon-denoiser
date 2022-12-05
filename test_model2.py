#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:47:15 2022

@author: danilo
"""

import torch
from torchvision import transforms

import matplotlib.pyplot as plt

from PIL import Image

transform = transforms.ToTensor()

model1 = torch.load("model0.5-1.pth").cpu()
model2 = torch.load("model1-1.5.pth").cpu()
model3 = torch.load("model1.5-2.pth").cpu()
model4 = torch.load("model2-4.pth").cpu()
model5 = torch.load("model4-8.pth").cpu()
model6 = torch.load("model8-1000.pth").cpu()

noise1 = torch.randn(3, 64, 64) / 0.5
noise2 = torch.randn(3, 64, 64) / 1
noise3 = torch.randn(3, 64, 64) / 1.5
noise4 = torch.randn(3, 64, 64) / 2
noise5 = torch.randn(3, 64, 64) / 4
noise6 = torch.randn(3, 64, 64) / 8

def show_images(img):
    
    img = img + noise1
    plt.imshow(img.detach().permute(1, 2, 0))
    plt.show()
    img = model1(img.unsqueeze(0)).squeeze()
    
    img = img + noise2
    plt.imshow(img.detach().permute(1, 2, 0))
    plt.show()
    img = model2(img.unsqueeze(0)).squeeze()
    
    img = img + noise3
    plt.imshow(img.detach().permute(1, 2, 0))
    plt.show()
    img = model3(img.unsqueeze(0)).squeeze()
    
    img = img + noise4
    plt.imshow(img.detach().permute(1, 2, 0))
    plt.show()
    img = model4(img.unsqueeze(0)).squeeze()
    
    img = img + noise5
    plt.imshow(img.detach().permute(1, 2, 0))
    plt.show()
    img = model5(img.unsqueeze(0)).squeeze()
    
    img = img + noise6
    plt.imshow(img.detach().permute(1, 2, 0))
    plt.show()
    img = model6(img.unsqueeze(0)).squeeze()
    
    plt.imshow(img.detach().permute(1, 2, 0))
    plt.show()
    

# In[]:
    
img1 = Image.open("pkm.png").convert('RGB')
img1 = transform(img1)

img2 = Image.open("4.png").convert('RGB')
img2 = transform(img2)

for i in torch.arange(0, 1.1, 0.1):
    show_images(img1 * (1-i) + img2 * i)
    
# In[]:

img = Image.open("4.png").convert('RGB')
img = transform(img)

show_images(img)