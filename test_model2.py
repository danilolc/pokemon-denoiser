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

model0 = torch.load("model0.5-1.pth").cpu()
model1 = torch.load("model1-2.pth").cpu()
model2 = torch.load("model2-4.pth").cpu()
model3 = torch.load("model4-8.pth").cpu()
model4 = torch.load("model8-1000.pth").cpu()

transform = transforms.ToTensor()

img = Image.open("4.png").convert('RGB')
img = transform(img)

plt.imshow(img.detach().permute(1, 2, 0))
plt.show()

noise = torch.randn(3, 64, 64) / 0.5
img = img + noise
img = model0(img.unsqueeze(0)).squeeze()
plt.imshow(img.detach().permute(1, 2, 0))
plt.show()

noise = torch.randn(3, 64, 64) / 1
img = img + noise
img = model1(img.unsqueeze(0)).squeeze()
plt.imshow(img.detach().permute(1, 2, 0))
plt.show()

noise = torch.randn(3, 64, 64) / 2
img = img + noise
img = model2(img.unsqueeze(0)).squeeze()
plt.imshow(img.detach().permute(1, 2, 0))
plt.show()

noise = torch.randn(3, 64, 64) / 4
img = img + noise
img = model3(img.unsqueeze(0)).squeeze()
plt.imshow(img.detach().permute(1, 2, 0))
plt.show()

noise = torch.randn(3, 64, 64) / 8
img = img + noise
img = model4(img.unsqueeze(0)).squeeze()
plt.imshow(img.detach().permute(1, 2, 0))
plt.show()