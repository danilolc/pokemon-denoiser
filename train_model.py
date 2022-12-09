#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import optim, nn
from math import exp

import numpy as np

from tqdm import tqdm
from random import randint
import matplotlib.pyplot as plt

from load_dataset import load_dataset, load_types
from auto_encoder import PAutoE

pimages = load_dataset().to("cuda")
pimages[:,:,0] *= pimages[:,:,3]
pimages[:,:,1] *= pimages[:,:,3]
pimages[:,:,2] *= pimages[:,:,3]

#VALS = [-3.0, -2.5, -2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3.0]

STEP = 0.25
VALS = np.arange(3, -3, -STEP)

plt.imshow(pimages[0][251].cpu().detach().permute(1, 2, 0))
plt.show()

types = load_types().to("cuda")

def plot_image(im):
    alpha = 1 - im[3]
    im = im[0:3] + alpha
    plt.imshow(im.clamp(0,1).cpu().detach().permute(1, 2, 0))
    plt.show()

for i in VALS:
    
    n1 = exp(i)
    n2 = exp(i + STEP)
    
    model = PAutoE().to("cuda")
    loss_func = nn.L1Loss()
    #loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=2e-1)
    
    pbar = tqdm(range(20001))
    def closure():
        optimizer.zero_grad()
    
        source = randint(0, 2)
    
        batch_size = 8
        batch = torch.randperm(385)[:batch_size]
    
        img = pimages[source][batch]
        typ = types[batch]
    
        noise1 = torch.randn(img.size(), device="cuda") / n1
        noise2 = torch.randn(img.size(), device="cuda") / n2
        
        img1 = noise1 + img
        img2 = noise2 + img
        
        pred = model(img1, typ)
    
        loss = loss_func(pred, img2)
        loss.backward()
        
        pbar.set_description(f"{i} {i+STEP} \t %.8f" % loss)
        
        return loss
        
    model.train()
    for j in pbar:
        optimizer.step(closure) 
        
        if j % 1000 == 0:
            source = randint(0, 384)
            image = pimages[1][source]
            plot_image(image)
            image = image + torch.randn(image.size(), device="cuda") / n1
            plot_image(image)
            typ = types[source]
            timage = model(image.unsqueeze(0), typ.unsqueeze(0))[0]
            plot_image(timage)
    

    torch.save(model, f"model{i}-{i+STEP}.pth")