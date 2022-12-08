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

#VALS = [-3.0, -2.5, -2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3.0]

STEP = 0.25
VALS = np.arange(2, 4, STEP)

plt.imshow(pimages[0][251].cpu().detach().permute(1, 2, 0))
plt.show()

types = load_types().to("cuda")

for i in VALS:
    
    n1 = exp(i)
    n2 = exp(i + STEP)
    
    model = PAutoE().to("cuda")
    loss_func = nn.L1Loss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    
    pbar = tqdm(range(50001))
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
        
        pbar.set_description("%.8f" % loss)
        
        return loss
        
    model.train()
    for j in pbar:
        optimizer.step(closure) 
        
        if j % 1000 == 0:
            source = randint(0, 384)
            image = pimages[1][source]
            image = image + torch.randn(image.size(), device="cuda") / n1
            typ = types[source]
            timage = model(image.unsqueeze(0), typ.unsqueeze(0))[0]
            plt.imshow(timage.cpu().detach().permute(1, 2, 0))
            plt.show()
    

    torch.save(model, f"model{i}-{i+0.5}.pth")