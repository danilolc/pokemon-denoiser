#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import optim, nn
from math import exp

from tqdm import tqdm
from random import randint
import matplotlib.pyplot as plt

from load_dataset import load_dataset
from auto_encoder import PAutoE

pimages = load_dataset().to("cuda")

for i in [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]:
    
    n1 = exp(i)
    n2 = exp(i + 0.5)
    
    model = PAutoE().to("cuda")
    loss_func = nn.L1Loss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=5e-1)
    
    pbar = tqdm(range(100001))
    def closure():
        optimizer.zero_grad()
    
        source = randint(0, 2)
    
        batch_size = 4
        batch = torch.randperm(385)[:batch_size]
    
        img = pimages[source][batch]
    
        noise1 = torch.randn(batch_size, 3, 64, 64, device="cuda") / n1
        noise2 = torch.randn(batch_size, 3, 64, 64, device="cuda") / n2
        
        img1 = noise1 + img
        img2 = noise2 + img
        
        pred = model(img1)
    
        loss = loss_func(pred, img2)
        loss.backward()
        
        pbar.set_description("%.8f" % loss)
        
        return loss
        
    model.train()
    for j in pbar:
        optimizer.step(closure) 
        
        if j % 1000 == 0:
            source = randint(0, 384)
            image = pimages[1][source] + torch.randn(3, 64, 64, device="cuda") / n1
            timage = model(image.unsqueeze(0)).squeeze()
            plt.imshow(timage.cpu().detach().permute(1, 2, 0))
            plt.show()
    

    torch.save(model, f"model{i}-{i+0.5}.pth")