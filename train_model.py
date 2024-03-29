#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import optim, nn
from math import exp

import numpy as np

from tqdm import tqdm
from random import randint

from load_dataset import load_dataset, load_types, plot_image
from auto_encoder import PAutoE

pimages = load_dataset().to("cuda")

STEP = 0.25
VALS = np.arange(2.75, -3, -STEP)

types = load_types().to("cuda")

for i in VALS:
    
    n1 = exp(i)
    n2 = exp(i + STEP)
    
    model = PAutoE(4,4).to("cuda")
    #loss_func = nn.L1Loss()
    loss_func = nn.MSELoss()
    
    optimizer = optim.SGD(model.parameters(), lr=3e-2)
    #optimizer = optim.LBFGS(model.parameters(), history_size=20)
    
    pbar = tqdm(range(30001))
    def closure():
        optimizer.zero_grad()
    
        source = randint(0, 2)
    
        batch_size = 4
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
        
        if j % 5000 == 0:
            print("A")
        
        if j % 1000 == 0:
            source = randint(0, 384)
            image = pimages[1][source]
            plot_image(image, h=0)
            image = image + torch.randn(image.size(), device="cuda") / n1
            plot_image(image, h=0)
            typ = types[source]
            timage = model(image[None], typ[None])[0]
            plot_image(timage, h=0)

    script = torch.jit.script(model)
    script.save(f"model{i}-{i+STEP}.pt")
    
    