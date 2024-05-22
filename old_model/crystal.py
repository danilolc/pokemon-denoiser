#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
from torch import optim, nn
from math import exp

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from random import randint

from auto_encoder import PAutoE
from load_dataset import load_types

path = "/home/danilo/Git/pokecrystal/gfx/pokemon/"

def load_image(path):
    toTensor = transforms.ToTensor()
    
    im = Image.open(path)
    w, h = im.size
    
    im = im.crop((0,0,w,w))
    
    x = (64 - w)/2
    
    im = im.crop((-x, -x, w + x, w + x))
    imt = toTensor(im)
    imt /= torch.max(imt)
    
    return imt[None]

def plot_image(im):
    plt.imshow(1 - im.cpu().detach().repeat(3,1,1).permute(1,2,0).clamp(0,1))
    plt.show()

with open('pokemon.csv', newline='') as csvfile:
    lines = csvfile.readlines()

lines = [l.split(',') for l in lines]

pokemons = [l[0] for l in lines]
#types0   = [l[1] for l in lines]
#types1   = [l[2] for l in lines]



ims = []
for p in pokemons[:251]:
    if (p != 'unown'):
        im = load_image(path + p + "/front.png")
        ims.append(im)
        
ims = torch.cat(ims).to("cuda")

STEP = 0.25
VALS = np.arange(2.75, -3, -STEP)

types = load_types().to("cuda")

for i in VALS:
    
    n1 = exp(i)
    n2 = exp(i + STEP)

    bulba = ims[0]
    noise1 = torch.randn(bulba.size(), device="cuda") / n1
    bulba = bulba + noise1
    plot_image(bulba)
    
    model = PAutoE(1,1).to("cuda")

    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=2e-2)
    
    pbar = tqdm(range(10001))
    def closure():
        optimizer.zero_grad()
        
        batch_size = 4
        batch = torch.randperm(250)[:batch_size]
    
        img = ims[batch]
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
            typ = types[0]
            timage = model(bulba[None], typ[None])[0]
            plot_image(timage)

    script = torch.jit.script(model)
    script.save(f"model{i}-{i+STEP}.pt")
    







