#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# model 0.5 - 1
# model 1 - 2
# model 2 - 4
# model 4 - 8
# model 8 - 16
# model 16 - 1000


# Var(X+N)) / (1 + Var(N) / Var(X))  = Var(X)

n1 = 0.5
n2 = 1

import torch
from torch import optim, nn

from tqdm import tqdm
from random import randint
import matplotlib.pyplot as plt

from load_dataset import load_dataset
from auto_encoder import PAutoE

pimages = load_dataset().to("cuda")
model = PAutoE().to("cuda")

#loss_func = nn.CrossEntropyLoss()
#loss_func = nn.MSELoss(reduction='mean')
loss_func = nn.L1Loss(reduction='mean')

optimizer = optim.SGD(model.parameters(), lr=1e-1)
#optimizer = optim.LBFGS(model.parameters(), history_size=100)

noise = torch.randn(3, 64, 64, device="cuda") / n1

# Plot treecko
image = torch.clamp(pimages[1][251] + noise, min=0, max=1)
plt.imshow(image.cpu().permute(1, 2, 0))
plt.show()

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
for i in pbar:
    optimizer.step(closure) 
    
    if i % 100 == 0:
        source = randint(0, 250)
        image = pimages[1][source] + torch.randn(3, 64, 64, device="cuda") / n1
        timage = model(image.unsqueeze(0)).squeeze()
        plt.imshow(timage.cpu().detach().permute(1, 2, 0))
        plt.show()
    

torch.save(model, f"model{n1}-{n2}.pth")