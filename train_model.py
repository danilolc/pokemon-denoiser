#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
#loss_func = nn.MSELoss()
loss_func = nn.L1Loss(reduction='mean')

#optimizer = optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1e-8)
#optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=1e-4)
optimizer = optim.SGD(model.parameters(), lr=1e-1)

noise = torch.randn(3, 64, 64, device="cuda") / 3

# Plot treecko
image = pimages[1][251] + noise
plt.imshow(image.cpu().permute(1, 2, 0))
plt.show()

model.train()
for i in tqdm(range(20000)):
    optimizer.zero_grad()

    source = randint(0, 2)
    
    batch_size = 4
    batch = torch.randperm(385)[:batch_size]

    img = pimages[source][batch]

    noise = torch.randn(batch_size, 3, 64, 64, device="cuda") / 3
    pred = model(img + noise)
    
    loss = loss_func(pred, img)
    loss.backward()
    
    optimizer.step()
    
    if i % 1000 == 0:
        timage = model(image)
        plt.imshow(timage.cpu().detach().permute(1, 2, 0))
        plt.show()
    

torch.save(model, "model.pth")