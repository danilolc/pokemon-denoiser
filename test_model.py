#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt

from load_dataset import load_dataset

pimages = load_dataset().to("cuda")
model = torch.load("model.pth")

noise = torch.randn(3, 64, 64, device="cuda") / 3

image = pimages[1][250] + noise
plt.imshow(image.cpu().permute(1, 2, 0))
plt.show()

timage = model(image)
plt.imshow(timage.cpu().detach().permute(1, 2, 0))
plt.show()