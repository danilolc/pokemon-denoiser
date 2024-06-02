#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import optim, nn
import time

from tqdm import tqdm

from load_dataset import load_dataset, plot_image
from torchvision.transforms import v2
from matplotlib import pyplot as plt

from UNet import ssim_loss

pos_transform = v2.Compose([
    #v2.Pad(6, 1.0),
    #v2.RandomCrop((64 + 6, 64 + 6)),
    #v2.Pad(1, 1.0),
    v2.Pad(8, 1.0),
    v2.RandomCrop((64 + 8, 64 + 8)),
    v2.RandomChannelPermutation(),
])

#https://www.researchgate.net/figure/The-Vision-Transformer-architecture-a-the-main-architecture-of-the-model-b-the_fig2_348947034
class Transformer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.mha = nn.MultiheadAttention(emb_dim, num_heads=2, batch_first=True) ##2?

        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            #nn.Dropout(0.1),
        )

    def forward(self, x):
        x_ln = self.ln1(x)
        att, _ = self.mha(x_ln, x_ln, x_ln)
        
        att = self.ln2(att + x)
        return self.mlp(att) + att
    
#https://medium.com/@14prakash/masked-autoencoders-9e0f7a4a2585
class MyMAE(nn.Module):
    def __init__(self, img_size, patch_size, emb_dim):
        super().__init__()
        assert img_size % patch_size == 0

        self.emb_dim = emb_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.masked_size = int(0.75 * self.num_patches)

        self.patch_embedding = nn.Conv2d(3, self.emb_dim, 
                                         kernel_size=patch_size, 
                                         stride=patch_size,
                                         bias=True) #?

        self.pos_embedding = nn.Parameter(torch.zeros(self.num_patches, self.emb_dim), requires_grad=False)
        
        self.encoder = nn.Sequential(*[Transformer(self.emb_dim) for i in range(8)])
        self.decoder = nn.Sequential(*[Transformer(self.emb_dim) for i in range(1)])

        self.decoder_emb_dim = self.emb_dim
        self.decoder_emb = nn.Linear(self.emb_dim, self.decoder_emb_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_emb_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.decoder_pos_emb = nn.Parameter(torch.zeros(self.num_patches, self.decoder_emb_dim))

        self.img_recov = nn.Linear(self.decoder_emb_dim, 3 * (self.patch_size ** 2), bias=True)
        

    def forward(self, x):
        bs, _, _, _ = x.shape
        device = x.device

        patches = self.patch_embedding(x)
        patches = patches.flatten(2, 3).transpose(1, 2)

        mask = torch.randperm(self.num_patches, device=device) #img size independent?
        mask = mask[:-self.masked_size]
        masked = patches[:, mask, :]

        pos_emb = self.pos_embedding[mask, :]
        tokens = masked + pos_emb[None, ...]
        features = self.encoder(tokens)

        ###### bottleneck

        tokens = self.mask_token.repeat(bs, self.num_patches, 1)
        tokens[:, mask, :] = self.decoder_emb(features)

        tokens = tokens + self.decoder_pos_emb
        features = self.decoder(tokens)

        image = self.img_recov(features)

        image = image.transpose(1,2)
        image = nn.functional.fold(image, 
                                   kernel_size=self.patch_size, 
                                   output_size=self.img_size, 
                                   stride=self.patch_size)
        
        return image

# palette [3, 16]      -> [128] 256?
# image   [16, 64, 64] -> [128, 64, 64]

# shuffle palette

def train():
    device = "cuda"

    pimages = load_dataset().to(device)

    model = MyMAE(72, 4, 64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    mse_loss = nn.MSELoss()

    for i in tqdm(range(200000), miniters=10):
        bs = 64
        source = torch.randint(0, 5, (bs,))
        batch = torch.randperm(385)[:bs]
        x0 = pimages[source, batch]
        x0 = torch.stack([pos_transform(x) for x in x0], dim=0)

        optimizer.zero_grad()

        reconstruction = model(x0)
        loss = mse_loss(reconstruction, x0) + 2.0 * ssim_loss(reconstruction, x0).mean()

        loss.backward()
        optimizer.step()

        if i % 2000 == 0:
            fig, axes = plt.subplots(1, 4, figsize=(15, 15))
            plot_image(reconstruction[0], axes[0])
            plot_image(reconstruction[1], axes[1])
            plot_image(reconstruction[2], axes[2])
            plot_image(reconstruction[3], axes[3])
            plt.show()

    torch.save(model.state_dict(), f'{int(time.time())}_mae.pt') #overfit?

if __name__ == '__main__':
    train()