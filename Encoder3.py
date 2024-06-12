#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import optim, nn
import time

from tqdm import tqdm

from load_dataset import load_dataset, load_dataset2, plot_image
from torchvision.transforms import v2
from matplotlib import pyplot as plt

from UNet import ssim_loss

pos_transform = v2.Compose([
    v2.Pad(6, 1.0),
    v2.RandomCrop((64 + 6, 64 + 6)),
    v2.Pad(1, 1.0),
    #v2.Pad(8, 1.0),
    #v2.RandomCrop((64 + 8, 64 + 8)),
    #v2.RandomChannelPermutation(),
])

pos_transform2 = v2.Compose([
    v2.Pad(6, [1] + [0] * 15),
    v2.RandomCrop((64 + 6, 64 + 6)),
    v2.Pad(1, [1] + [0] * 15),
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
    def __init__(self, in_c, img_size, patch_size, emb_dim):
        super().__init__()
        assert img_size % patch_size == 0

        self.emb_dim = emb_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.masked_size = int(0.5 * self.num_patches)

        self.palette_emb = nn.Linear(3 * 16, self.emb_dim)

        self.patch_embedding = nn.Conv2d(in_c, self.emb_dim, 
                                         kernel_size=patch_size, 
                                         stride=patch_size,
                                         bias=True)

        self.pos_embedding = nn.Parameter(torch.zeros(self.num_patches, self.emb_dim), requires_grad=False)
        
        self.encoder = nn.Sequential(*[Transformer(self.emb_dim) for _ in range(15)])
        self.decoder = nn.Sequential(*[Transformer(self.emb_dim) for _ in range(3)])

        self.decoder_emb_dim = self.emb_dim
        self.decoder_emb = nn.Linear(self.emb_dim, self.decoder_emb_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_emb_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.decoder_pos_emb = nn.Parameter(torch.zeros(self.num_patches, self.decoder_emb_dim))

        self.img_recov = nn.Linear(self.decoder_emb_dim, in_c * (self.patch_size ** 2), bias=True)
        

    def forward(self, x, xp):
        bs, _, _, _ = x.shape
        device = x.device

        patches = self.patch_embedding(x)
        patches = patches.flatten(2, 3).transpose(1, 2)

        mask = torch.randperm(self.num_patches, device=device) #img size independent?
        mask = mask[:-self.masked_size]
        masked = patches[:, mask, :]

        pal_emb = self.palette_emb(xp.flatten(1, 2))
        pos_emb = self.pos_embedding[mask, :]
        tokens = masked + pos_emb[None, ...] + pal_emb[:, None, :]
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

def get_batch(bs, pimages):
    source = torch.randint(0, 5, (bs,))
    batch = torch.randperm(385)[:bs]
    x0 = pimages[source, batch]
    x0 = torch.stack([pos_transform(x) for x in x0], dim=0)

    return x0

def plot_images(x):
    fig, axes = plt.subplots(1, 4, figsize=(15, 15))
    plot_image(x[0], axes[0])
    plot_image(x[1], axes[1])
    plot_image(x[2], axes[2])
    plot_image(x[3], axes[3])
    plt.show()

def train():
    device = "cuda"

    pimages = load_dataset().to(device)

    model = MyMAE(3, 72, 4, 64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()

    for i in tqdm(range(2000000), miniters=10):
        x0 = get_batch(64, pimages)

        optimizer.zero_grad()

        reconstruction = model(x0)
        loss = mse_loss(reconstruction, x0) + 2.0 * ssim_loss(reconstruction, x0).mean()

        loss.backward()
        optimizer.step()

        if i % 5000 == 0:
            plot_images(reconstruction)

    torch.save(model.state_dict(), f'{int(time.time())}_mae.pt') #overfit?
    torch.save(model.state_dict(), 'last_mae.pt') #overfit?

def train2():
    device = "cuda"

    pimages, ppalett = load_dataset2()
    pimages = pimages.to(device)
    ppalett = ppalett.to(device)

    model = MyMAE(16, 72, 4, 128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1.2e-4)
    mse_loss = nn.MSELoss()

    for i in tqdm(range(300000), miniters=10):
        bs = 32
        source = torch.randint(0, 5, (bs,), device=device)
        batch = torch.randperm(385, device=device)[:bs]

        x0 = pimages[source, batch]
        x0 = torch.stack([pos_transform2(x) for x in x0], dim=0)
        xp = ppalett[source, batch]
        
        for j in range(bs):
            xp[j] = v2.ColorJitter(0.1, 0.1, 0.1, 0.2)(xp[[j]].transpose(2, 1)[..., None])[..., 0].transpose(2, 1)
            pal_shuffle = torch.randperm(16, device=device)
            x0[j] = x0[j, pal_shuffle, :, :]
            xp[j] = xp[j, pal_shuffle, :]

        optimizer.zero_grad()

        reconstruction = model(x0, xp)
        reconstruction = torch.softmax(reconstruction, 1)
        loss = mse_loss(reconstruction, x0)# + 2.0 * ssim_loss(reconstruction, x0).mean()

        loss.backward()
        optimizer.step()

        if i % 2000 == 0:
            rgb = (xp.transpose(1,2) @ x0.flatten(2,3)).unflatten(2, (72, 72))
            plot_images(rgb)
            rgbr = (xp.transpose(1,2) @ reconstruction.flatten(2,3)).unflatten(2, (72, 72))
            plot_images(rgbr)

    torch.save(model.state_dict(), f'{int(time.time())}_mae.pt') #overfit?
    torch.save(model.state_dict(), 'last_mae.pt') #overfit?

def test2():
    device = "cuda"

    pimages, ppalett = load_dataset2()
    pimages = pimages.to(device)
    ppalett = ppalett.to(device)

    model = MyMAE(16, 72, 4, 128).to(device)

    sd = torch.load('last_mae.pt')
    model.load_state_dict(sd)
    model.eval()

    bs = 4
    source = torch.randint(0, 5, (bs,), device=device)
    batch = torch.randperm(385, device=device)[:bs]

    x0 = pimages[source, batch]
    x0 = torch.stack([pos_transform2(x) for x in x0], dim=0)
    xp = ppalett[source, batch]

    for j in range(bs):
        xp[j] = v2.ColorJitter(0.1, 0.1, 0.1, 0.2)(xp[[j]].transpose(2, 1)[..., None])[..., 0].transpose(2, 1)
        pal_shuffle = torch.randperm(16, device=device)
        x0[j] = x0[j, pal_shuffle, :, :]
        xp[j] = xp[j, pal_shuffle, :]

    reconstruction = model(x0, xp)
    reconstruction = torch.softmax(reconstruction, 1)

    max_idx = torch.argmax(reconstruction, 1, keepdim=True)
    one_hot = torch.zeros_like(reconstruction).cpu()
    one_hot.zero_()
    one_hot.scatter_(1, max_idx.cpu(), 1)
    one_hot = one_hot.to(device)

    rgbr = (xp.transpose(1,2) @ one_hot.flatten(2,3)).unflatten(2, (72, 72))
    plot_images(rgbr)

    

def test():
    device = "cuda"
    model = MyMAE(72, 4, 64).to(device)

    sd = torch.load('last_mae.pt')
    model.load_state_dict(sd)
    model.eval()

    pimages = load_dataset().to(device)
    x0 = get_batch(4, pimages)

    plot_images(model(x0))


if __name__ == '__main__':
    train()