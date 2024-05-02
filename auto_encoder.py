#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/LICENSE

#self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

from torch import nn, cat
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, residual=False):
        super().__init__()
        self.residual = residual

        self.double_conv = nn.Sequential(
            
            nn.Conv2d(in_c, out_c, kernel_size=3, padding='same', bias=False),
            nn.GroupNorm(1, out_c),
            nn.GELU(),
            
            nn.Conv2d(out_c, out_c, kernel_size=3, padding='same', bias=False),
            nn.GroupNorm(1, out_c),
        
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class TConv(nn.Module): # Use  nn.Upsample
    def __init__(self, in_c, out_c):
        super().__init__()
        #self.tconv = nn.Upsample(scale_factor=2, 
        #                         mode="bilinear", 
        #                         align_corners=True)
        self.tconv = nn.ConvTranspose2d(in_c, out_c,
                                  kernel_size=3, 
                                  stride=2, 
                                  padding = 1, 
                                  output_padding=1)
    
    def forward(self, x):
        return self.tconv(x)

class PAutoE(nn.Module):
    def __init__(self, in_c=3, out_c=3):
        super().__init__()
        
        self.convs1 = nn.Sequential(
        
            DoubleConv(in_c, 16),
            DoubleConv(16, 16, residual=True),
        
        )
        
        self.pool = nn.MaxPool2d(2)

        self.convs2 = nn.Sequential(
        
            DoubleConv(16, 32),
            DoubleConv(32, 32, residual=True),
        
        )
         
        self.tconv = nn.Sequential(
                TConv(32, 32),
                nn.ReLU(),
            )
                
        self.convs3 = nn.Sequential(
            
                DoubleConv(32+16, 32),
                DoubleConv(32, 32, residual=True),
                
                nn.Conv2d(32, out_c, kernel_size=1, padding='same'),
                #nn.Sigmoid(),

            )
        """
        self.type_linear = nn.Sequential( # Embeeding com uma camada linear
                nn.Linear(18, 32),
                nn.ReLU(),
                
                nn.Linear(32, 64),
                nn.ReLU(),
            )
        """
        self.emb = nn.Embedding(18, 32)
        self.emb1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(32, 16)
        )
        self.emb2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(32, 32+16)
        )
        
    def forward(self, x, ty):
        #ty = self.emb(ty)
        
        x = self.convs1(x)
        x1 = self.pool(x)
        
        #ty1 = self.emb1(ty)[:, :, None, None]
        #ty1 = ty1.repeat(1, 1, x1.shape[-2], x1.shape[-1])
        #x1 += ty1
        
        x1 = self.convs2(x1)        
        x1 = self.tconv(x1)
        x = cat([x1, x], dim=1)
        
        #ty2 = self.emb2(ty)[:, :, None, None]
        #ty2 = ty2.repeat(1, 1, x.shape[-2], x.shape[-1])
        #x += ty2

        x = self.convs3(x)

        return x 