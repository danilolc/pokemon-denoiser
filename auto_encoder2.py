#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/LICENSE



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
            return self.double_conv(x) # gelu?
            #return F.gelu(self.double_conv(x)) # gelu?


class TConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_c, out_c,
                                  kernel_size=3, 
                                  stride=2, 
                                  padding=1, 
                                  output_padding=1)
    
    def forward(self, x):
        return self.tconv(x)

class PAutoE(nn.Module):
    def __init__(self, in_c=3, out_c=3):
        super().__init__()
        
        self.convs1 = nn.Sequential(

            #PixelSight
            nn.Conv2d(in_c, 16, kernel_size=1),
            nn.ReLU(),
        
            DoubleConv(16, 16),
            DoubleConv(16, 16, residual=True),
        
        )
        
        self.pool = nn.MaxPool2d(2)

        self.convs2 = nn.Sequential(

            DoubleConv(16, 32),
            DoubleConv(32, 32, residual=True),
        
        )

        ######
         
        self.tconv = nn.Sequential(

            DoubleConv(32, 32),
            DoubleConv(32, 32, residual=True),

            TConv(32, 32),
            nn.ReLU(),
        )
                
        self.convs3 = nn.Sequential(

            DoubleConv(32 + 16, 16),
            DoubleConv(16, 16, residual=True),
            
            #PixelSight
            nn.Conv2d(16, out_c, kernel_size=1),

        )
        
    def forward(self, x):
        
        x = self.convs1(x)
        
        x1 = self.pool(x)
        x1 = self.convs2(x1)
        
        #####
        
        x1 = self.tconv(x1)
        x = cat([x1, x], dim=1)

        x = self.convs3(x)

        return x 