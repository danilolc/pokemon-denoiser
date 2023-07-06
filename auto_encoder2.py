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
        
        self.pool = nn.MaxPool2d(2)
        
        self.convs1 = nn.Sequential(
            
            #nn.ChannelShuffle()

            #PixelSight
            nn.Conv2d(in_c, 16, kernel_size=1),
            nn.ReLU(),
        
            DoubleConv(16, 16),
            DoubleConv(16, 16, residual=True),
        
        ) # 16 --->

        self.convs2 = nn.Sequential(

            DoubleConv(16, 32),
            DoubleConv(32, 32, residual=True),
        
        ) # 32 --->
        
        self.convs3 = nn.Sequential(

            DoubleConv(32, 64),
            DoubleConv(64, 64, residual=True),
        
        ) # 64

        ######
         
        self.tconv1 = nn.Sequential(

            TConv(64, 64),
            nn.ReLU(),

        )
        
        self.convs4 = nn.Sequential(

            DoubleConv(64 + 32, 32),
            DoubleConv(32, 32, residual=True),
        
        )
        
        self.tconv2 = nn.Sequential(

            TConv(32, 32),
            nn.ReLU(),
            
        )
                
        self.convs5 = nn.Sequential(

            DoubleConv(32 + 16, 16),
            DoubleConv(16, 16, residual=True),
            
            #PixelSight
            nn.Conv2d(16, out_c, kernel_size=1),

        )
        
    def forward(self, x):
        
        x = self.convs1(x)
        
        x1 = self.pool(x)
        x1 = self.convs2(x1)

        x2 = self.pool(x1)
        x2 = self.convs3(x2)
        
        ####
        
        x2 = self.tconv1(x2)
        x2 = self.convs4(cat([x1, x2], dim=1))
        
        x2 = self.tconv2(x2)
        x2 = self.convs5(cat([x, x2], dim=1))

        return x2