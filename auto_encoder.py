#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn, cat

def PConv(in_c, out_c):
    return nn.Conv2d(in_c, out_c, 
                     kernel_size=3, 
                     padding='same',  
                     padding_mode = "replicate")

def TConv(in_c, out_c):
    return nn.ConvTranspose2d(in_c, out_c,
                              kernel_size=3, 
                              stride=2, 
                              padding = 1, 
                              output_padding=1)

class PAutoE(nn.Module):
    def __init__(self, in_c=3, out_c=3):
        super().__init__()
        
        self.convs1 = nn.Sequential(
                PConv(in_c, 64),
                nn.ReLU(),

                PConv(64, 64),
                nn.ReLU(),
            )
        self.pool = nn.Sequential(nn.AvgPool2d(2))

        self.convs2 = nn.Sequential(
                PConv(64, 64),
                nn.ReLU(),

                PConv(64, 64),
                nn.ReLU(),
            )
            
        self.tconv = nn.Sequential(
                TConv(64, 64),
                nn.ReLU(),
            )
                
        self.convs3 = nn.Sequential(
                PConv(128, 64),
                nn.ReLU(),
                
                PConv(64, out_c),
                nn.Sigmoid(),
            )
        
    def forward(self, x):
        x = self.convs1(x)
        x1 = self.pool(x)

        x1 = self.convs2(x1)
        x1 = self.tconv(x1)

        x = cat([x1, x], dim=1)
        x = self.convs3(x)

        return x 