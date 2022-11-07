#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn

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
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
                PConv(3, 64),
                nn.ReLU(),

                PConv(64, 64),
                nn.ReLU(),

                nn.AvgPool2d(2),
            )
        self.tconvs = nn.Sequential(
                TConv(64, 64),
                nn.ReLU(),
                
                PConv(64, 64),
                nn.ReLU(),
                
                PConv(64, 3),
                nn.Sigmoid(),
            )
        
    def forward(self, x):
        x = self.convs(x)
        x = self.tconvs(x)

        return x 