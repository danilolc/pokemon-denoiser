#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn, cat

def PConv(in_c, out_c):
    return nn.Conv2d(in_c, out_c, 
                     kernel_size=3, 
                     padding='same',  
                     padding_mode = "zeros")

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
                PConv(in_c, 32),
                nn.ReLU(),
                
                nn.BatchNorm2d(32),

                PConv(32, 32),
                nn.ReLU(),
            )
        
        self.pool = nn.Sequential(nn.AvgPool2d(2))

        self.convs2 = nn.Sequential(
                PConv(32, 64),
                nn.ReLU(),
                
                nn.BatchNorm2d(64),

                PConv(64, 64),
                nn.ReLU(),
            )
         
        self.tconv = nn.Sequential(
                TConv(64, 64),
                nn.ReLU(),
            )
                
        self.convs3 = nn.Sequential(
                PConv(96, 64),
                nn.ReLU(),
                
                nn.BatchNorm2d(64),
                
                PConv(64, out_c),
                nn.Sigmoid(),
            )
        """
        self.type_linear = nn.Sequential(
                nn.Linear(18, 32),
                nn.ReLU(),
                
                nn.Linear(32, 64),
                nn.ReLU(),
            )
        """
        
    def forward(self, x, ty):
        x = self.convs1(x)
        
        x1 = self.pool(x)
        x1 = self.convs2(x1)

        #ty = self.type_linear(ty)
        #x1 = ty.permute(1,0) * x1.permute(3,2,1,0)
        #x1 = x1.permute(3,2,1,0)
        
        x1 = self.tconv(x1)
        x = cat([x1, x], dim=1)
        x = self.convs3(x)

        return x 