#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/LICENSE


import torch
from torch import nn, cat
import torch.nn.functional as F

# https://arxiv.org/pdf/1804.03999.pdf
class AdditiveAttentionGate(nn.Module):
    def __init__(self, fx, fg, fint):
        super().__init__()

        self.Wx = nn.Sequential(
            nn.Conv2d(fx, fint, kernel_size=1, stride=2),
            nn.BatchNorm2d(fint),
        )
        self.Wg = nn.Sequential(
            nn.Conv2d(fg, fint, kernel_size=1, bias=False),
            nn.BatchNorm2d(fint),
        )

        self.Psi = nn.Sequential(
            nn.Conv2d(fint, 1, kernel_size=1),
            nn.BatchNorm2d(1),
        )

    def forward(self, x, g):
        
        xv = self.Wx(x)
        gv = self.Wg(g)
        
        s = nn.functional.relu(xv + gv, inplace=True)
        s = self.Psi(s)

        s = torch.sigmoid(s)

        s = nn.functional.interpolate(s,
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=True)

        return s

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, residual=False):
        super().__init__()
        self.residual = residual

        self.double_conv = nn.Sequential(
            
            nn.Conv2d(in_c, out_c, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            
            nn.Conv2d(out_c, out_c, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_c),
        
        )

    def forward(self, x):
        if self.residual:
            return x + F.relu(self.double_conv(x))
        else:
            return F.relu(self.double_conv(x))


class TConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.tconv = nn.Sequential(

            nn.ConvTranspose2d(in_c, out_c,
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),

        )

                                
    
    def forward(self, x):
        return self.tconv(x)

class PAutoE(nn.Module):
    def __init__(self, in_c=3, out_c=3):
        super().__init__()
        
        self.pool = nn.MaxPool2d(2)

        #self.att1 = AdditiveAttentionGate(64, 128, 128)
        #self.att2 = AdditiveAttentionGate(32, 64, 64)
        #self.att3 = AdditiveAttentionGate(16, 32, 32)
                
        self.convs1 = nn.Sequential(

            #PixelSight
            nn.Conv2d(in_c, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        
            DoubleConv(16, 16, residual=True),
            DoubleConv(16, 16, residual=True),
            #nn.Dropout2d(0.2),
        
        ) # 16 --->

        self.convs2 = nn.Sequential(

            DoubleConv(16, 32),
            DoubleConv(32, 32, residual=True),
            DoubleConv(32, 32, residual=True),
            #nn.Dropout2d(0.2),
        
        ) # 32 --->
        
        self.convs3 = nn.Sequential(

            DoubleConv(32, 64),
            DoubleConv(64, 64, residual=True),
            DoubleConv(64, 64, residual=True),
            #nn.Dropout2d(0.2),
        
        ) # 64

        self.convs4 = nn.Sequential(

            DoubleConv(64, 128),
            DoubleConv(128, 128, residual=True),
            DoubleConv(128, 128, residual=True),
            #nn.Dropout2d(0.2),
        
        ) # 128

        ######
         
        self.tconv1 = TConv(128, 128)

        self.convs5 = nn.Sequential(

            DoubleConv(128 + 64, 64),
            DoubleConv(64, 64, residual=True),
            DoubleConv(64, 64, residual=True),
        
        )
        
        self.tconv2 = TConv(64, 64)
                
        self.convs6 = nn.Sequential(

            DoubleConv(64 + 32, 32),
            DoubleConv(32, 32, residual=True),
            DoubleConv(32, 32, residual=True),

        )

        self.tconv3 = TConv(32, 32)

        self.convs7 = nn.Sequential(

            DoubleConv(32 + 16, 16),
            DoubleConv(16, 16, residual=True),
            DoubleConv(16, 16, residual=True),

            nn.Conv2d(16, out_c, kernel_size=1),

        )
        
    def forward(self, x):

        x0 = self.convs1(x)
        
        x1 = self.pool(x0)
        x1 = self.convs2(x1)

        x2 = self.pool(x1)
        x2 = self.convs3(x2)

        x3 = self.pool(x2)
        x3 = self.convs4(x3)
        
        ####
        
        #att = self.att1(x2, x3)
        x3 = self.tconv1(x3)
        x3 = self.convs5(cat([x2, x3], dim=1))
        
        #att = self.att2(x1, x3)
        x3 = self.tconv2(x3)
        x3 = self.convs6(cat([x1, x3], dim=1))

        #att = self.att3(x0, x3)
        x3 = self.tconv3(x3)
        x3 = self.convs7(cat([x0, x3], dim=1))

        return x3

        