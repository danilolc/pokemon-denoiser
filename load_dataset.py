#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms

from PIL import Image
import json

import matplotlib.pyplot as plt
import numpy as np

PATH = "./dataset/"
MODE = "RGB"

def ten_to_RGB(ten):
    toPIL = transforms.ToPILImage(mode=MODE)
    
    # [-1, 1] -> [0, 1]
    ten = (ten + 1) / 2

    return toPIL(ten.clamp(0,1))


def plot_image(im, axes=plt):
    im = ten_to_RGB(im)
    
    axes.imshow(im)
    #axes.show()


def save_image(im, path):
    im = ten_to_RGB(im)
    
    im.convert("RGBA").save(path)


def load_image_RGB(path):
    
    toTensor = transforms.ToTensor()
    
    img = Image.open(path)
    img = toTensor(img.convert("RGBA"))
    
    alpha = img[3]
    
    # Set RGB = 0 when transparent
    img[0] *= alpha
    img[1] *= alpha
    img[2] *= alpha
    
    if (MODE == "RGB"):
        # Set white when transparent
        img[0] += 1 - alpha
        img[1] += 1 - alpha
        img[2] += 1 - alpha
        img = img[0:3]
        
    # [0, 1] -> [-1, 1]
    img = (img * 2) - 1
    
    return img


def load_dataset():
    
    def load_images(path):
        ten = []
        for i in range(386):
            im = load_image_RGB(path + f"{i+1}.png")
            ten.append(im)
        
        return torch.stack(ten)
        
    emerald = load_images(PATH + "emerald/")
    frlg = load_images(PATH + "frlg/")
    rs = load_images(PATH + "rs/")
    
    return torch.stack([emerald, frlg, rs])
