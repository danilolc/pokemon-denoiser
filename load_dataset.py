#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms

from PIL import Image
import json

import matplotlib.pyplot as plt
import numpy as np

MEAN = np.array([0.1264, 0.2050, 0.3135])
STD = np.array([0.2522, 0.3451, 0.4639])

iMEAN = -MEAN / STD
iSTD = 1 / STD

def ten_to_PIL(ten, h):
    toPIL = transforms.ToPILImage(mode="HSV")
    
    denormalize = transforms.Normalize(iMEAN, iSTD)
    ten = denormalize(ten)
    
    ten = torch.cat([torch.zeros(1,64,64, device="cuda") + h, ten]) # Add H

    ten[2] += 1 - ten[3] # White background
    ten = ten[:3]
    
    return toPIL(ten.clamp(0,1))


def plot_image(im, h=0):
    im = ten_to_PIL(im, h=h)
    
    plt.imshow(im)
    plt.show()

    
def save_image(im, path, h=0):
    im = ten_to_PIL(im, h=h)
    
    im.convert("RGB").save(path)

    
def load_image(path):
    
    toTensor = transforms.ToTensor()
    
    img = Image.open(path).convert("RGBA")
    hsv = img.convert("HSV")
    
    hsv = toTensor(hsv)
    alpha = toTensor(img)[3]
    
    hsv[0] *= alpha
    hsv[1] *= alpha
    hsv[2] *= alpha
    
    hsv = torch.cat([hsv[1:], alpha.unsqueeze(0)]) # Remove H and add alpha
    
    normalize = transforms.Normalize(MEAN, STD)
    hsv = normalize(hsv)
    
    return hsv


def load_dataset():
    
    PATH = "./dataset/"
    PATH = "/home/danilo/Downloads/spritesA/"
    
    def load_images(path):
        ten = []
        for i in range(386):
            ten.append(load_image(path + f"{i+1}.png"))
        
        return torch.stack(ten)
        
    emerald = load_images(PATH + "emerald/")
    frlg = load_images(PATH + "frlg/")
    rs = load_images(PATH + "rs/")
    
    return torch.stack([emerald, frlg, rs])

def load_types():
    with open("types.json") as f:
        types = json.load(f)
        types = [t["english"] for t in types]
        
    with open("pokedex.json") as f:
        pokemons = json.load(f)
        pokemons = [t["type"] for t in pokemons]
    
    type_tensors = [[float(t in pokemons[i]) for t in types] for i in range(386)]
    
    return torch.tensor(type_tensors)
    
