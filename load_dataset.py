#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms

from random import random
from PIL import Image
import json

import matplotlib.pyplot as plt

def ten_to_PIL(ten, h=random()):
    t = transforms.ToPILImage(mode="HSV")
    
    ten = torch.cat([torch.zeros(1,64,64, device="cuda") + h, ten]) # Add H

    ten[2] += 1 - ten[3] # White background
    return t(ten[:3].clamp(0,1))

def plot_image(im, h=random()):
    im = ten_to_PIL(im)
    
    plt.imshow(im)
    plt.show()
    
def save_image(im, path, h=random()):
    im = ten_to_PIL(im)
    
    im.convert("RGB").save(path)

def load_dataset():
    
    PATH = "./dataset/"
    PATH = "/home/danilo/Downloads/spritesA/"
    
    transform = transforms.ToTensor()
    
    def load_images(path):
        ten = []
        for i in range(386):
            img = Image.open(path + f"{i+1}.png").convert("RGBA")
            hsv = img.convert("HSV")
            
            hsv = transform(hsv)
            alpha = transform(img)[3]
            
            hsv[0] *= alpha
            hsv[1] *= alpha
            hsv[2] *= alpha
            
            
            img = torch.cat([hsv, alpha.unsqueeze(0)])
            
            ten.append(img)
        
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
    
