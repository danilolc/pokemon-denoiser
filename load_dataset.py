#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms

from PIL import Image
import json

import matplotlib.pyplot as plt
import numpy as np

mode = "RGB"
normalize = True

if mode == "HSV":
    MEAN = np.array([0.1264, 0.2050, 0.3135]) # SVA
    STD = np.array([0.2522, 0.3451, 0.4639])
elif mode == "RGB":
    MEAN = np.array([0.1811, 0.1605, 0.1359, 0.3135]) # RGBA
    STD = np.array([0.3186, 0.2844, 0.2617, 0.4639])

iMEAN = -MEAN / STD
iSTD = 1 / STD

def ten_to_HSV(ten, h):
    toPIL = transforms.ToPILImage(mode="HSV")
    
    if normalize:
        deNorm = transforms.Normalize(iMEAN, iSTD)
        ten = deNorm(ten)
    
    ten = torch.cat([torch.zeros(1,64,64, device="cuda") + h, ten]) # Add H

    ten[2] += 1 - ten[3] # White background
    ten = ten[:3]
    
    return toPIL(ten.clamp(0,1))

def ten_to_RGB(ten):
    toPIL = transforms.ToPILImage(mode="RGBA")
    
    if normalize:
        deNorm = transforms.Normalize(iMEAN, iSTD)
        ten = deNorm(ten)

    return toPIL(ten.clamp(0,1))



def plot_image(im, h=0):
    if mode == "HSV":
        im = ten_to_HSV(im, h)
    elif mode == "RGB":
        im = ten_to_RGB(im)
    
    plt.imshow(im)
    plt.show()


def save_image(im, path, h=0):
    if mode == "HSV":
        im = ten_to_HSV(im, h)
    elif mode == "RGB":
        im = ten_to_RGB(im)
    
    im.convert("RGBA").save(path)


def load_image_RGB(path):
    
    toTensor = transforms.ToTensor()
    
    img = Image.open(path).convert("RGBA")
    img = toTensor(img)
    
    alpha = img[3]
    
    img[0] *= alpha
    img[1] *= alpha
    img[2] *= alpha
    
    if normalize:
        toNorm = transforms.Normalize(MEAN, STD)
        img = toNorm(img)
    
    return img

def load_image_HSV(path):
    
    toTensor = transforms.ToTensor()
    
    img = Image.open(path).convert("RGBA")
    hsv = img.convert("HSV")
    
    hsv = toTensor(hsv)
    alpha = toTensor(img)[3]
    
    hsv[0] *= alpha
    hsv[1] *= alpha
    hsv[2] *= alpha
    
    hsv = torch.cat([hsv[1:], alpha.unsqueeze(0)]) # Remove H and add alpha
    
    if normalize:
        toNorm = transforms.Normalize(MEAN, STD)
        hsv = toNorm(hsv)
    
    return hsv


def load_dataset():
    
    PATH = "./dataset/"
    PATH = "/home/danilo/Downloads/spritesA/"
    
    def load_images(path):
        ten = []
        for i in range(386):
            if mode == "HSV":
                im = load_image_HSV(path + f"{i+1}.png")
            elif mode == "RGB":
                im = load_image_RGB(path + f"{i+1}.png")
            ten.append(im)
        
        return torch.stack(ten)
        
    emerald = load_images(PATH + "emerald/")
    frlg = load_images(PATH + "frlg/")
    rs = load_images(PATH + "rs/")
    
    return torch.stack([emerald, frlg, rs])

def load_types(): # Get color and body type
    with open("types.json") as f:
        types = json.load(f)
        types = [t["english"] for t in types]
        
    with open("pokedex.json") as f:
        pokemons = json.load(f)
        pokemons = [t["type"] for t in pokemons]
    
    type_tensors = [types.index(pokemons[i][0]) for i in range(386)]
    
    return torch.tensor(type_tensors)
    
