#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms

from PIL import Image
import json

def load_dataset(black_white=False):
    
    transform = transforms.ToTensor()
    
    def load_images(path):
        ten = []
        for i in range(386):
            img = Image.open(path + f"{i+1}.png").convert('RGB')
            if black_white:
                img = img.convert('L')
            img = transform(img)
            ten.append(img)
        
        return torch.stack(ten)
        
    emerald = load_images("./dataset/emerald/")
    frlg = load_images("./dataset/frlg/")
    rs = load_images("./dataset/rs/")
    
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
    
