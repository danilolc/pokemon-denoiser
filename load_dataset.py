#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, os
from torchvision import transforms

from PIL import Image

import matplotlib.pyplot as plt

PATH = "./dataset/"
MODE = "RGB"

def ten_to_RGB(ten):
    toPIL = transforms.ToPILImage(mode=MODE)

    return toPIL(ten.clamp(0,1))


def plot_image(im, axes=plt):
    im = ten_to_RGB(im)
    
    axes.imshow(im)


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
    
    return img


def load_dataset():
    
    def load_images(path):
        ten = []
        for i in range(386):
            im = load_image_RGB(path + f"{i+1}.png")
            ten.append(im)
        
        return torch.stack(ten)
        
    rs = load_images(PATH + "rs/")
    frlg = load_images(PATH + "frlg/")
    emerald = load_images(PATH + "emerald/")

    rs_back = load_images(PATH + "rs/back/")
    frlg_back = load_images(PATH + "frlg/back/")

    return torch.stack([rs, frlg, emerald, rs_back, frlg_back])


def load_dataset2():

    src = './dataset2/'

    pokes = os.listdir(src + 'emerald/')

    def load_images(path):
        ten = []
        for p in pokes:
            im = Image.open(path + f"{p}")
            im = transforms.functional.pil_to_tensor(im)
            assert im.shape[0] == 1

            ten.append(im)
        
        return torch.stack(ten)
        
    rs = load_images(src + "rs/")
    frlg = load_images(src + "frlg/")
    em = load_images(src + "emerald/")

    rs_back = load_images(src + "rs_back/")
    frlg_back = load_images(src + "frlg_back/")

    return torch.stack([rs, frlg, em, rs_back, frlg_back])


def load_contour():
    
    def load_images(path):
        ten = []
        for i in range(386):
            im = load_image_RGB(path + f"{i+1}.png")
            ten.append(im)
        
        return torch.stack(ten)
        
    emerald = load_images(PATH + "emerald_cont/")
    frlg = load_images(PATH + "frlg_cont/")
    rs = load_images(PATH + "rs_cont/")

    return torch.stack([emerald, frlg, rs]).mean(dim=2, keepdim=True)
