from __future__ import division
import torch
import torchfile
import torchvision.transforms as transforms
import numpy as np
import argparse
import time
import os
from PIL import Image
import torch.nn as nn
import cv2

def load_t7(filename):
    print("loading " + filename)
    ret = []
    net = torchfile.load(filename)
    print(filename + ":")
    for mod in net.modules:
        b = mod.bias
        if b is not None:
            w = mod.weight.astype(np.float16)
            b = b.astype(np.float16)
            ret.append({"w": w, "b": b})
    print("loaded " + str(len(ret)) + " layers")
    return ret

def load_image(fpath,size = 1024):
    print("loading image " + str(fpath))
    cImg = cv2.imread(fpath)
    if cImg.shape[0] > cImg.shape[1]:
        ar = cImg.shape[1] / cImg.shape[0]
        cImg =  cv2.resize(cImg, (int(ar*size),size), interpolation=cv2.INTER_CUBIC)
    else:
        ar = cImg.shape[0] / cImg.shape[1]
        cImg = cv2.resize(cImg, (size, int(ar * size)), interpolation=cv2.INTER_CUBIC)
    cImg = cImg.transpose(2, 0, 1)
    a = cImg[0].copy()
    cImg[0] = cImg[2]
    cImg[2] = a
    cImg = np.expand_dims(cImg, axis=0).astype(np.float16)
    #cImg -= cImg.min()
    cImg /= 255.0
    with torch.no_grad():
        cImg = torch.from_numpy(cImg).half().to("cuda:0")
    return cImg
