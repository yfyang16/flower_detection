#   _*_ coding:utf-8 _*_
__author__ = 'yangyufeng'
"""
训练resnet之前预处理图片数据
"""

import torchvision.transforms as T
import torch
import os
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # 用于读取图片

img_size = 200


transform  = T.Compose([
         T.RandomResizedCrop(img_size),
         T.RandomHorizontalFlip(),
         T.ToTensor(),
         #T.Normalize(),
])


def load_train_dataset():
    flowers_dir = os.getcwd() + "/flowers_augmentation/"

    dataset = ImageFolder(flowers_dir, transform=transform)

    return dataset

def load_test_dataset(test_dir):
    flowers_dir = os.getcwd() + '/' + str(test_dir) +'/'

    dataset = ImageFolder(flowers_dir, transform=transform)

    return dataset


