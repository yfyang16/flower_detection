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

img_size = 224

#normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
transform  = T.Compose([
         T.RandomResizedCrop(img_size),
         T.RandomHorizontalFlip(),
         T.ToTensor(),
         T.Normalize(),
])


def load_train_dataset():
    flowers_dir = os.getcwd() + "/flowers/"

    dataset = ImageFolder(flowers_dir, transform=transform)

    return dataset

def load_test_dataset():
    flowers_dir = os.getcwd() + "/flowers/"

    dataset = ImageFolder(flowers_dir, transform=transform)

    return dataset


# np.save("dataset.npy", dataset)

#print(dataset.class_to_idx)
#print(dataset.imgs)
# print(dataset[0][0].size())

# to_img = T.ToPILImage()

#draw = ImageDraw.Draw(to_img(dataset[0][0]*0.2+0.4))

# lena1 = mpimg.imread(to_img(dataset[0][0]*0.2+0.4))
# plt.imshow(to_img(dataset[0][0]*0.2+0.4))
# plt.show()