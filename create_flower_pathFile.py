#   _*_ coding:utf-8 _*_
__author__ = 'yangyufeng'

"""
yolo训练准备:
创建图片放置位置的绝对路径
"""

import os

Image_path = "/Users/yangyufeng/Desktop/flower_detection/JPEGImages"
files_name = os.listdir(Image_path)

with open("/Users/yangyufeng/Desktop/flower_detection/flower_train.txt", 'w') as f1:
    for i in range(len(files_name)):
        f1.write(Image_path + '/' + files_name[i] + '\n')

with open("/Users/yangyufeng/Desktop/flower_detection/train.txt", 'w') as f2:
    for i in range(len(files_name)):
        f2.write(files_name[i].split('.')[0] + '\n')