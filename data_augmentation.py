#   _*_ coding:utf-8 _*_
__author__ = 'yangyufeng'

"""
PIL数据扩充，翻转、颜色增强、旋转等
"""

from PIL.ImageOps import mirror
from PIL import Image, ImageEnhance
import os

#flowers_dir = os.getcwd() + "/flowers_augmentation/"
flowers_dir = "/Users/yangyufeng/Desktop/lingyushujuxue/flower_classification/flowers_augmentation"

imgFolds = ['daisy', 'rose', 'sunflower', 'tulip']
new_imgFolds = ['daisy_new', 'rose_new', 'sunflower_new', 'tulip_new']

for imgFold in imgFolds:

    imgs = os.listdir(flowers_dir + '/' + imgFold)
    imgNum = len(imgs)
    waiting = 0
    for i in range(imgNum):
        if imgs[i] == ".DS_Store":
            continue
        im_path = flowers_dir + '/' + imgFold + '/' + imgs[i]
        im = Image.open(im_path)
        #newM_im = mirror(im)
        #newM_im.save(flowers_dir + '/' + imgFold + '/' + "newM_" + imgs[i])

        #newC_im = ImageEnhance.Color(im).enhance(1.5)
        #newC_im.save(flowers_dir + '/' + imgFold + '/' + "newC_" + imgs[i])

        #newR_im = im.rotate(15)
        #newR_im.save(flowers_dir + '/' + imgFold + '/' + "newR_" + imgs[i])

        newCt_im = ImageEnhance.Contrast(im).enhance(2)
        newCt_im.save(flowers_dir + '/' + imgFold + '/' + "newCt_" + imgs[i])

        waiting += 1
        if waiting % 2000 == 0:
            print('=============' + str(waiting) + '==============')



