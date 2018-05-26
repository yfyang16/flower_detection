#   _*_ coding:utf-8 _*_
__author__ = 'yangyufeng'

import os

flowers_train_dir = os.getcwd() + "/flowers/"
#print(flowers_dir)

# flower_dict = {'daisy': 1, 'dandelion': 2, 'rose': 3, 'sunflower': 4, 'tulip': 5}


def changePicName(imgFoldName, imgDir=flowers_train_dir):
	"""
	将所有图片的名称依次编号
	"""
    imgs = os.listdir(imgDir + "/" + imgFoldName)
    count = 0
    for temp in imgs:
        new_name = (3 - len(str(count)))*str(0)+str(count) + '_' + imgFoldName+'.jpg'
        os.renames(imgDir+'/'+imgFoldName+'/' + temp, imgDir+'/'+imgFoldName+'/' + new_name)
        count += 1

    return 0

changePicName('daisy')
changePicName('rose')
changePicName('sunflower')
changePicName('tulip')