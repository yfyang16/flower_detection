import torchvision.transforms as transforms
import torch
import os
import numpy as np
from PIL import Image

transform = transforms.Compose([
    transforms.Resize(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

#crop = transforms.Scale(32,32)

flowers_dir = os.getcwd() + "/flowers/"
#print(flowers_dir)

flower_dict = {'daisy': 1, 'dandelion': 2, 'rose': 3, 'sunflower': 4, 'tulip': 5}


def load_Img(imgFoldName, imgDir=flowers_dir):
    imgs = os.listdir(imgDir + "/" +imgFoldName)
    imgNum = len(imgs)
    data = np.empty((imgNum, 3, 32, 32))
    label = np.empty((imgNum,))
    for i in range(imgNum):
        img = Image.open(imgDir + "/" +imgFoldName + "/" + imgs[i])
        img = transform(img)
        arr = np.asarray(img, dtype="float32")
        data[i, :, :, :] = arr
        label[i] = flower_dict[imgFoldName]
    return data, label


#roseData, roseLabel = load_Img("rose")

#print(roseData.dtype, roseLabel.dtype)

dasiyDATA, dasiyLabel = load_Img("daisy")
dandelionDATA, dandelionLabel = load_Img("dandelion")
roseDATA, roseLabel = load_Img("rose")
sunflowerDATA, sunflowerLabel = load_Img("sunflower")
tulipDATA, tulipLabel = load_Img("tulip")


train_x_dataset = np.concatenate((dasiyDATA, dandelionDATA, roseDATA, sunflowerDATA, tulipDATA))
train_y_label = np.concatenate((dasiyLabel, dandelionLabel, roseLabel, sunflowerLabel, tulipLabel))

np.random.shuffle(train_x_dataset)
np.random.shuffle(train_y_label)
#print(train_x_dataset)

x_train = train_x_dataset[:int(0.9*len(train_x_dataset))]
x_test = train_x_dataset[int(0.9*len(train_x_dataset)):]

y_train = train_y_label[:int(0.9*len(train_x_dataset))]
y_test = train_y_label[int(0.9*len(train_x_dataset)):]

np.save("x_train.npy", x_train)
np.save("x_test.npy", x_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
