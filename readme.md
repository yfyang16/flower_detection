# Flower Classification after Detection

![Aaron Swartz](https://github.com/yyf710670079/flower_detection_minorPJ/raw/master/test%20images/predictions_19_daisy-%2061%25%20187%20311%20138%20244%20daisy-%2034%25%20252%20381%2046%20130%20daisy-%2027%25%2085%20166%2049%20109%20daisy-%2016%25%2029%20106%20185%20246.jpg)

### Python Package, Tool Source and Data Source

    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import keras
    import os
    import torchvision.transforms as T
    import torch,torchvision
    from ann_visualizer.visualize import ann_viz

#### lable tool
    git clone https://github.com/tzutalin/labelImg.git

#### kaggle dataset
    kaggle datasets download -d alxmamaev/flowers-recognition

#### YOLOv3 architecture
    git clone https://github.com/pjreddie/darknet

#### Resnet18 architecture
    cd darknet
    git clone https://github.com/yyf710670079/flower_detection_minorPJ.git





### The first step: Preprocess the data
1) Use Python to rename all the picture in order to make labeling convenient.
(number_train_imagedir.jpg, number_test_imagedir.jpg)


    cd flower_detection_minorPJ

    vi change_picname.py # "os.getcwd" change to your path of dataset

    python3 change_picname.py
 

2) Data augmentation: Flip and Random Crop by using python PIL package.


    cd flower_detection_minorPJ

    vi data_augmentation.py # delete several "#" and change the path to your own
    
    python3 data_augmentation.py




### The second step: Label the image

1) Use LabelImg to label the dataset and generate a file including cordinates and classes.


    git clone https://github.com/tzutalin/labelImg.git
    cd labelImg
    python3 labelImg.py

2) Pack the whole label text and match each picture.


    cd flower_detection_minorPJ
    vi create_flower_pathFile.py # change the image file's path to your own
    
    python3 create_flower_pathFile.py
    cd ..
    mv cfg/voc.data flower_detection_minorPJ/flower_voc.data
    vi flower_detection_minorPJ/flower_voc.data
    
    classes= 4  #类别数
    train  = flower_detection_minorPJ/flower_train.txt # path of training set labels
    valid  = flower_detection_minorPJ/flower_train.txt # path of valid set labels which you should create by your own like flower_train.txt
    names = flower_detection_minorPJ/flower_voc.names 
    backup = flower_detection_minorPJ/backup #create by your own
    
    mv data/voc.names flower_detection_minorPJ/flower_voc.names 
    vi flower_detection_minorPJ/flower_voc.names 
    '''
    daisy
    tulip
    rose
    sunflower
    '''

    mv cfg/yolov3-voc.cfg flower_detection_minorPJ/yolov3-voc.cfg
    vi flower_detection_minorPJ/yolov3-voc.cfg
    
    [convolutional]
    size=1
    stride=1
    pad=1
    filters=27         #filters = 3*(classes + 5)  3 places need to be altered
    activation=linear
    [yolo]
    mask = 0,1,2

### The third step: Train the Yolo model

While predicting:
	input : 3 x height x width image
	return: bounding boxes' cordinates and corresponding class information
	
	cd darknet
	# download weights
	wget https://pjreddie.com/media/files/darknet53.conv.74
	
	# train
	./darknet detector train flower_detection_minorPJ/flower_voc.data flower_detection_minorPJ/yolov3-voc.cfg darknet53.conv.74 
	
	# test
	./darknet detector test flower_detection_minorPJ/flower_voc.data flower_detection_minorPJ/yolov3-voc.cfg flower_detection_minorPJ/backup/yolov3-voc_final.weights flower_detection_minorPJ/test_data/test_img.jpg
	
	# limited by github, I cannot upload my weights.
	# You can contact me if you want
	# 16300290001@fudan.edu.cn


### The fourth step: Crop the images into several segments

To crop images by the bounding boxes and temporarily save the class information.

    
    cd flower_detection_minorPJ
    vi cropBox.py 


### The fifth step: Train the Residual Neural Network

While predicting:
	input: an particular enlarged image
	return: the class information
	
	# train
	cd flower_detection_minorPJ
	python3 model_2.py train resnet18_100.pkl
	
	# test
	python3 model_2.py test resnet18_best3.pkl flower_test


### The last step: Outcome and visualization

##### Resnet-18 loss function
![Aaron Swartz](https://github.com/yyf710670079/flower_detection_minorPJ/raw/master/result/resnet18_loss.png)

##### YOLOv3 loss function
![Aaron Swartz](https://github.com/yyf710670079/flower_detection_minorPJ/raw/master/result/yolov3_loss.png)

##### Test accuracy of Resnet
![Aaron Swartz](https://github.com/yyf710670079/flower_detection_minorPJ/raw/master/result/RES.jpeg)

##### Train accuracy of Resnet
![Aaron Swartz](https://github.com/yyf710670079/flower_detection_minorPJ/raw/master/result/RES2.jpeg)

##### YOLOv3 test RECALL
![Aaron Swartz](https://github.com/yyf710670079/flower_detection_minorPJ/raw/master/result/YOLO.jpeg)


https://github.com/yyf710670079/flower_detection_minorPJ/raw/master/test%20images/flower_test_latex.001.jpeg

##### TEST IMAGES
![Aaron Swartz](https://github.com/yyf710670079/flower_detection_minorPJ/raw/master/test%20images/flower_test_latex.001.jpeg)