## README(about the whole process)

#######################################################################################
Python Package, Tool Source and Data Source
#######################################################################################
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import os
import torchvision.transforms as T
import torch
from ann_visualizer.visualize import ann_viz

# lable tool
git clone https://github.com/tzutalin/labelImg.git

# kaggle dataset
kaggle datasets download -d alxmamaev/flowers-recognition

# YOLO architecture
git clone https://github.com/pjreddie/darknet

# Resnet34 architecture
We make this by tensorflow.
The reference is at the bottom.




#######################################################################################
The first step: Reprocess the data
#######################################################################################
1) Use Python to rename all the picture in order to make labeling convenient.
   (number_train_imagedir.jpg, number_test_imagedir.jpg)

2) Data augmentation: Flip and Random Crop by using python PIL package.



#######################################################################################
The second step: Label the image
#######################################################################################
1) Use LabelImg to label the dataset and generate a file including cordinates and classes.

2) Pack the whole label text and match each picture.


#######################################################################################
The third step: Train the Yolo model
#######################################################################################
While predicting:
	input : 3 x height x width image
	return: bounding boxes' cordinates and corresponding class information




#######################################################################################
The fourth step: Crop the images into several segments
#######################################################################################
To crop images by the bounding boxes and temporarily save the class information




#######################################################################################
The fifth step: Train the Residual Neural Network
#######################################################################################
While predicting:
	input: an particular enlarged image
	return: the class information




#######################################################################################
The last step: Outcome and visualization
#######################################################################################





#######################################################################################
Reference
#######################################################################################
[1]Redmon J, Divvala S, Girshick R, et al. You Only Look Once: Unified, Real-Time Object Detection[C]// IEEE Conference on Computer Vision and Pattern Recognition. IEEE Computer Society, 2016:779-788.

[2]Redmon J, Farhadi A. YOLO9000: Better, Faster, Stronger[J]. 2016:6517-6525.

[3] R. Girshick. Fast R-CNN. arXiv:1504.08083, 2015. 

[4] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014. 

[5] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015. 

[6]K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition, arXiv. cs.CV (2015).

[7]S. Ren, K. He, R. Girshick, J. Sun, Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, arXiv. cs.CV (2015).