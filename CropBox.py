#   _*_ coding:utf-8 _*_
__author__ = 'yangyufeng'

from PIL import Image


def change_yolo_cordinates_order(yolo_cordinates_tuple):

    """

    PIL和YOLO的bounding box的坐标顺序有区别

    :param:   yolo_cordinates_tuple
    :return:  PIL_cordinate_tuple

    """
    left = yolo_cordinates_tuple[0]
    right = yolo_cordinates_tuple[2]
    top = yolo_cordinates_tuple[1]
    bot = yolo_cordinates_tuple[3]

    PIL_order = (left, right, top, bot)

    return PIL_order


def cropBox(im_path, yolo_cordinates_tuple):
    """
    将yolo预测出来的bounding box抠出来
    :param im_path:
    :param yolo_cordinates_tuple:
    :return:
    """
    box = change_yolo_cordinates_order(yolo_cordinates_tuple)

    im = Image.open(im_path)
    new_im = im.crop(box)

    return new_im
