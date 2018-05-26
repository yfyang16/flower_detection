#   _*_ coding:utf-8 _*_
__author__ = 'yangyufeng'

"""
yolo训练准备:
将图片的绝对路径改为与服务器的前缀一致

"""

new_content = []
with open("/Users/yangyufeng/Desktop/flower_detection/flower_train.txt", 'r') as f3:
    lines = f3.readlines()

    for line in lines:
        new_line = "/home/yyh/workspace_yyf/darknet/flower_detection/JPEGImages/" + line.split('/')[-1]
        new_content.append(new_line)

with open("/Users/yangyufeng/Desktop/flower_detection/flower_train1.txt", 'w') as f4:
    f4.writelines(new_content)


new_content2 = []
with open("/Users/yangyufeng/Desktop/flower_detection/flower_val.txt", 'r') as f5:
    lines = f5.readlines()

    for line in lines:
        new_line = "/home/yyh/workspace_yyf/darknet/flower_detection/JPEGImages/" + line.split('/')[-1]
        new_content2.append(new_line)

with open("/Users/yangyufeng/Desktop/flower_detection/flower_val1.txt", 'w') as f6:
    f6.writelines(new_content2)