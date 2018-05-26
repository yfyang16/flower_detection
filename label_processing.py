#   _*_ coding:utf-8 _*_
__author__ = 'yangyufeng'
import os

daisy_label_dir = os.getcwd() + "/daisy_label"
new_daisy_label_dir = os.getcwd() + "/new_daisy_label"

tulip_label_dir = os.getcwd() + "/tulip_label"
new_tulip_label_dir = os.getcwd() + "/new_tulip_label"

rose_label_dir = os.getcwd() + "/rose_label"
new_rose_label_dir = os.getcwd() + "/new_rose_label"

sunflower_label_dir = os.getcwd() + "/sunflower_label"
new_sunflower_label_dir = os.getcwd() + "/new_sunflower_label"

def label_processing(label_dir, new_dir, class_num_str):
    """
    用labelImg标注时，前面有自带的类别比如人和车之类，而我们的训练中只有花，
    所以将所有标注文件（txt）中的类别序号依次改成1，2，3，4...
    :param label_dir:
    :param new_dir:
    :param class_num_str:
    :return:
    """
    files = os.listdir(label_dir)
    file_num = len(files)

    for i in range(1, file_num):
        file_path = label_dir + '/' + files[i]
        content_temp = []
        try:
            with open(file_path, 'r') as f:
                all_lines = f.readlines()

                for line in all_lines:
                    line_split = line.split(' ')
                    line_split[0] = class_num_str
                    line = ' '.join(line_split)
                    content_temp.append(line)
        except BaseException:
            print("第" + str(i) + "个文件读取有问题")
            break

        new_file_path = new_dir + '/' + files[i]
        with open(new_file_path, 'w', encoding='utf-8') as nf:
            nf.writelines(content_temp)


label_processing(daisy_label_dir, new_daisy_label_dir, '1')
label_processing(tulip_label_dir, new_tulip_label_dir, '2')
label_processing(rose_label_dir, new_rose_label_dir, '3')
label_processing(sunflower_label_dir, new_sunflower_label_dir, '4')