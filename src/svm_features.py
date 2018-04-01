#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
#=============================================================================
#  ProjectName: identify-captcha
#     FileName: svm_features
#         Desc: 获取图片特征值
#       Author: hjd
#     HomePage: seekplun.github.io
#   LastChange: 2018-03-24 02:28
#=============================================================================
"""
import os

from PIL import Image
from svmutil import svm_read_problem, svm_save_model
from svmutil import svm_train

from config import train_file_name, cut_pic_folder, svm_model_path


def get_feature(img):
    """获取指定图片的特征值,
        1. 按照每排的像素点,高度为10,然后宽度为6,总共16个维度
        2. 计算每个维度（行 或者 列）上有效像素点的和

    :param img: 图片对象
    :type img PIL.Image.Image

    :rtype pixel_cnt list
    :return: 一个维度为16的列表
    """
    width, height = img.size

    pixel_cnt = []
    height = 10
    for y in range(height):
        pix_cnt_x = 0
        for x in range(width):
            if img.getpixel((x, y)) == 0:  # 黑色点
                pix_cnt_x += 1

        pixel_cnt.append(pix_cnt_x)

    for x in range(width):
        pix_cnt_y = 0
        for y in range(height):
            if img.getpixel((x, y)) == 0:  # 黑色点
                pix_cnt_y += 1

        pixel_cnt.append(pix_cnt_y)

    return pixel_cnt


def get_svm_train_txt():
    """获取 测试集 的像素特征文件

    所有的数字的可能分类为10，分别放在以相应的数字命名的目录中
    """
    with open(train_file_name, 'w') as f:
        for i in range(10):
            img_folder = os.path.join(cut_pic_folder, str(i))
            # 不断地以追加的方式写入到同一个文件当中
            convert_images_to_feature_file(i, f, img_folder)


def convert_images_to_feature_file(dig, svm_feature_file, img_folder):
    """将某个目录下二进制图片文件，转换成特征文件

    :param dig: 检查的数字
    :type dig int

    :param svm_feature_file: svm的特征文件完整路径
    :type svm_feature_file fp

    :param img_folder: 图片路径
    :type img_folder str
    """
    # 获取 img_folder 目录下所有文件
    file_list = os.listdir(img_folder)

    for file_name in file_list:
        img = Image.open(os.path.join(img_folder, file_name))
        dif_list = get_feature(img)
        # sample_cnt += 1
        line = convert_values_to_str(dig, dif_list)
        svm_feature_file.write(line)
        svm_feature_file.write('\n')


def convert_values_to_str(dig, dif_list):
    """将特征值串转化为标准的svm输入向量

    9 1:4 2:2 3:2 4:2 5:3 6:4 7:1 8:1 9:1 10:3 11:5 12:3 13:3 14:3 15:3 16:6

    最前面的是 标记值，后续是特征值
    :param dig: 标记置
    :type dig int

    :param dif_list: 一个维度为16的列表
    :type dif_list list

    :return line 标准的svm输入向量
    :rtype line str
    """
    line = [str(dig)]

    for index, item in enumerate(dif_list, start=1):
        fmt = ' %d:%d' % (index, item)
        line.append(fmt)

    return "".join(line)


def convert_feature_to_vector(feature_list):
    """将所有的特征转化为标准化的SVM单行的特征向量

    :param feature_list: 一个维度为16的列表
    :type feature_list list

    :return: xt_vector 标准化的SVM单行的特征向量
    :rtype: xt_vector list
    """
    xt_vector = []
    feature_dict = {}
    for index, item in enumerate(feature_list, start=1):
        feature_dict[index] = item
    xt_vector.append(feature_dict)
    return xt_vector


def train_svm_model():
    """训练并生成model文件
    """
    y, x = svm_read_problem(train_file_name)
    model = svm_train(y, x)
    svm_save_model(svm_model_path, model)
