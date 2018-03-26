#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#=============================================================================
#  ProjectName: seekplum
#     FileName: config
#         Desc: 配置文件
#       Author: hjd
#     HomePage: seekplun.github.io
#   LastChange: 2018-03-24 02:28
#=============================================================================
"""

import os

curr_path = os.path.dirname(os.path.abspath(__file__))

# 下载验证码网址
captcha_url = "http://www.xxx.com/device/validate_code"

# 验证码长度
captcha_length = 4

# ================================ 所有图片的训练目录 ================================
# 数据目录
data_root = os.path.join(os.path.dirname(curr_path), "captcha")

# 原始图像目录
origin_pic_folder = os.path.join(data_root, "origin")

# "原始图像 -> 二值 -> 除噪声" 之后的图片文件目录
bin_clear_folder = os.path.join(data_root, "bin_clear")

# 1张4位验证字符图片 -> 4张单字符图片。然后再将相应图片拖动到指定目录，完全数据标记工作
cut_pic_folder = os.path.join(data_root, "cut_pic")

# 测试验证码分割目录
cut_test_folder = os.path.join(data_root, "cut_test")

# 识别验证码结果目录
identify_result_folder = os.path.join(data_root, "identify")

# ================================ SVM训练相关路径 ================================

# 用于SVM训练的特征文件
svm_root = os.path.join(data_root, "svm_train")

# 保存训练集的 像素特征文件
train_file_name = os.path.join(svm_root, "train_pix_feature_xy.txt")

# 只以一组数字(比如 4)的特征文件为例子来做简单的验证测试
test_feature_file = os.path.join(svm_root, "last_test_pix_xy_{}.txt")

# 训练完毕后，保存的SVM模型参数文件
model_path = os.path.join(svm_root, "svm_model_file")

# ================================ 字体 ================================
monaco_ttf = os.path.join(curr_path, "msyh.ttf")
