#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
#=============================================================================
#  ProjectName: seekplum
#     FileName: identify_code
#         Desc: 训练模型之后测试
#                1. 从网站动态请求相应的验证文件
#                2. 进行图像预处理
#                3. 将图像进行分割成最小基本单位
#                4. 计算出本图像的特征
#                5. 使用SVM训练好的模型进行
#                6. 对新的验证图片来做结果预测
#       Author: hjd
#     HomePage: seekplun.github.io
#   LastChange: 2018-03-24 02:28
#=============================================================================
"""

import os

from PIL import Image
from svmutil import svm_predict, svm_load_model

from captcha_code import gen_captcha_text_image
from config import model_path, origin_pic_folder, identify_result_folder
from svm_features import get_feature, convert_feature_to_vector
from utils import get_clear_bin_image, get_crop_images


def crack_identify_captcha(suffix="png"):
    """破解验证码,完整的演示流程

    :param suffix: 图片后缀名
    :type suffix str
    """
    # 生成数字验证码图片
    text, image_path = gen_captcha_text_image(origin_pic_folder, draw_lines=True, draw_points=True)
    img = Image.open(image_path)

    # 降噪的获取二值图
    bin_img = get_clear_bin_image(img)

    child_img_list = get_crop_images(bin_img)

    # 加载SVM模型进行预测
    model = svm_load_model(model_path)

    img_ocr_names = []
    for index, child_img in enumerate(child_img_list):
        # child_img.save("{}.png".format(index))

        # 使用特征算法，将图像进行特征化降维
        img_feature_list = get_feature(child_img)

        yt = [0]  # 测试数据标签
        # xt = [{1: 1, 2: 1}]  # 测试数据输入向量

        # 将所有的特征转化为标准化的SVM单行的特征向量
        xt = convert_feature_to_vector(img_feature_list)

        # p_label即为识别的结果
        p_label, p_acc, p_val = svm_predict(yt, xt, model)

        # 将识别结果合并起来
        img_ocr_names.append(str(int(p_label[0])))

    # 识别结果
    result = "".join(img_ocr_names)
    print "result: {}, text: {}".format(result, text)

    # 保存的文件名
    file_name = "{}.{}".format(result, suffix)
    file_path = os.path.join(identify_result_folder, file_name)

    # 保存图片
    img.save(file_path)


def crack_download_captcha(number=1):
    """自动生成验证码图片，然后识别出来

    :param number: 生成图片张数
    :type number int
    """
    for i in range(number):
        crack_identify_captcha()


if __name__ == '__main__':
    crack_download_captcha()
