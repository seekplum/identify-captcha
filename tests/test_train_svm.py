# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#=============================================================================
#  ProjectName: identify-captcha
#     FileName: test_train_svm
#         Desc: 测试svm模型
#       Author: hjd
#     HomePage: seekplun.github.io
#   LastChange: 2018-04-01 17:40
#=============================================================================
"""
import os

from PIL import Image
from generate_captcha import gen_captcha_text_image

from config import cut_test_folder, captcha_xy
from utils import generate_alias_name, get_clear_bin_image
from utils import get_crop_images


def test_cut_pic():
    """测试图片分割
    """
    file_name = "test"
    suffix = "png"
    length = 4
    text, image_path = gen_captcha_text_image(cut_test_folder, file_name=file_name, suffix=suffix,
                                              draw_lines=True, draw_points=True,
                                              length=length, xy=captcha_xy)
    img = Image.open(image_path)

    assert os.path.exists(image_path)

    assert "{}.{}".format(file_name, suffix) == os.path.basename(image_path)

    # 获取干净的二值化的图片
    img = get_clear_bin_image(img)
    new_image_path = generate_alias_name(image_path, "_bin")
    img.save(new_image_path)

    child_img_list = get_crop_images(img)

    assert length == len(child_img_list)

    for index, child_img in enumerate(child_img_list):
        file_name = "cut-{}.png".format(index)
        file_path = os.path.join(cut_test_folder, file_name)
        child_img.save(file_path)

        assert os.path.exists(file_path)
