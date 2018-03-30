# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#=============================================================================
#  ProjectName: identify-captcha
#     FileName: test_train_cnn
#         Desc: 
#       Author: hjd
#     HomePage: seekplun.github.io
#   LastChange: 2018-03-31 01:07
#=============================================================================
"""
import os
import time

import pytest
import numpy as np
from generate_captcha import gen_captcha_text_image

from common import get_next_batch, get_next_batch_thread
from train_cnn import text2vec, vec2text, origin_folder, SIZE, print_info

CAPTCHA_COUNT = 30


@pytest.mark.parametrize("text", [
    "F5Sd",
    "SFd5",
    "Md19"
])
def test_text_to_vec(text):
    """测试文本转vec

    :param text: 字符串
    :type text str
    """
    vec = text2vec(text)

    assert isinstance(vec, np.ndarray)

    assert vec2text(vec) == text


def test_generate_captcha():
    """测试生成验证码是否成功
    """
    for i in range(CAPTCHA_COUNT):
        text, file_path = gen_captcha_text_image(origin_folder, size=SIZE)
        file_name = os.path.basename(file_path)

        assert os.path.exists(file_path)
        assert text == file_name.rsplit(".", 1)[0]


def test_get_next_batch():
    """测试使用多线程下和单线程时间消耗
    """
    for i in range(CAPTCHA_COUNT):
        start_time = time.time()
        batch_x, batch_y = get_next_batch()
        end_time = time.time()

        start_time2 = time.time()
        thread_batch_x, thread_batch_y = get_next_batch_thread()
        end_time2 = time.time()
        interval_time = end_time - start_time
        interval_time2 = end_time2 - start_time2

        # 检查数据是否一致
        assert all([all(r) for r in thread_batch_x == batch_x])
        assert all([all(r) for r in thread_batch_y == batch_y])

        # 检查时间是否更短
        print_info("interval time: {}, time2: {}".format(interval_time, interval_time2))
        assert interval_time2 < interval_time
