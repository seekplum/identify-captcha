# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#=============================================================================
#  ProjectName: identify-captcha
#     FileName: common
#         Desc: 
#       Author: hjd
#     HomePage: seekplun.github.io
#   LastChange: 2018-03-31 01:24
#=============================================================================
"""

import os

from threading import Thread, Lock

import numpy as np

from train_cnn import text2vec, origin_folder, convert2gray, get_image_array
from train_cnn import IMAGE_HEIGHT, IMAGE_WIDTH, CHAR_SET_LEN, MAX_CAPTCHA


def get_next_batch():
    """批量处理训练

    :return batch_x
    :rtype batch_x numpy.ndarray

    :return batch_y
    :rtype batch_y numpy.ndarray
    """
    origin_files = os.listdir(origin_folder)
    batch_size = len(origin_files)
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    for i, file_name in enumerate(origin_files):
        text = file_name.rsplit(".", 1)[0]
        file_path = os.path.join(origin_folder, file_name)
        image = get_image_array(file_path)
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def get_image_text_vector(batch_x, batch_y, index, file_name, lock):
    """获取一个图片向量

    :param batch_x
    :type batch_x numpy.ndarray

    :param batch_y
    :type batch_y numpy.ndarray

    :param index: 下标值
    :type index int

    :param file_name: 图片名
    :type file_name str

    :param lock: 线程锁
    """

    text = file_name.rsplit(".", 1)[0]
    file_path = os.path.join(origin_folder, file_name)

    image = get_image_array(file_path)
    image = convert2gray(image)

    with lock:
        batch_x[index, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[index, :] = text2vec(text)


def get_next_batch_thread():
    """批量处理训练

    :return batch_x
    :rtype batch_x numpy.ndarray

    :return batch_y
    :rtype batch_y numpy.ndarray
    """
    origin_files = os.listdir(origin_folder)
    batch_size = len(origin_files)
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    lock = Lock()
    threads = []
    for index, file_name in enumerate(origin_files):
        thread = Thread(target=get_image_text_vector, args=(batch_x, batch_y, index, file_name, lock,))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return batch_x, batch_y
