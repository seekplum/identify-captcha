# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#=============================================================================
#  ProjectName: identify-captcha
#     FileName: identify_cnn
#         Desc: 通过cnn训练后的数据预测验证码
#       Author: hjd
#     HomePage: seekplun.github.io
#   LastChange: 2018-04-01 18:30
#=============================================================================
"""
import numpy as np
import tensorflow as tf

from config import MAX_CAPTCHA
from config import cnn_root
from train_cnn import X, CHAR_SET_LEN
from train_cnn import wrap_gen_captcha_text_and_image, convert2gray, crack_captcha_cnn
from train_cnn import vec2text, keep_prob
from utils import print_info


def crack_captcha(image):
    """预测验证码内容

    :param image: 图片对象
    :type image numpy.ndarray

    :return 预测结果
    :rtype str
    """
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(cnn_root))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * CHAR_SET_LEN + n] = 1
            i += 1
        return vec2text(vector)


def identify_captcha():
    """识别验证码
    """
    text, _, image = wrap_gen_captcha_text_and_image()

    # 二值化
    image = convert2gray(image)
    image = image.flatten() / 255

    predict_text = crack_captcha(image)
    print_info("text: {}  crack: {}".format(text, predict_text))


if __name__ == '__main__':
    # 预测识别验证码
    identify_captcha()
