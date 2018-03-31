#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#=============================================================================
#  ProjectName: identify-captcha
#     FileName: train_cnn
#         Desc: 通过卷积神经网络进行识别验证码
#               cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
#               # 在图像上补2行，下补3行，左补2行，右补2行
#               np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))
#       Author: hjd
#     HomePage: seekplum.github.io
#   LastChange: 2018-03-29 20:17
#=============================================================================
"""
import os
import uuid

import numpy as np
import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt

from generate_captcha import gen_captcha_text_image
from generate_captcha import NUMBERS, CAPITAL_LETTERS, LOWERCASE_LETTERS

IMAGE_HEIGHT = 60  # 验证码高度
IMAGE_WIDTH = 160  # 验证码宽度
MAX_CAPTCHA = 4  # 验证码最长4字符; 全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐
SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
CHAR_SET_LEN = len(NUMBERS + CAPITAL_LETTERS + LOWERCASE_LETTERS)  # 文本转向量
TARGET_ACCURACY = 0.5  # 目标正确率
COUNT_INTERNAL = 100  # 计算次数间隔
BATCH_NUMBER = 64  # 批量生成图片数量

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout

curr_path = os.path.dirname(os.path.abspath(__file__))
mode_folder = os.path.join(os.path.dirname(curr_path), "captcha", "cnn_train")
mode_path = os.path.join(mode_folder, "crack_captcha.model")  # 模型文件路径
origin_folder = os.path.join(os.path.dirname(curr_path), "captcha", "origin")  # 生成的验证码路径


class TrainCnnError(Exception):
    """cnn训练异常
    """
    pass


class CaptchaLengthError(TrainCnnError):
    """验证码长度错误
    """
    pass


class CharSetError(TrainCnnError):
    """字符ascii码错误
    """
    pass


def create_folder():
    """创建目录
    """
    paths = [
        mode_folder,
        origin_folder
    ]
    for file_path in paths:
        if not os.path.exists(file_path):
            os.makedirs(file_path)


def print_info(text, newline=True):
    """打印信息

    :param text: 打印内容
    :type text any

    :param newline: 是否换行
    :type newline bool
    """
    if newline:
        print text
    else:
        print text,


def convert2gray(image):
    """把彩色图像转为灰度图像

    :param image 图片对象
    :type image numpy.ndarray

    :rtype image numpy.ndarray
    """
    if len(image.shape) > 2:
        image = np.mean(image, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        # image = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return image
    else:
        return image


def text2vec(text):
    """文本转向量

    向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符

    :param text: 字符串
    :type text str

    :rtype vector list
    :return: vector 向量列表
    """
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        # 验证码超过最大长度
        raise CaptchaLengthError("The verification code has a maximum of {} characters.".format(MAX_CAPTCHA))

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c_):
        if c_ == '_':
            k = 62
            return k
        k = ord(c_) - 48
        if k > 9:
            k = ord(c_) - 55
            if k > 35:
                k = ord(c_) - 61
                if k > 61:
                    raise CharSetError("{} must be less than {}".format(k, CHAR_SET_LEN))
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


def vec2text(vector):
    """向量转回文本

    :param: vector 向量列表
    :type vector numpy.ndarray

    :return text 字符串内容
    :rtype text str
    """
    char_pos = vector.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise CharSetError("{} must be less than {}".format(char_idx, CHAR_SET_LEN))
        text.append(chr(char_code))
    return "".join(text)


def wrap_gen_captcha_text_and_image():
    """生成验证码

    :rtype text str
    :return text 验证码内容

    :rtype file_path str
    :return file_path 验证码路径

    :rtype image numpy.ndarray
    :return image 图片二维数组
    """
    text, file_path = gen_captcha_text_image(origin_folder, size=SIZE, file_name=uuid.uuid4().hex)
    image = get_image_array(file_path)
    return text, file_path, image


def get_image_array(file_path):
    """查询图片对象

    :param file_path: 图片路径
    :type file_path str

    :rtype image numpy.ndarray
    :return image 图片二维数组
    """
    captcha_image = Image.open(file_path)
    image = np.array(captcha_image)
    return image


def get_next_batch(batch_size):
    """批量处理训练

    通过测试发现，多进程和多线程的情况下，速度并没有更快

    :param batch_size: 验证码数量
    :type batch_size int

    :return batch_x
    :rtype batch_x numpy.ndarray

    :return batch_y
    :rtype batch_y numpy.ndarray
    """
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    for i in range(batch_size):
        text, file_path, image = wrap_gen_captcha_text_and_image()

        # 删除文件
        os.remove(file_path)

        if i % 10 == 0:
            print_info("\ngenerate captcha: {}".format(text), newline=False)
        else:
            print_info(text, newline=False)
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    """定义CNN

    :param w_alpha: 张量系数
    :type w_alpha float

    :param b_alpha: 张量系数
    :type b_alpha float

    :rtype tensorflow.python.framework.ops.Tensor
    """
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    con_v1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    con_v1 = tf.nn.max_pool(con_v1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    con_v1 = tf.nn.dropout(con_v1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    con_v2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(con_v1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    con_v2 = tf.nn.max_pool(con_v2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    con_v2 = tf.nn.dropout(con_v2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    con_v3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(con_v2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    con_v3 = tf.nn.max_pool(con_v3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    con_v3 = tf.nn.dropout(con_v3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(con_v3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


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
        saver.restore(sess, tf.train.latest_checkpoint(mode_folder))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * CHAR_SET_LEN + n] = 1
            i += 1
        return vec2text(vector)


def train_captcha():
    """训练数据

    :return loss_step_data: 训练次数 y轴信息
    :rtype loss_step_data list

    :return loss_data: loss次数 x轴信息
    :rtype loss_data list

    :return accuracy_step_data: 训练次数 y轴信息
    :rtype accuracy_step_data list

    :return accuracy_data: 识别率 x轴信息
    :rtype accuracy_data list
    """
    output = crack_captcha_cnn()
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    loss_data = []
    loss_step_data = []
    accuracy_step_data = []
    accuracy_data = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(BATCH_NUMBER)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print_info("\n\nstep: {}, loss: {}".format(step, loss_))
            loss_data.append(loss_)
            loss_step_data.append(step)
            # 每固定间隔计算一次准确率
            if step % COUNT_INTERNAL == 0:
                batch_x_test, batch_y_test = get_next_batch(BATCH_NUMBER)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print_info("\nstep: {}, acc: {}".format(step, acc))
                accuracy_step_data.append(step)
                accuracy_data.append(acc)
                # 如果准确率大于 目标正确率 ,保存模型,完成训练
                if acc > TARGET_ACCURACY:
                    saver.save(sess, mode_path, global_step=step)
                    break

            step += 1
    return loss_step_data, loss_data, accuracy_step_data, accuracy_data


def save_line_chart(loss_step_data, loss_data, accuracy_step_data, accuracy_data):
    """保存折线图

    :param loss_step_data: 训练次数 y轴信息
    :type loss_step_data list

    :param loss_data: loss次数 x轴信息
    :type loss_data list

    :param accuracy_step_data: 训练次数 y轴信息
    :type accuracy_step_data list

    :param accuracy_data: 识别率 x轴信息
    :type accuracy_data list
    """
    plt.figure()
    plt.plot(loss_step_data, loss_data, "b--", linewidth=1)  # 设置 x/y轴数据，线的颜色/虚线，线的宽度
    plt.xlabel("loss")
    plt.ylabel("step")
    plt.title("Loss/Step")
    loss_path = os.path.join(mode_folder, "loss.png")
    plt.savefig(loss_path)
    plt.figure()
    plt.plot(accuracy_step_data, accuracy_data, "b--", linewidth=1)
    plt.xlabel("accuracy")
    plt.ylabel("step")
    plt.title("Accuracy/Step")
    loss_path = os.path.join(mode_folder, "accuracy.png")
    plt.savefig(loss_path)


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
    # 训练数据
    create_folder()
    loss_step_data, loss_data, accuracy_step_data, accuracy_data = train_captcha()
    save_line_chart(loss_step_data, loss_data, accuracy_step_data, accuracy_data)

    # # 预测识别验证码
    # identify_captcha()
