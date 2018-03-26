#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#=============================================================================
#  ProjectName: identify-captcha
#     FileName: utils
#         Desc: 配置文件
#       Author: hjd
#     HomePage: seekplun.github.io
#   LastChange: 2018-03-24 02:28
#=============================================================================
"""
import os
import uuid

import requests

from PIL import Image

from config import captcha_url, captcha_length


def download_captcha(number, image_path, suffix="png"):
    """下载验证码图片

    :param number: 要下载的张数
    :type number int

    :param image_path: 保存的路径
    :type image_path str

    :param suffix: 图片后缀
    :type suffix str
    """
    # 防止图片名字冲突标记
    name_tag = uuid.uuid4().hex

    for i in range(number):
        # 访问url
        response = requests.get(captcha_url, stream=True)

        file_name = "{}_{}.{}".format(name_tag, i, suffix)
        file_path = os.path.join(image_path, file_name)

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()


def get_bin_table(threshold=200):
    """获取灰度转二进值的映射table

    注意: 对于验证码中字体不同颜色,阀值会不同,需要调整

    :param threshold: 阀值
    :type threshold int

    :rtype table list
    :return: table 二进值的映射table
    """
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    return table


def sum_9_region(img, x, y):
    """9邻域框,以当前点为中心的田字框,黑点个数,作为移除一些孤立的点的判断依据

    :param img: Image对象
    :type img PIL.Image.Image

    :param x: 横坐标
    :type x int

    :param y: 纵坐标
    :type y int

    :rtype region int
    :return: region 中心点数量
    """
    cur_pixel = img.getpixel((x, y))  # 当前像素点的值
    width, height = img.size

    if cur_pixel == 1:  # 如果当前点为白色区域,则不统计邻域值
        region = 0
        return region

    if y == 0:  # 第一行
        if x == 0:  # 左上顶点,4邻域
            # 中心点旁边3个点
            sum_ = cur_pixel \
                   + img.getpixel((x, y + 1)) \
                   + img.getpixel((x + 1, y)) \
                   + img.getpixel((x + 1, y + 1))
            region = 4 - sum_
        elif x == width - 1:  # 右上顶点
            sum_ = cur_pixel \
                   + img.getpixel((x, y + 1)) \
                   + img.getpixel((x - 1, y)) \
                   + img.getpixel((x - 1, y + 1))

            region = 4 - sum_
        else:  # 最上非顶点,6邻域
            sum_ = img.getpixel((x - 1, y)) \
                   + img.getpixel((x - 1, y + 1)) \
                   + cur_pixel \
                   + img.getpixel((x, y + 1)) \
                   + img.getpixel((x + 1, y)) \
                   + img.getpixel((x + 1, y + 1))
            region = 6 - sum_
    elif y == height - 1:  # 最下面一行
        if x == 0:  # 左下顶点
            # 中心点旁边3个点
            sum_ = cur_pixel \
                   + img.getpixel((x + 1, y)) \
                   + img.getpixel((x + 1, y - 1)) \
                   + img.getpixel((x, y - 1))
            region = 4 - sum_
        elif x == width - 1:  # 右下顶点
            sum_ = cur_pixel \
                   + img.getpixel((x, y - 1)) \
                   + img.getpixel((x - 1, y)) \
                   + img.getpixel((x - 1, y - 1))

            region = 4 - sum_
        else:  # 最下非顶点,6邻域
            sum_ = cur_pixel \
                   + img.getpixel((x - 1, y)) \
                   + img.getpixel((x + 1, y)) \
                   + img.getpixel((x, y - 1)) \
                   + img.getpixel((x - 1, y - 1)) \
                   + img.getpixel((x + 1, y - 1))
            region = 6 - sum_
    else:  # y不在边界
        if x == 0:  # 左边非顶点
            sum_ = img.getpixel((x, y - 1)) \
                   + cur_pixel \
                   + img.getpixel((x, y + 1)) \
                   + img.getpixel((x + 1, y - 1)) \
                   + img.getpixel((x + 1, y)) \
                   + img.getpixel((x + 1, y + 1))

            region = 6 - sum_
        elif x == width - 1:  # 右边非顶点
            sum_ = img.getpixel((x, y - 1)) \
                   + cur_pixel \
                   + img.getpixel((x, y + 1)) \
                   + img.getpixel((x - 1, y - 1)) \
                   + img.getpixel((x - 1, y)) \
                   + img.getpixel((x - 1, y + 1))

            region = 6 - sum_
        else:  # 具备9领域条件的
            sum_ = img.getpixel((x - 1, y - 1)) \
                   + img.getpixel((x - 1, y)) \
                   + img.getpixel((x - 1, y + 1)) \
                   + img.getpixel((x, y - 1)) \
                   + cur_pixel \
                   + img.getpixel((x, y + 1)) \
                   + img.getpixel((x + 1, y - 1)) \
                   + img.getpixel((x + 1, y)) \
                   + img.getpixel((x + 1, y + 1))
            region = 9 - sum_
    return region


def remove_noise_pixel(img, noise_point_list):
    """根据噪点的位置信息，消除二值图片的黑点噪声

    :param img: Image图片对象
    :type img PIL.Image.Image

    :param noise_point_list: 噪点列表
    :type noise_point_list list
    """
    for item in noise_point_list:
        img.putpixel((item[0], item[1]), 1)


def get_clear_bin_image(image):
    """获取干净的二值化的图片

        图像的预处理：
        1. 先转化为灰度
        2. 再二值化
        3. 然后清除噪点
        参考:http://python.jobbole.com/84625/

    :param image: Image图片对象
    :type image PIL.Image.Image

    :rtype img PIL.Image.Image
    :return: img 处理后的Image图片对象
    """
    img_gray = image.convert('L')  # 转化为灰度图

    table = get_bin_table()
    img = img_gray.point(table, '1')  # 变成二值图片:0表示黑色,1表示白色

    width, height = img.size
    noise_point_list = []  # 通过算法找出噪声点,第一步比较严格,可能会有些误删除的噪点
    for x in range(width):
        for y in range(height):
            res_9 = sum_9_region(img, x, y)
            if (0 < res_9 < 3) and img.getpixel((x, y)) == 0:  # 找到孤立点
                pos = (x, y)
                noise_point_list.append(pos)
    remove_noise_pixel(img, noise_point_list)
    return img


def get_crop_images(img, number=captcha_length):
    """按照图片的特点,进行切割,这个要根据具体的验证码来进行工作

    分割图片是传统机器学习来识别验证码的重难点，如果这一步顺利的话，则多位验证码的问题可以转化为1位验证字符的识别问题

    :param img: Image图片对象
    :type img Image图片对象

    :param number: 验证码字符数
    :type number int

    :return: child_img_list 图片对象列表
    :rtype child_img_list list
    """
    # # 参数会和验证码的长宽有关
    # width, height = img.size
    #
    # # 头尾的空白长度
    # head_blank = 2
    # tail_blank = 29
    #
    # width -= tail_blank
    #
    # child_img_list = []
    # avg_width = width / number
    # y = 0
    # for i in range(number):
    #     # 计算每个字符的对角坐标
    #     x = head_blank + i * avg_width
    #     child_img = img.crop((x, y, x + avg_width, y + height))
    #     child_img_list.append(child_img)

    # 切分之后的图片列表
    child_img_list = []

    # TODO: hjd 切割参数会和验证码的长宽有关,需要进一步优化
    # 每个字符位置,只适用于验证码比较规整的情况下,后续需要根据算法识别出来
    # svm训练的难点在于如何分割出正确的数据，如果训练出的数据都有问题，后续识别肯定无法实现
    positions = [
        (5, 3, 25, 38),
        (48, 3, 68, 38),
        (90, 3, 110, 38),
        (132, 3, 151, 38)
    ]
    for item in positions:
        x, y, x1, y1 = item
        child_img = img.crop((x, y, x1, y1))
        child_img_list.append(child_img)

    return child_img_list


def print_line_x(img, x):
    """打印一个Image图像的第x行，方便调试

    :param img: Image图片对象
    :type img Image图片对象

    :param x: 行数
    :type x int
    """
    print "line: {}".format(x)
    for w in range(img.width):
        print "w: {} {}".format(w, img.getpixel((w, x))),


def save_crop_images(bin_clear_image_path, child_img_list, cut_path):
    """保存切割的图片
        输入：整个干净的二化图片
        输出：每张切成4版后的图片集

        例如： A.png ---> A-1.png,A-2.png,... A-4.png 并保存，这个保存后需要去做label标记的

    :param bin_clear_image_path: xxxx/xxxxx/xxxxx.png 主要是用来提取切割的子图保存的文件名称
    :type bin_clear_image_path str

    :param child_img_list: 图片对象列表
    :type child_img_list list

    :param cut_path: 切割后图片保存路径
    :type cut_path str
    """
    full_file_name = os.path.basename(bin_clear_image_path)  # 文件名称
    full_file_name_split = full_file_name.split('.')
    file_name = full_file_name_split[0]
    # file_ext = full_file_name_split[1]

    for i, child_img in enumerate(child_img_list):
        cut_img_file_name = file_name + '-' + ("%s.png" % i)
        child_img.save(os.path.join(cut_path, cut_img_file_name))


# ================================== 训练素材准备 ==================================

def batch_get_all_bin_clear(source_path, bin_path):
    """获取所有去噪声的二值图片

    :param source_path: 原始图片路径
    :type source_path str

    :param bin_path: 降噪处理后保存的图片路径
    :type bin_path str
    """

    file_list = os.listdir(source_path)
    for file_name in file_list:
        file_full_path = os.path.join(source_path, file_name)
        image = Image.open(file_full_path)

        # 二值处理
        img = get_clear_bin_image(image)

        # 保存二值照片
        bin_clear_path = os.path.join(bin_path, file_name)
        img.save(bin_clear_path)


def batch_cut_images(bin_path, cut_path):
    """训练素材准备

    批量操作：分割切除所有 "二值 -> 除噪声" 之后的图片，变成所有的单字符的图片。然后保存到相应的目录，方便打标签

    :param bin_path: 降噪处理后图片路径
    :type bin_path str

    :param cut_path: 切割后图片保存路径
    :type cut_path str
    """

    file_list = os.listdir(bin_path)
    for file_name in file_list:
        bin_clear_img_path = os.path.join(bin_path, file_name)
        img = Image.open(bin_clear_img_path)

        child_img_list = get_crop_images(img)
        save_crop_images(bin_clear_img_path, child_img_list, cut_path)  # 将切割的图进行保存，后面打标签时要用


def generate_alias_name(file_path, suffix, separator="."):
    """生成别名，保留后缀格式不变

    :param file_path: 文件全路径
    :type file_path str

    :param suffix: 要增加的后缀
    :type suffix str

    :param separator: 分隔符，默认为 `.`
    :type separator str

    :rtype: new_path str
    :return: new_path 生成后的文件名

    :raise ValueError 分割符不在文件名中
    """
    parent_directory = os.path.dirname(file_path)
    # 获取文件名和类型
    file_name, file_type = os.path.basename(file_path).rsplit(separator, 1)
    # 生成新的文件路径
    new_path = os.path.join(parent_directory, "{}{}{}{}".format(file_name, suffix, separator, file_type))
    return new_path
