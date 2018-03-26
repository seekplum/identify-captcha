#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#=============================================================================
#  ProjectName: identify-captcha
#     FileName: captcha_code
#         Desc: 生成验证码
#       Author: hjd
#     HomePage: seekplun.github.io
#   LastChange: 2018-03-24 02:28
#=============================================================================
"""
import os
import random

from PIL import Image, ImageDraw, ImageFont, ImageFilter

_letter_cases = "abcdefghjkmnpqrstuvwxy"  # 小写字母，去除可能干扰的 i，l，o，z
_upper_cases = _letter_cases.upper()  # 大写字母
_numbers = "23456789"  # 数字 去除 0,1
init_chars = ''.join((_letter_cases, _upper_cases, _numbers))


def gen_captcha_text_image(image_path,
                           file_name=None,
                           suffix="png",
                           size=(160, 40),
                           chars=_numbers,
                           mode="RGB",
                           bg_color=(255, 255, 255),
                           fg_color=(0, 0, 255),
                           font_size=47,
                           font_type="msyh.ttf",
                           length=4,
                           draw_lines=False,
                           n_line=(1, 2),
                           draw_points=False,
                           point_chance=2,
                           draw_transform=False):
    """生成验证码图片

    :param image_path: 生成的验证码路径
    :type image_path str

    :param file_name: 生成的验证码名字
    :type file_name str

    :param suffix: 验证码后缀
    :type suffix str

    :param size: 图片的大小，格式（宽，高），默认为(120, 30)
    :type size tuple

    :param chars: 允许的字符集合，格式字符串
    :type chars str

    :param mode: 图片模式，默认为RGB
    :type mode str

    :param bg_color: 背景颜色，默认为白色
    :type bg_color tuple(颜色rgb值)

    :param fg_color: 验证码字符颜色，默认为蓝色#0000FF
    :type bg_color tuple(颜色rgb值)

    :param font_size: 验证码字体大小
    :type font_size int

    :param font_type: 验证码字体，默认为 ae_AlArabiya.ttf(这个字体必须得是系统/usr/share/fonts/存在的 )
    :type font_type str

    :param length: 验证码字符个数
    :type length int

    :param draw_lines: 是否划干扰线
    :type draw_lines bool

    :param n_line: 干扰线的条数范围，格式元组，默认为(1, 2)，只有draw_lines为True时有效
    :type n_line tuple

    :param draw_points: 是否画干扰点
    :type draw_points bool

    :param point_chance: 干扰点出现的概率，大小范围[0, 100]
    :type point_chance list

    :param draw_transform: 是否对字体进行变形
    :type draw_transform bool


    :return: text: 验证码字符串内容
    :rtype text str

    :return: file_path: 生成的验证码图片路径
    :rtype file_path str
    """

    width, height = size  # 宽， 高
    img = Image.new(mode, size, bg_color)  # 创建图形
    draw = ImageDraw.Draw(img)  # 创建画笔

    def get_chars():
        """生成给定长度的字符串，返回列表格式
        """
        return random.sample(chars, length)

    def create_lines():
        """绘制干扰线
        """
        line_num = random.randint(*n_line)  # 干扰线条数

        for i in range(line_num):
            # 起始点
            begin = (random.randint(0, size[0]), random.randint(0, size[1]))
            # 结束点
            end = (random.randint(0, size[0]), random.randint(0, size[1]))
            draw.line([begin, end], fill=(0, 0, 0))

    def create_points():
        """绘制干扰点
        """
        chance = min(100, max(0, int(point_chance)))  # 大小限制在[0, 100]

        for w in range(width):
            for h in range(height):
                tmp = random.randint(0, 100)
                if tmp > 100 - chance:
                    draw.point((w, h), fill=(0, 0, 0))

    def create_text():
        """绘制验证码字符
        """
        c_chars = get_chars()
        text_ = '%s' % ' '.join(c_chars)  # 每个字符前后以空格隔开

        font = ImageFont.truetype(font_type, font_size)
        # font_width, font_height = font.getsize(text_)
        # xy = ((width - font_width) / 3, (height - font_height) / 3) # 左右距离/上下距离
        xy = (2, -11)  # 左右距离/上下距离
        draw.text(xy, text_, font=font, fill=fg_color)

        return ''.join(c_chars)

    def create_transform(img_):
        """对验证码中的文件进行偏移
        """
        # 图形扭曲参数
        params = [1 - float(random.randint(1, 2)) / 100,
                  0,
                  0,
                  0,
                  1 - float(random.randint(1, 10)) / 100,
                  float(random.randint(1, 2)) / 500,
                  0.001,
                  float(random.randint(1, 2)) / 500
                  ]
        return img_.transform(size, Image.PERSPECTIVE, params)  # 创建扭曲

    # 增加干扰线
    if draw_lines:
        create_lines()

    # 增加干扰点
    if draw_points:
        create_points()

    # 绘制验证码字符
    text = create_text()

    # 增加字体变形
    if draw_transform:
        img = create_transform(img)

    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 滤镜，边界加强（阈值更大）

    # 图片路径
    file_name_ = file_name if file_name else text
    file_name = "{}.{}".format(file_name_, suffix)
    file_path = os.path.join(image_path, file_name)
    img.save(file_path)
    return text, file_path
