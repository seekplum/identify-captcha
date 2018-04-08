#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#=============================================================================
#  ProjectName: identify-captcha
#     FileName: pretreatment_image
#         Desc: 预处理图片
#       Author: hjd
#     HomePage: seekplum.github.io
#   LastChange: 2018-04-03 13:01
#=============================================================================
"""
import os

from collections import Counter
from Queue import Queue

import cv2
import pytesseract

from PIL import Image, ImageDraw


class PixelError(Exception):
    """坐标错误
    """
    pass


def generate_alias_path(file_path, suffix, separator="."):
    """生成别名，保留后缀格式不变

    :param file_path: str 文件全路径
    :param suffix: str 要增加的后缀
    :param separator: str 分隔符，默认为 `.`

    :return: 生成后的文件名

    :raise ValueError 分割符不在文件名中
    """
    parent_directory = os.path.dirname(file_path)
    # 获取文件名和类型
    file_name = os.path.basename(file_path)
    # 生成新的文件路径
    return os.path.join(parent_directory,
                        generate_alias_name(file_name, suffix, separator))


def generate_alias_name(file_name, suffix, separator="."):
    """生成别名，保留后缀格式不变

    :param file_name: str 文件全路径
    :param suffix: str 要增加的后缀
    :param separator: str 分隔符，默认为 `.`

    :return: 生成后的文件名

    :raise ValueError 分割符不在文件名中
    """
    # 获取文件名和类型
    file_name, file_type = file_name.rsplit(separator, 1)
    # 生成新的文件名
    return "{}{}{}{}".format(file_name, suffix, separator, file_type)


def identify_code(image_path):
    """识别验证码

    识别用的是typesseract库，主要识别一行字符和单个字符时的参数设置，识别中英文的参数设置

    :param image_path: 验证码图片路径
    :type image_path str

    :rtype str
    :return 图片中的文字
    """
    # 识别字符, # 单个字符是10，一行文本是7
    img_text = pytesseract.image_to_string(Image.open(image_path), lang='eng',
                                           config='-psm 10')
    return img_text


class Pretreatment(object):
    """预处理验证码图片
    """

    def __init__(self, image_path, is_save=False, save_folder=None,
                 color_type=1, color_deviation=5):
        """初始化属性

        :param image_path: 验证码图片路径
        :type image_path str

        :param is_save: 是否需要保存图片
        :type is_save bool

        :param save_folder: 保存图片目录
        :type save_folder str

        :param color_type 字符的颜色种类
        :type color_type int

        :param color_deviation 字符颜色偏差
        :type color_deviation int
        """
        self.image_path = image_path
        self.image_name = os.path.basename(self.image_path)
        self.is_save = is_save
        self.save_folder = save_folder
        self.color_type = color_type
        self.color_deviation = color_deviation

        self.cut_paths = set()  # 切割后路径
        self.numpy_img = None  # numpy.array 图片对象
        self.image_img = None  # PIL.Image 图片对象

    def _save_image(self, image_name, image_obj=None, is_numpy_img=True):
        """保存图片

        :param image_name: 需要保存的图片名字
        :type image_name str

        :param is_numpy_img: 是否是numpy图片对象
        :type is_numpy_img bool

        :return image_path 保存后的图片路径
        :rtype image_path str
        """
        # 是否需要保存
        if self.is_save:
            # 图片对象类型
            if image_obj is None:
                image_obj = self.numpy_img

            # 获取保存路径
            if self.save_folder:
                save_folder = self.save_folder
            else:
                save_folder = os.path.dirname(self.image_path)
            image_path = os.path.join(save_folder, image_name)
            # 保存图片
            if is_numpy_img:
                cv2.imwrite(image_path, image_obj)
            else:
                self.image_img.save(image_path)
            return image_path

    def _get_dynamic_binary_image(self):
        """自适应阀值二值化

        灰度处理，就是把彩色的验证码图片转为灰色的图片。

    　　 二值化，是将图片处理为只有黑白两色的图片，利于后面的图像处理和识别
        """
        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰值化
        # 二值化
        self.numpy_img = cv2.adaptiveThreshold(img, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 21, 1)
        self._save_image(generate_alias_name(self.image_name, "_binary"))

    def _get_threshold(self):
        """查询背景色阀值
        """
        img = Image.open(self.image_path)
        # img = img.convert('L')
        pix_data = img.load()
        width, height = img.size
        image_data = [pix_data[x, y] for y in range(height) for x in
                      range(width)]
        image_counts = Counter(image_data)
        most_common = image_counts.most_common(self.color_type + 1)
        # 频率出现最高的认为是背景色
        # 第二到第五 高的为验证码颜色
        thresholds = [item[0] for item in most_common[1:]]
        return thresholds

    def _get_static_binary_image(self):
        """手动二值化

        1. 对于超过阀值的设置成白色,文字部分设置黑色
        2. 重新保存图片
        """

        def check_dot_turn_white():
            """检查该点是否需要转成白色
            """
            result = True
            for item in thresholds:
                if all([abs(item[i] - pix_data[x, y][i]) < self.color_deviation
                        for i in range(len(pix_data[x, y]))]):
                    result = False
                    break
            return result

        thresholds = self._get_threshold()
        img = Image.open(self.image_path)
        # img = img.convert('L')
        pix_data = img.load()
        width, height = img.size
        for y in range(height):
            for x in range(width):
                if check_dot_turn_white():
                    pix_data[x, y] = (255, 255, 255, 255)  # 白色
                    # pix_data[x, y] = 255  # 白色
                    # else:
                    # pix_data[x, y] = 0  # 黑色
        self.image_img = img
        self.image_path = self._save_image(generate_alias_name(self.image_name,
                                                               "_static"),
                                           is_numpy_img=False)
        # image = Image.new("1", img.size)
        # draw = ImageDraw.Draw(image)
        #
        # for x in xrange(0, width):
        #     for y in xrange(0, height):
        #         draw.point((x, y), pix_data[(x, y)])
        # self.image_img = image
        # self.image_path = self._save_image(
        #     generate_alias_name(self.image_name, "_static"),
        #     is_numpy_img=False)

    @classmethod
    def get_static_binary_image(cls, image_path, new_image_path, threshold=150,
                                dot_count=3):
        """降噪

        根据一个点A的RGB值，与周围的8个点的RBG值比较，设定一个值  dot_count (0 < dot_count <8)
        当检查点的RGB值与周围8个点的RGB相等数小于 dot_count 时，此点为噪点

        :param image_path 要识别的图片路径
        :type image_path str

        :param new_image_path 处理后的图片路径
        :type new_image_path str

        :param threshold RGB阀值,小于该值认为时字符内容
        :type threshold int

        :param dot_count 与周围点相等数量,小于该值认为是噪点
        :type dot_count int
        """
        pix_data = {}
        img = Image.open(image_path)
        image_img = img.convert('L')
        width, height = image_img.size
        for y in xrange(0, height):
            for x in xrange(0, width):
                if image_img.getpixel((x, y)) > threshold:
                    pix_data[(x, y)] = 1
                else:
                    pix_data[(x, y)] = 0

        pix_data[(0, 0)] = 1
        pix_data[(width - 1, height - 1)] = 1

        for x in xrange(1, width - 1):
            for y in xrange(1, height - 1):
                near_dot = 0
                curr_dot = pix_data[(x, y)]
                if curr_dot == pix_data[(x - 1, y - 1)]:
                    near_dot += 1
                if curr_dot == pix_data[(x - 1, y)]:
                    near_dot += 1
                if curr_dot == pix_data[(x - 1, y + 1)]:
                    near_dot += 1
                if curr_dot == pix_data[(x, y - 1)]:
                    near_dot += 1
                if curr_dot == pix_data[(x, y + 1)]:
                    near_dot += 1
                if curr_dot == pix_data[(x + 1, y - 1)]:
                    near_dot += 1
                if curr_dot == pix_data[(x + 1, y)]:
                    near_dot += 1
                if curr_dot == pix_data[(x + 1, y + 1)]:
                    near_dot += 1

                if near_dot < dot_count:  # 确定是噪声
                    pix_data[(x, y)] = 1

        image_img = Image.new("1", image_img.size)
        draw = ImageDraw.Draw(image_img)

        for x in xrange(0, width):
            for y in xrange(0, height):
                draw.point((x, y), pix_data[(x, y)])

        image_img.save(new_image_path)

    def _clear_border(self):
        """去除边框

        去除边框就是遍历像素点，找到四个边框上的所有点，把他们都改为白色

        注意：在用OpenCV时，图片的矩阵点是反的，就是长和宽是颠倒的
        """
        height, width = self.numpy_img.shape[:2]
        for y in range(0, width):
            for x in range(0, height):
                # if y ==0 or y == width -1 or y == width - 2:
                if y < 4 or y > width - 4:
                    self.numpy_img[x, y] = 255
                # if x == 0 or x == width - 1 or x == width - 2:
                if x < 4 or x > height - 4:
                    self.numpy_img[x, y] = 255
        self._save_image(generate_alias_name(self.image_name, "_border"))

    def _interference_line(self):
        """降噪去除干扰线

        线降噪的思路就是检测这个点相邻的四个点，判断这四个点中是白点的个数，
        如果有两个以上的白色像素点，那么就认为这个点是白色的，从而去除整个干扰线，
        但是这种方法是有限度的，如果干扰线特别粗就没有办法去除，只能去除细的干扰线
        """
        height, width = self.numpy_img.shape[:2]
        value = 245
        for y in range(1, width - 1):
            for x in range(1, height - 1):
                count = 0
                if self.numpy_img[x, y - 1] > value:
                    count += 1
                if self.numpy_img[x, y + 1] > value:
                    count += 1
                if self.numpy_img[x - 1, y] > value:
                    count += 1
                if self.numpy_img[x + 1, y] > value:
                    count += 1
                if count > 2:
                    self.numpy_img[x, y] = 255
        self._save_image(generate_alias_name(self.image_name, "_line"))

    def _interference_point(self, x=0, y=0):
        """降噪去除干扰点

        邻域框,以当前点为中心的田字框,黑点个数

        :param x: int 开始像素点的横坐标
        :param y: int 开始像素点的纵坐标
        """
        height, width = self.numpy_img.shape[:2]
        # 检查x,y 坐标值是否有效
        if x > height or x < 0 or y > width or y < 0:
            raise PixelError(
                "({}, {}) beyond ({}, {})".format(x, y, height, width))
        curr_pixel = self.numpy_img[x, y]  # 当前像素点的值
        for y in range(1, width - 1):
            for x in range(1, height - 1):
                if y == 0:  # 第一行
                    if x == 0:  # 左上顶点,4邻域
                        # 中心点旁边3个点
                        number = int(curr_pixel) \
                                 + int(self.numpy_img[x, y + 1]) \
                                 + int(self.numpy_img[x + 1, y]) \
                                 + int(self.numpy_img[x + 1, y + 1])
                        if number <= 2 * 245:
                            self.numpy_img[x, y] = 0
                    elif x == height - 1:  # 右上顶点
                        number = int(curr_pixel) \
                                 + int(self.numpy_img[x, y + 1]) \
                                 + int(self.numpy_img[x - 1, y]) \
                                 + int(self.numpy_img[x - 1, y + 1])
                        if number <= 2 * 245:
                            self.numpy_img[x, y] = 0
                    else:  # 最上非顶点,6邻域
                        number = int(self.numpy_img[x - 1, y]) \
                                 + int(self.numpy_img[x - 1, y + 1]) \
                                 + int(curr_pixel) \
                                 + int(self.numpy_img[x, y + 1]) \
                                 + int(self.numpy_img[x + 1, y]) \
                                 + int(self.numpy_img[x + 1, y + 1])
                        if number <= 3 * 245:
                            self.numpy_img[x, y] = 0
                elif y == width - 1:  # 最下面一行
                    if x == 0:  # 左下顶点
                        # 中心点旁边3个点
                        number = int(curr_pixel) \
                                 + int(self.numpy_img[x + 1, y]) \
                                 + int(self.numpy_img[x + 1, y - 1]) \
                                 + int(self.numpy_img[x, y - 1])
                        if number <= 2 * 245:
                            self.numpy_img[x, y] = 0
                    elif x == height - 1:  # 右下顶点
                        number = int(curr_pixel) \
                                 + int(self.numpy_img[x, y - 1]) \
                                 + int(self.numpy_img[x - 1, y]) \
                                 + int(self.numpy_img[x - 1, y - 1])

                        if number <= 2 * 245:
                            self.numpy_img[x, y] = 0
                    else:  # 最下非顶点,6邻域
                        number = int(curr_pixel) \
                                 + int(self.numpy_img[x - 1, y]) \
                                 + int(self.numpy_img[x + 1, y]) \
                                 + int(self.numpy_img[x, y - 1]) \
                                 + int(self.numpy_img[x - 1, y - 1]) \
                                 + int(self.numpy_img[x + 1, y - 1])
                        if number <= 3 * 245:
                            self.numpy_img[x, y] = 0
                else:  # y不在边界
                    if x == 0:  # 左边非顶点
                        number = int(self.numpy_img[x, y - 1]) \
                                 + int(curr_pixel) \
                                 + int(self.numpy_img[x, y + 1]) \
                                 + int(self.numpy_img[x + 1, y - 1]) \
                                 + int(self.numpy_img[x + 1, y]) \
                                 + int(self.numpy_img[x + 1, y + 1])

                        if number <= 3 * 245:
                            self.numpy_img[x, y] = 0
                    elif x == height - 1:  # 右边非顶点
                        number = int(self.numpy_img[x, y - 1]) \
                                 + int(curr_pixel) \
                                 + int(self.numpy_img[x, y + 1]) \
                                 + int(self.numpy_img[x - 1, y - 1]) \
                                 + int(self.numpy_img[x - 1, y]) \
                                 + int(self.numpy_img[x - 1, y + 1])

                        if number <= 3 * 245:
                            self.numpy_img[x, y] = 0
                    else:  # 具备9领域条件的
                        number = int(self.numpy_img[x - 1, y - 1]) \
                                 + int(self.numpy_img[x - 1, y]) \
                                 + int(self.numpy_img[x - 1, y + 1]) \
                                 + int(self.numpy_img[x, y - 1]) \
                                 + int(curr_pixel) \
                                 + int(self.numpy_img[x, y + 1]) \
                                 + int(self.numpy_img[x + 1, y - 1]) \
                                 + int(self.numpy_img[x + 1, y]) \
                                 + int(self.numpy_img[x + 1, y + 1])
                        if number <= 4 * 245:
                            self.numpy_img[x, y] = 0
        self._save_image(generate_alias_name(self.image_name, "_point"))

    def _detect_block_point(self, x_max):
        """搜索区块起点

        :param x_max: int x 横坐标

        :rtype numpy.ndarray
        :returns 图片二维坐标
        """
        height, width = self.numpy_img.shape[:2]
        for y_fd in range(x_max + 1, width):
            for x_fd in range(height):
                if self.numpy_img[x_fd, y_fd] == 0:
                    return x_fd, y_fd

    def _completely_fair_scheduler(self, x_fd, y_fd):
        """用队列和集合记录遍历过的像素坐标代替单纯递归以解决cfs访问过深问题

        :param x_fd: int x 横坐标
        :param y_fd: int y 纵坐标

        :rtype int, int, int, int
        :returns y坐标最大值，y坐标最小值，x坐标最大值，x坐标最小值
        """
        x_axis = []
        y_axis = []
        visited = set()
        q = Queue()
        q.put((x_fd, y_fd))
        visited.add((x_fd, y_fd))
        offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # 四邻域

        while not q.empty():
            x, y = q.get()
            for x_offset, y_offset in offsets:
                x_neighbor, y_neighbor = x + x_offset, y + y_offset
                if (x_neighbor, y_neighbor) in visited:
                    continue  # 已经访问过了
                visited.add((x_neighbor, y_neighbor))
                try:
                    if self.numpy_img[x_neighbor, y_neighbor] == 0:
                        x_axis.append(x_neighbor)
                        y_axis.append(y_neighbor)
                        q.put((x_neighbor, y_neighbor))

                except IndexError:
                    pass
        if len(x_axis) == 0 or len(y_axis) == 0:
            x_max = x_fd + 1
            x_min = x_fd
            y_max = y_fd + 1
            y_min = y_fd
        else:
            x_max = max(x_axis)
            x_min = min(x_axis)
            y_max = max(y_axis)
            y_min = min(y_axis)
        return y_max, y_min, x_max, x_min

    def _cut_position_block(self):
        """切割图片中的字符

        字符切割通常用于验证码中有粘连的字符，粘连的字符不好识别，
        所以我们需要将粘连的字符切割为单个的字符，在进行识别

    　　字符切割的思路就是找到一个黑色的点，然后在遍历与他相邻的黑色的点，
        直到遍历完所有的连接起来的黑色的点，
        找出这些点中的最高的点、最低的点、最右边的点、最左边的点，
        记录下这四个点，认为这是一个字符，
        然后在向后遍历点，直至找到黑色的点，继续以上的步骤。
        最后通过每个字符的四个点进行切割


        :rtype list, list, list
        :returns
        各区块长度列表
        各区块的X轴[起始，终点]列表
        各区块的Y轴[起始，终点]列表
        (
            [14, 13, 14, 14], # 各个字符长度
            [[4, 18], [28, 41], [49, 63], [72, 86]], # 字符 X 轴[起始，终点]位置
            [[6, 32], [14, 28], [8, 29], [8, 28]]  # 字符 Y 轴[起始，终点]位置
        )
        """
        zone_length = []  # 各区块长度列表
        zone_width = []  # 各区块的X轴[起始，终点]列表
        zone_height = []  # 各区块的Y轴[起始，终点]列表

        x_max = 0  # 上一区块结束黑点横坐标,这里是初始化
        for i in range(10):

            try:
                x_fd, y_fd = self._detect_block_point(x_max)
                x_max, x_min, y_max, y_min = self._completely_fair_scheduler(
                    x_fd, y_fd)
                length = x_max - x_min
                zone_length.append(length)
                zone_width.append([x_min, x_max])
                zone_height.append([y_min, y_max])
            except TypeError:
                return zone_length, zone_width, zone_height

        return zone_length, zone_width, zone_height

    def _get_stick_together_position(self):
        """查询粘在一起的字符的切割位置

        如果有粘连字符，如果一个字符的长度过长就认为是粘连字符，并从中间进行切割


        :rtype tuple
        :return
        (
            [14, 13, 14, 14], # 各个字符长度
            [[4, 18], [28, 41], [49, 63], [72, 86]], # 字符 X 轴[起始，终点]位置
            [[6, 32], [14, 28], [8, 29], [8, 28]]  # 字符 Y 轴[起始，终点]位置
        )
        """
        # 切割的位置
        img_position = self._cut_position_block()

        max_length = max(img_position[0])
        min_length = min(img_position[0])

        # 如果有粘连字符，如果一个字符的长度过长就认为是粘连字符，并从中间进行切割
        if max_length > (min_length + min_length * 0.7):
            max_length_index = img_position[0].index(max_length)
            # 设置字符的宽度
            img_position[0][max_length_index] = max_length // 2
            img_position[0].insert(max_length_index + 1, max_length // 2)
            # 设置字符X轴[起始，终点]位置
            img_position[1][max_length_index][1] = \
                img_position[1][max_length_index][0] + max_length // 2
            img_position[1].insert(max_length_index + 1,
                                   [img_position[1][max_length_index][1] + 1,
                                    img_position[1][max_length_index][
                                        1] + 1 + max_length // 2])
            # 设置字符的Y轴[起始，终点]位置
            img_position[2].insert(max_length_index + 1,
                                   img_position[2][max_length_index])
        return img_position

    def _cutting_img(self, x_offset=1, y_offset=1):
        """切割图片

        切割字符，要想切得好就得配置参数，通常 1 or 2 就可以

        比如 (
            [14, 13, 14, 14], # 各个字符长度
            [[4, 18], [28, 41], [49, 63], [72, 86]], # 字符 X 轴[起始，终点]位置
            [[6, 32], [14, 28], [8, 29], [8, 28]]  # 字符 Y 轴[起始，终点]位置
            )
        :param x_offset: int X 轴偏移位置
        :param y_offset: int Y 轴偏移位置

        :rtype set()
        :return: 切割之后的路径集合
        """
        img_position = self._get_stick_together_position()
        # 识别出的字符个数
        im_number = len(img_position[1])
        # 切割字符
        for i in range(im_number):
            img_start_x = img_position[1][i][0] - x_offset
            img_end_x = img_position[1][i][1] + x_offset
            img_start_y = img_position[2][i][0] - y_offset
            img_end_y = img_position[2][i][1] + y_offset
            cropped = self.numpy_img[
                      img_start_y:img_end_y, img_start_x:img_end_x]

            new_image_name = generate_alias_name(self.image_path,
                                                 suffix="cut_{}".format(i))
            self._save_image(new_image_name, image_obj=cropped)

            self.cut_paths.add(new_image_name)

    def prepare(self):
        """预处理

        1. 自适应阈值二值化
        2. 去除边框
        3. 对图片进行干扰线降噪
        4. 对图片进行点降噪
        5. 对字符进行切割
        """
        self._get_static_binary_image()
        self._get_dynamic_binary_image()
        # self._clear_border()
        self._interference_line()
        self._interference_point()
        self._cutting_img()
        return self.cut_paths


def test():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    static_folder = os.path.join(os.path.dirname(curr_path), "static")
    image_folder = os.path.join(static_folder, "images")
    test_image_path = os.path.join(image_folder, "test.png")
    new_test_image_path = os.path.join(image_folder, "test2.png")
    Pretreatment.get_static_binary_image(test_image_path, new_test_image_path)
    manager = Pretreatment(test_image_path, is_save=True)
    paths = manager.prepare()
    for image_path in paths:
        text = identify_code(image_path)
        print "image path: {}, text: {}".format(image_path, text)


if __name__ == '__main__':
    test()
