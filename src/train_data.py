#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#=============================================================================
#  ProjectName: seekplum
#     FileName: train_data
#         Desc: 训练数据
#       Author: hjd
#     HomePage: seekplun.github.io
#   LastChange: 2018-03-24 02:28
#=============================================================================
"""
import os
import shutil
import time
import uuid

from multiprocessing import Process

from PIL import Image
from svmutil import svm_read_problem, svm_load_model, svm_predict

from captcha_code import gen_captcha_text_image
from config import captcha_length
from config import test_feature_file, model_path
from config import data_root, svm_root
from config import cut_pic_folder, cut_test_folder, bin_clear_folder, identify_result_folder, origin_pic_folder
from svm_features import get_svm_train_txt, convert_images_to_feature_file, train_svm_model
from utils import get_clear_bin_image, get_crop_images, generate_alias_name
from utils import batch_get_all_bin_clear, batch_cut_images


def test_number_svm_txt(number):
    """获取 测试集 的像素特征文件

    注意： 测试时只用 数字 `number` 进行测试

    :param number: 要测试的数字
    :type number int
    """
    # 把数字目录中的所有文件复制到 测试目录下
    image_path = os.path.join(cut_pic_folder, str(number))

    feature_file = test_feature_file.format(number)
    # 生成模型文件
    with open(feature_file, 'w') as f:
        convert_images_to_feature_file(number, f, image_path)


def test_number_svm_model(number):
    """使用测试集测试模型

    :param number: 要测试的数字
    :type number int
    """
    feature_file = test_feature_file.format(number)

    yt, xt = svm_read_problem(feature_file)
    print "yt: {}".format(yt)

    model = svm_load_model(model_path)
    p_label, p_acc, p_val = svm_predict(yt, xt, model)

    for cnt, item in enumerate(p_label):
        print "result: {}".format(item),
        # 每十个打印一行
        if cnt % 10 == 0:
            print


def test_cut_pic():
    """测试图片分割
    """
    text, image_path = gen_captcha_text_image(cut_test_folder, file_name="test", draw_lines=True, draw_points=True)
    img = Image.open(image_path)

    # 获取干净的二值化的图片
    img = get_clear_bin_image(img)
    new_image_path = generate_alias_name(image_path, "_bin")
    img.save(new_image_path)

    child_img_list = get_crop_images(img)

    for index, child_img in enumerate(child_img_list):
        file_name = "cur-{}.png".format(index)
        file_path = os.path.join(cut_test_folder, file_name)
        child_img.save(file_path)


def test_svm():
    """测试通过svm模型识别单张图片中的数字
    """
    for number in range(10):
        print "=" * 100
        # 只针对 number 进行测试
        test_number_svm_txt(number)
        test_number_svm_model(number)
        print "\n\n"


def create_folder():
    """创建目录
    """
    paths = [
        data_root,
        origin_pic_folder,
        cut_pic_folder,
        cut_test_folder,
        identify_result_folder,
        bin_clear_folder,
        svm_root
    ]
    for file_path in paths:
        if not os.path.exists(file_path):
            os.mkdir(file_path)

    # 创建数据目录 0-9
    for i in range(10):
        file_path = os.path.join(cut_pic_folder, str(i))
        if not os.path.exists(file_path):
            os.mkdir(file_path)


def _mock_image(name):
    """生成假数据

    :param name: 具体数字 比如 1
    :type name str
    """
    # 根据 name 把图片分开，不需要锁
    # 把图片区分根据数字区分开
    temp_folder = os.path.join(origin_pic_folder, name)
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    # 只生成同一数字的验证码，方便造数据
    gen_captcha_text_image(temp_folder, file_name=uuid.uuid4().hex, chars=name * captcha_length)

    # 预处理
    batch_get_all_bin_clear(source_path=temp_folder, bin_path=temp_folder)

    # 切割
    cut_path = os.path.join(cut_pic_folder, name)
    batch_cut_images(temp_folder, cut_path)

    # 删除验证码源文件
    shutil.rmtree(temp_folder, ignore_errors=False)


def mock_image(name, number):
    """生成假数据

    :param name: 具体数字 比如 1
    :type name str

    :param number: 生成图片的数量
    :type number int
    """
    for j in range(number):
        _mock_image(name)


def generate_image(number=1):
    """生成图片

    多线程下 30 张需要 18 秒左右
    多进程下 30 张需要 5 秒左右
    单线程下 30 张需要 15 秒左右

    :param number: 生成图片的数量
    :type number int
    """
    count = 10

    # 多线程 线程池生成验证码
    start_time = time.time()
    # 使用线程池
    # from concurrent.futures import ThreadPoolExecutor
    # with ThreadPoolExecutor(count) as executor:
    #     for i in range(count):
    #         executor.submit(mock_image, str(i), number)

    # 使用多进程
    processes = []
    for i in range(count):
        process = Process(target=mock_image, args=(str(i), number,))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    # 检查进程是否结束
    for process in processes:
        if process.is_alive():
            process.terminate()

    end_time = time.time()
    print "use time: {}".format(end_time - start_time)


def train():
    """训练模型

    准备原始图片素材
    图片预处理
    图片字符切割
    图片尺寸归一化
    图片字符标记
    字符图片特征提取
    生成特征和标记对应的训练数据集
    训练特征标记数据生成识别模型
    使用识别模型预测新的未知图片集
    达到根据 图片 返回识别正确的字符集的目标
    """
    generate_image()

    # 获取 0-9 特征像素
    get_svm_train_txt()
    # 生成 0-9 模型文件
    train_svm_model()


if __name__ == '__main__':
    create_folder()
    train()
    test_cut_pic()
    test_svm()
