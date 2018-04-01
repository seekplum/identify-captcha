#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

import pytest

# parametrize 中需要对每个测试用例指定id,对应不上会导致整个test文件无法运行
curr_path = os.path.dirname(os.path.abspath(__file__))


def _test_file(file_name):
    """通过pytest测试当前目录下指定的文件

    :param file_name: py文件名
    :type file_name str
    """
    file_path = os.path.join(curr_path, file_name)
    if os.path.exists(file_path):
        print("test file: {}.".format(file_path))
        # 测试单个文件
        pytest.main(file_path)
    else:
        print("file : {} not exists.".format(file_path))


def _test_folder():
    """通过pytest测试当前目录下所有文件
    """
    print("test folder: {}.".format(curr_path))
    # 测试整个目录下的文件
    pytest.main(curr_path)


if __name__ == '__main__':
    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser_fallback = parser.add_mutually_exclusive_group(required=False)
    parser_fallback.add_argument("-f", "--filename",
                                 required=False,
                                 action="store",
                                 dest="filename",
                                 default="",
                                 help="The file name to test.")

    # 获取命令行参数
    args = parser.parse_args()
    filename = args.filename

    if filename:
        _test_file(filename)
    else:
        _test_folder()
