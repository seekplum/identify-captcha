#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pytest


def main():
    # parametrize 中需要对每个测试用例指定id,对应不上会导致整个test文件无法运行
    curr_path = os.path.dirname(os.path.abspath(__file__))
    print("test folder: {}.".format(curr_path))

    file_path = os.path.join(curr_path, "test_train_cnn.py")
    if os.path.exists(file_path):
        print("test file: {}.".format(file_path))
    else:
        print("file : {} not exists.".format(file_path))

    # 测试单个文件
    pytest.main(file_path)
    # 测试整个目录下的文件
    # pytest.main(curr_path)


if __name__ == '__main__':
    main()
