# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#=============================================================================
#  ProjectName: identify-captcha
#     FileName: expections
#         Desc: 
#       Author: hjd
#     HomePage: seekplun.github.io
#   LastChange: 2018-04-01 18:50
#=============================================================================
"""


class TrainSvmError(Exception):
    """svm模型训练异常
    """
    pass


class TrainCnnError(Exception):
    """cnn模型训练异常
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
