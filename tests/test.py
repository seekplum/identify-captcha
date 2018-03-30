# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#=============================================================================
#  ProjectName: identify-captcha
#     FileName: test
#         Desc: 测试脚本
#       Author: hjd
#     HomePage: seekplun.github.io
#   LastChange: 2018-03-26 23:36
#=============================================================================
"""

import pytest
import numpy

from train_cnn import text2vec, vec2text


@pytest.mark.parametrize("text", [
    "F5Sd",
    "SFd5",
    "Md19"
])
def test_text_to_vec(text):
    """测试文本转vec
    """
    vec = text2vec(text)

    assert isinstance(vec, numpy.ndarray)

    assert vec2text(vec) == text
