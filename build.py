#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
#=============================================================================
#  ProjectName: identify-captcha
#     FileName: build
#         Desc: 编译项目依赖env环境
#       Author: hjd
#     HomePage: seekplun.github.io
#   LastChange: 2018-03-24 02:28
#=============================================================================
"""
import os
import shutil
import subprocess

from pybuilder.core import use_plugin, init, task, depends, Author

version = "1.0.0"
authors = [Author("hjd", "1131909224m@sin.cn")]
description = "Machine learning identification verification code."
name = "identify-captcha"

# 任务列表
default_task = ["init_environment", "install_lib_svm"]

use_plugin("python.install_dependencies")


@init
def initialize(project):
    project.set_property("install_dependencies_index_url", "https://mirrors.ustc.edu.cn/pypi/web/simple")
    project.build_depends_on("pipenv", version="==11.9.0")


@task
@depends("install_build_dependencies")
def init_environment(logger):
    logger.info("Init environment")
    logger.debug("Running cmd: pipenv install --skip-lock")
    subprocess.check_call(["PIPENV_VENV_IN_PROJECT=1 pipenv install --skip-lock"], shell=True)


@task
@depends("init_environment")
def install_lib_svm(logger):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    logger.info("Install: libsvm")
    lib_path = os.path.join(curr_path, ".venv", "lib")
    py_path = os.path.join(curr_path, ".venv", "lib", "python2.7")
    packages_path = os.path.join(curr_path, "packages", "libsvm")

    logger.info("Copy file...")
    shutil.copy(os.path.join(packages_path, "svm.py"), py_path)
    shutil.copy(os.path.join(packages_path, "svmutil.py"), py_path)
    shutil.copy(os.path.join(packages_path, "libsvm.so.2"), lib_path)
