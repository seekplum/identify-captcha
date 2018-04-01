================
构建运行环境
================

-----------
项目环境
-----------
* Ubuntu 16.04/CentOS 6.6
* Python2.7

----------
项目依赖库
----------
* \ `pipenv==11.9.0 <https://docs.pipenv.org/>`_
* `pybuilder==0.11.12 <http://pybuilder.readthedocs.io/en/latest/>`_

------------
依赖系统插件
------------
Pillow 依赖一些系统组件,需要先进行安装

* Ubuntu

.. code-block:: bash

    sudo apt-get install python-dev

* CentOS

.. code-block:: bash

    sudo yum install libtiff-devel libjpeg-devel libzip-devel freetype-devel lcms2-devel libwebp-devel tcl-devel tk-devel

-----------
构建env环境
-----------
* 操作步骤

.. code-block:: bash

    # 克隆项目
    git clone git@github.com:seekplum/identify-captcha.git

    # 进入项目
    cd identify-captcha

    # 构建env环境
    pyb

    # 进入env环境
    source .venv/bin/activate

    # 测试环境是否有问题
    python tests/test_main.py

* 构建env环境操作截图

.. image:: /_static/images/build-env.png

* 通过测试截图

.. image:: /_static/images/test-main.png
