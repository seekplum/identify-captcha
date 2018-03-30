================
目录介绍
================

-----------------
目录结构
-----------------

.. code-block:: bash

    ├── build.py               # 编译脚本，构建env
    ├── LICENSE
    ├── packages               # 相关包目录
    │   ├── libsvm          # libsvm安装包
    │   └── msyh.ttf         # 验证码字体包
    ├── Pipfile                # pipenv配置文件
    ├── Pipfile.lock           # pipenv版本文件， 不要手动修改
    ├── README.md
    ├── requirements.txt       # 依赖库版本文件
    └── src
    │   ├── captcha_code.py   # 生成验证码文件
    │   ├── config.py         # 各种目录等配置文件
    │   ├── identify_code.py  # 训练结束后识别验证码文件
    │   ├── __init__.py
    │   ├── svm_features.py   # 获取图片特征值
    │   ├── train_svm.py     # svm训练数据
    │   └── utils.py          # 工具函数
    ├── static                 # 静态资源目录
    │   └── images          # 图片目录
    └── tests                  # 测试目录
        └── test.py
