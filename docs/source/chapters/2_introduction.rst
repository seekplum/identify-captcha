================
目录介绍
================

-----------------
目录结构
-----------------

.. code-block:: bash

    ├── build.py               # 编译脚本，构建env环境
    ├── captcha                # 验证码相关图片目录
    ├── docs                   # 文档目录
    ├── LICENSE
    ├── packages               # 相关依赖包目录
    ├── Pipfile                # pipenv配置文件
    ├── Pipfile.lock           # pipenv版本文件,不要手动修改
    ├── README.md
    ├── src                    # 代码源码
    │   ├── config.py          # 各种目录等配置文件
    │   ├── exception.py       # 代码异常类
    │   ├── identify_cnn.py    # cnn训练结束后识别验证码文件
    │   ├── identify_svm.py    # svm训练结束后识别验证码文件
    │   ├── __init__.py
    │   ├── svm_features.py    # 获取验证码图片特征值
    │   ├── train_cnn.py       # cnn训练数据
    │   ├── train_svm.py       # svm训练数据
    │   └── utils.py           # 工具函数
    ├── static                 # 静态资源目录
    │   └── images             # 图片目录
    └── tests                  # 测试目录
        ├── pytest.ini         # pytest配置文件
        ├── test_main.py       # pytest入口文件