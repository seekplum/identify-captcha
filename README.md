# 验证码识别

---------------------

## 项目环境
* Ubuntu 16.04/CentOS 6.6
* Python2.7

## 安装运行环境
Pillow 依赖一些系统组件,需要先进行安装
* Ubuntu
> sudo apt-get install python-dev

* CentOS
> sudo yum install libtiff-devel libjpeg-devel libzip-devel freetype-devel lcms2-devel libwebp-devel tcl-devel tk-devel

## 准备步骤

### 创建目录
* 创建相关需要的目录

### 简单测试
* 测试图片分割是否成功

图片分割示例
![切割后图片](https://github.com/seekplum/identify-captcha/blob/master/static/images/cut-test.png)

* 测试识别分割后的图片

### 模型训练
* 准备分割好的验证码切图
* 获取测试集的像素特征文件
* 训练并生成model文件

## 识别验证码流程
* 获取要识别的验证码图片
* 降噪的获取二值图
* 加载SVM模型进行预测
* 使用特征算法，将图像进行特征化降维
* 将所有的特征转化为标准化的SVM单行的特征向量
* 将识别结果合并起来
* 以识别结果为文件名保存图片
* 检查识别是否成功


## 待解决
* 验证码进行动态的切割
* 不同字符形态的特征值收集

## 目录介绍
```text
├── build.py               # 编译脚本，构建env
├── LICENSE
├── packages               # 相关包目录
│   ├── libsvm             # libsvm安装包
│   └── msyh.ttf           # 验证码字体包
├── Pipfile                # pipenv配置文件
├── Pipfile.lock           # pipenv版本文件，`不要手动修改`
├── README.md
├── requirements.txt       # 依赖库版本文件
└── src
│   ├── captcha_code.py    # 生成验证码文件
│   ├── config.py          # 各种目录等配置文件
│   ├── identify_code.py   # 训练结束后识别验证码文件
│   ├── __init__.py
│   ├── svm_features.py    # 获取图片特征值
│   ├── train_data.py      # svm训练数据
│   └── utils.py           # 工具函数
├── static                 # 静态资源目录
│   └── images             # 图片目录
└── tests
    └── test.py
```

## 参考
* [字符型图片验证码识别完整过程及Python实现](https://www.cnblogs.com/beer/p/5672678.html)

## 后续交流

如果有对相关技术有持续关注的兴趣的同学，欢迎加入QQ群： 592109504

或者手机QQ扫码加入：

![QQ群图片](https://github.com/seekplum/identify-captcha/blob/master/static/images/qq-group.jpg)
