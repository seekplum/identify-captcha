================
识别步骤解析
================

-------------
创建相关目录
-------------
train_data.py

.. code-block:: python

    def create_folder():
        """创建目录
        """
        pass


---------------
测试图片分割
---------------
train_data.py

.. code-block:: python

    def test_cut_pic():
    """测试图片分割
    """

* 图片分割示例

.. image:: /_static/images/cut-test.png

----------------------
测试识别分割后的图片
----------------------
train_data.py

.. code-block:: python

    def test_svm():
        """测试通过svm模型识别单张图片中的数字
        """

-----------------
模型训练
-----------------
* 准备分割好的验证码切图
* 获取测试集的像素特征文件
* 训练并生成model文件


