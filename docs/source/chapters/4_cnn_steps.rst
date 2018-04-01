=================
cnn模型识别步骤解析
=================

-------------
创建相关目录
-------------
train_cnn.py

.. code-block:: python

    def create_folder():
        """创建目录
        """
        pass


---------------
定义CNN模型
---------------
train_cnn.py

.. code-block:: python

    def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
        """定义CNN模型

        :param w_alpha: 张量系数
        :type w_alpha float

        :param b_alpha: 张量系数
        :type b_alpha float

        :rtype tensorflow.python.framework.ops.Tensor
        """

----------------------
训练数据直到准确率达到目录值
----------------------
train_cnn.py

.. code-block:: python

    while True:
        batch_x, batch_y = get_next_batch(BATCH_NUMBER)
        _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
        print_info("\n\nstep: {}, loss: {}".format(step, loss_))
        loss_result.append(loss_)
        loss_step_result.append(step)
        # 每固定间隔计算一次准确率
        if step % COUNT_INTERNAL == 0:
            batch_x_test, batch_y_test = get_next_batch(BATCH_NUMBER)
            acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
            print_info("\nstep: {}, acc: {}".format(step, acc))
            accuracy_step_result.append(step)
            accuracy_result.append(acc)
            # 如果准确率大于 目标正确率 ,保存模型,完成训练
            if acc > TARGET_ACCURACY:
                saver.save(sess, cnn_mode_path, global_step=step)
                break

-----------------
预测结果
-----------------
train_cnn.py

.. code-block:: python

    batch_x_test, batch_y_test = get_next_batch(BATCH_NUMBER)
    acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})

* 如果准确率大于 目标正确率 ,保存模型,完成训练


