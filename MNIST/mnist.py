"""
Kaggle 练习 -- MNIST -- skflow版本与tensorflow不兼容造成不能运行
author: 王建坤
date: 2018-7-30
"""
import pandas as pd
import tensorflow as tf
import skflow
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = train.drop('label', 1)
y_train = train['label']
X_test = test

print('train_shape: ', X_train.shape, 'test_shape: ', X_test.shape)

# 线性分类器
# classifier = skflow.TensorFlowLinearClassifier(n_classes=10, batch_size=100, steps=1000, learning_rate=0.01)
# classifier.fit(X_train, y_train)
# linear_y_predict = classifier.predict(X_test)
# linear_submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': linear_y_predict})
# linear_submission.to_csv('linear_submission.csv')


# DNN分类器
# classifier_dnn = skflow.TensorFlowDNNClassifier(hidden_units=[200, 50, 10], n_classes=10, batch_size=50,
#                                                 steps=5000, learning_rate=0.01)
# classifier_dnn.fit(X_train, y_train)
# dnn_y_predict = classifier_dnn.predict(X_test)
# dnn_submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': dnn_y_predict})
# dnn_submission.to_csv('dnn_submission.csv')


# CNN分类器
def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_model(X, y):
    X = tf.reshape(X, [-1, 28, 28, 1])
    with tf.variable_scope('conv1'):
        conv1 = skflow.ops.conv2d(X, n_filters=32, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        pool1 = max_pool_2x2(conv1)

    with tf.variable_scope('conv2'):
        conv2 = skflow.ops.conv2d(pool1, n_filters=64, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        pool2 = max_pool_2x2(conv2)

    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    fc1 = skflow.ops.dnn(pool2_flat, [1024], activation=tf.nn.relu, keep_prob=0.5)
    return skflow.models.logistic_regression(fc1, y)


classifier_cnn = skflow.TensorFlowEstimator(model_fn=conv_model, n_classes=10,
                                            batch_size=100, steps=20000, learning_rate=0.001)
classifier_cnn.fit(X_train, y_train)
cnn_y_predict = []
for i in range(100, 28001, 100):
    cnn_y_predict = np.append(cnn_y_predict, classifier_cnn.predict(X_test[i-100:i]))
cnn_submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': np.int32(cnn_y_predict)})
cnn_submission.to_csv('cnn_submission.csv')






