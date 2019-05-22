#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:08:58 2019

@author: diegogarcia-viana
"""

import sys
import numpy as np
import tensorflow as tf

session = tf.Session()

mnist = tf.keras.datasets.mnist
(train_xdata, train_labels), (test_xdata, test_labels) = mnist.load_data()

train_xdata = train_xdata/255.0
test_xdata = test_xdata/255.0

batch_size = 100
learning_rate = 0.005
evaluation_size = 100
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
target_size = max(train_labels) + 1
num_channels = 1
generations = 100
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 = 2
fully_connected_size1 = 100
dropout_prob = 0.75

x_input_shape = (batch_size, image_width, image_height, num_channels)
x_input = tf.placeholder(shape=x_input_shape, dtype=tf.float32)
y_target = tf.placeholder(shape=(batch_size), dtype=tf.int32)

eval_input_shape = (evaluation_size, image_width, image_height, num_channels)
eval_input = tf.placeholder(shape=eval_input_shape, dtype=tf.float32)
eval_target = tf.placeholder(shape=(evaluation_size), dtype=tf.float32)

dropout = tf.placeholder(dtype=tf.float32, shape=())

conv1_weight = tf.Variable(tf.truncated_normal(shape=[4,4,num_channels,conv1_features], stddev=0.1, 
                                               dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros(shape=[conv1_features], dtype=tf.float32))

conv2_weight = tf.Variable(tf.truncated_normal(shape=[4,4,conv1_features,conv2_features], stddev=0.1, 
                                               dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros(shape=[conv2_features], dtype=tf.float32))

result_width = image_width//(max_pool_size1+max_pool_size2)
result_height = image_height//(max_pool_size1+max_pool_size2)
full1_input_size = result_height*result_width*conv2_features

full1_weight = tf.Variable(tf.truncated_normal(shape=[full1_input_size, fully_connected_size1], stddev=0.1, 
                                               dtype=tf.float32))
full1_bias = tf.Variable(tf.zeros(shape=[fully_connected_size1], dtype=tf.float32))

output_weight = tf.Variable(tf.truncated_normal(shape=[fully_connected_size1, target_size], stddev=0.1, 
                                               dtype=tf.float32))
output_bias = tf.Variable(tf.zeros(shape=[target_size]))

def cnn(input_data):
    conv1 = tf.nn.conv2d(input=input_data, filter=conv1_weight, strides=[1,1,1,1], padding="SAME")
    relu1 = tf.nn.relu(tf.nn.bias_add(value=conv1, bias=conv1_bias))
    maxpool1 = tf.nn.max_pool(value=relu1, ksize=[1,max_pool_size1,max_pool_size1,1], 
                              strides=[1,max_pool_size1,max_pool_size1,1], padding="SAME")
    
    conv2 = tf.nn.conv2d(input=maxpool1, filter=conv2_weight, strides=[1,1,1,1], padding="SAME")
    relu2 = tf.nn.relu(tf.nn.bias_add(value=conv2, bias=conv2_bias))
    maxpool2 = tf.nn.max_pool(value=relu2, ksize=[1,max_pool_size2,max_pool_size2,1], 
                              strides=[1,max_pool_size2,max_pool_size2,1], padding="SAME")
    
    final_conv_shape = maxpool2.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    flat_output = tf.reshape(maxpool2, [final_conv_shape[0], final_shape])
    
    fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))
    fully_connected2 = tf.add(tf.matmul(fully_connected1, output_weight), output_bias)
    
    return tf.nn.dropout(fully_connected2, keep_prob=dropout_prob)

model_output = cnn(x_input)
test_output = cnn(eval_input)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))

prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_output)

def get_accuracy(logits, labels):
    predictions = np.argmax(logits, axis=1)
    corrects = np.sum(np.equal(predictions, labels))
    return 100*corrects/predictions.shape[0]

train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)


init = tf.global_variables_initializer()
session.run(init)

class DropOutTest(tf.test.TestCase):
    def dropout_greater_than(self):
        with self.test_session():
            self.assertGreater(dropout.eval(), 25)

class AccuracyTest(tf.test.TestCase):
    def accuracy_exact_test(self):
        with self.test_session():
            test_preds = [[0.9, 0.1], [0.01, 0.99]]
            test_targets = [0,1]
            test_acc = get_accuracy(test_preds, test_targets)
            self.assertEqual(test_acc.eval(), 100.0)

class ShapeTest(tf.test.TestCase):
    def output_shape_test(self):
        with self.test_session():
            numpy_array = np.zeros([batch_size, target_size])
            self.assertShapeEqual(numpy_array, model_output)

def main(argv):
    train_loss = []
    train_acc = []
    test_acc = []
    
    for i in range(generations):
        rand_idx = np.random.choice(len(train_xdata), size=batch_size)
        rand_x = np.expand_dims(train_xdata[rand_idx], 3)
        rand_y = train_labels[rand_idx]
        train_dict = {x_input: rand_x, y_target: rand_y, dropout: dropout_prob}
        
        session.run(train, feed_dict=train_dict)
        
        temp_train_loss, temp_train_preds = session.run([loss, prediction], feed_dict=train_dict)
        temp_train_acc = get_accuracy(temp_train_preds, rand_y)
        
        if (i+1)%eval_every == 0:
            eval_idx = np.random.choice(len(test_xdata), size=evaluation_size)
            eval_x = np.expand_dims(test_xdata[eval_idx], 3)
            eval_y = test_labels[eval_idx]
            test_dict = {eval_input: eval_x, eval_target: eval_y, dropout: 1.0}
            
            test_preds = session.run(test_prediction, feed_dict=test_dict)
            temp_test_acc = get_accuracy(test_preds, eval_y)
            
            train_loss.append(temp_train_loss)
            train_acc.append(temp_train_acc)
            test_acc.append(temp_test_acc)
            
            acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_test_acc]
            acc_and_loss = [np.round(x,2) for x in acc_and_loss]
            print("Step #{} ==> Train loss: {}, Train acc: {}, Test acc: {}".format(*acc_and_loss))

if __name__ == "__main__":
    cmd_args = sys.argv
    if len(cmd_args) > 1 and cmd_args[1] == "test":
        tf.test.main(argv=cmd_args[1:])
    else:
        tf.app.run(main=None, argv=cmd_args)









