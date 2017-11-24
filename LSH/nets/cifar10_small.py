from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import selu

slim = tf.contrib.slim
#trunc_normal = lambda stddev: tf.xavier_initializer()

def cifar10_small_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
        return arg_sc

def cifar10_small(image, is_training=False, val = False, lr = None, prediction_fn=slim.softmax,scope='cifar10'):
    with tf.variable_scope(scope, 'cifar10', [image]):
        conv = slim.conv2d(image, 96, [5, 5], 4, padding = 'VALID', activation_fn=None,
                           scope='conv1', trainable=is_training, reuse=val)                
        std0 = tf.nn.relu(conv)
        std0 = tf.nn.lrn(std0)
#        std0 = slim.max_pool2d(std0, [3, 3], 2, scope='pool1')
        
        conv = slim.conv2d(std0, 256, [3, 3], 4, padding = 'VALID', activation_fn=None,
                           scope='conv2', trainable=is_training, reuse=val)        
        std1 = tf.nn.relu(conv)
        std1 = tf.nn.lrn(std1)
#        std1 = slim.max_pool2d(std1, [3, 3], 2, scope='pool1')        
        
        conv = slim.conv2d(std1, 512, [3, 3], 2, padding = 'VALID', activation_fn=None,
                           scope='conv3', trainable=is_training, reuse=val)
        std2 = tf.nn.relu(conv)
        
        conv = tf.contrib.layers.flatten(std2)
        fc1 = slim.fully_connected(conv, 512, activation_fn=tf.nn.relu, trainable=is_training, scope = 'full1', reuse = val)
        fc1 = slim.dropout(fc1,0.5,is_training=is_training)
        logits = slim.fully_connected(fc1, 100, activation_fn=None, trainable=is_training, scope = 'full3', reuse = val)
        
        
    end_points = {}
    end_points['Logits1'] = logits
    #end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return end_points
cifar10_small.default_image_size = 32
