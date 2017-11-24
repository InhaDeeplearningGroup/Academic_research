from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import selu
import scipy.io as sio
slim = tf.contrib.slim
#trunc_normal = lambda stddev: tf.xavier_initializer()

def vgg16_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
        return arg_sc

def vgg16(image, is_training=False, val = False, lr = None, prediction_fn=slim.softmax,scope='vgg13'):
    with tf.variable_scope(scope, 'vgg13', [image]):
        with tf.variable_scope('block0'):
            conv = slim.conv2d(image, 64, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv0', trainable=is_training, reuse=val)
            conv = slim.conv2d(conv, 64, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv1', trainable=is_training, reuse=val)
            conv = slim.max_pool2d(conv, [2, 2], 2, scope='pool')
        with tf.variable_scope('block1'):
            conv = slim.conv2d(conv, 128, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv0', trainable=is_training, reuse=val)
            conv = slim.conv2d(conv, 128, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv1', trainable=is_training, reuse=val)
            conv = slim.max_pool2d(conv, [2, 2], 2, scope='pool')
        with tf.variable_scope('block2'):
            conv = slim.conv2d(conv, 256, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv0', trainable=is_training, reuse=val)        
            conv = slim.conv2d(conv, 256, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv1', trainable=is_training, reuse=val)        
            conv = slim.conv2d(conv, 256, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv2', trainable=is_training, reuse=val)        
            conv = slim.max_pool2d(conv, [2, 2], 2, scope='pool')
        with tf.variable_scope('block3'):
            conv = slim.conv2d(conv, 512, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv0', trainable=is_training, reuse=val)        
            conv = slim.conv2d(conv, 512, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv1', trainable=is_training, reuse=val)        
            conv = slim.conv2d(conv, 512, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv2', trainable=is_training, reuse=val)        
            conv = slim.max_pool2d(conv, [2, 2], 2, scope='pool')
#        with tf.variable_scope('block4'):
#            conv = slim.conv2d(conv, 512, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
#                               scope='conv0', trainable=is_training, reuse=val)        
#            conv = slim.conv2d(conv, 512, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
#                               scope='conv1', trainable=is_training, reuse=val)        
#            conv = slim.conv2d(conv, 512, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
#                               scope='conv2', trainable=is_training, reuse=val)        
#            conv = slim.max_pool2d(conv, [2, 2], 2, scope='pool')
        
        conv = tf.contrib.layers.flatten(conv)
        fc1 = slim.fully_connected(conv, 1024, activation_fn=tf.nn.relu, trainable=is_training, scope = 'full1', reuse = val)
        fc1 = slim.dropout(fc1,0.5,is_training=is_training)
        fc2 = slim.fully_connected(fc1, 1024, activation_fn=tf.nn.relu, trainable=is_training, scope = 'full2', reuse = val)
        fc2 = slim.dropout(fc2,0.5,is_training=is_training)
        logits = slim.fully_connected(fc2, 100, activation_fn=None, trainable=is_training, scope = 'full3', reuse = val)
        
        
    end_points = {}
    end_points['Logits'] = logits
    #end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return end_points
vgg16.default_image_size = 32
