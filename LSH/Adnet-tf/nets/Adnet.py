from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import scipy.io as sio
slim = tf.contrib.slim
#trunc_normal = lambda stddev: tf.xavier_initializer()

def Adnet_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
        return arg_sc

def gradient_scale(x, factor = 1):
    x = (1 - factor) * tf.stop_gradient(x) + factor * x
    return x

def fully_connected_layers(x, num_label, video_num, is_training = False, scope = None, reuse = False):
    y_s = []
    for i in range(1):
        y_s.append(slim.fully_connected(x, num_label, activation_fn=None, trainable=is_training, scope = 'full_act', reuse = reuse))
    y_s = tf.concat(y_s,2)
    y = tf.reduce_sum(y_s*video_num,2)
    return y

def Adnet(image, is_training=False, val = False, vd = None, prediction_fn=slim.softmax,scope='Adnet'):
    with tf.variable_scope(scope, 'Adnet', [image]):
        image, video_num = image
        
        conv = slim.conv2d(image, 96, [7, 7], 2, padding = 'VALID', activation_fn=tf.nn.relu,
                           scope='conv0', trainable=is_training, reuse=val)
        conv = tf.nn.lrn(conv,5,2,1e-4,0.75)
        conv = slim.max_pool2d(conv, [3, 3], 2, scope='pool0')
        
        conv = slim.conv2d(conv, 256, [5, 5], 2, padding = 'VALID', activation_fn=tf.nn.relu,
                           scope='conv1', trainable=is_training, reuse=val)
        conv = tf.nn.lrn(conv,5,2,1e-4,0.75)
        conv = slim.max_pool2d(conv, [3, 3], 2, scope='pool1')
        
        conv = slim.conv2d(conv, 512, [3, 3], 1, padding = 'VALID', activation_fn=tf.nn.relu,
                           scope='conv2', trainable=is_training, reuse=val)
        
#        conv = gradient_scale(conv, 0.1)
        conv = tf.contrib.layers.flatten(conv)
        
        fc1 = slim.fully_connected(conv, 512, activation_fn=tf.nn.relu, trainable=is_training, scope = 'full0', reuse = val)
        fc1 = slim.dropout(fc1,0.5,is_training=is_training)
        fc2 = slim.fully_connected(fc1, 512, activation_fn=tf.nn.relu, trainable=is_training, scope = 'full1', reuse = val)
        fc2 = slim.dropout(fc2,0.5,is_training=is_training)
        
        
        act = fully_connected_layers(fc2, 11, video_num, is_training = is_training, scope = 'full_act', reuse = val)
        conf = fully_connected_layers(fc2, 2, video_num, is_training = is_training, scope = 'full_conf', reuse = val)
        
    end_points = {}
    end_points['action'] = act
    end_points['conf'] = conf
    #end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return end_points
Adnet.default_image_size = 32
