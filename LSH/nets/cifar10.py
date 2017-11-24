from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import selu
import scipy.io as sio
slim = tf.contrib.slim
#trunc_normal = lambda stddev: tf.xavier_initializer()

def cifar10_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
        return arg_sc

def CONV(x, depth, shape, name, is_training=False, val = False,batchnorm=True,stride=1, padding = 'SAME',act = True):
    if val == True:
        reuse = True
    else:
        reuse = False
    conv = slim.conv2d(x, depth, shape, stride, padding = padding,
                       biases_initializer=None, activation_fn=None,
                       scope=name, trainable=is_training, reuse=reuse)
    if batchnorm==True:
        conv = slim.batch_norm(conv, scale=True,activation_fn=None,scope=name+'/batch', is_training=is_training, reuse=reuse)
        conv = tf.nn.relu(conv)
#    if act == True:
#        conv = selu.selu(conv)
#        conv = tf.nn.relu(conv)
    if is_training == True:
        conv = selu.dropout_selu(conv)
    return conv
        
def residual(x, y, is_training):
    with tf.variable_scope('residual'):
        resi = tf.add(y,x)
#        resi = tf.nn.relu(resi)
        conv = selu.selu(resi)
        if is_training == True:
            conv = selu.dropout_selu(conv)
        return resi

def cifar10(image, is_training=False, val = False, lr = None, prediction_fn=slim.softmax,scope='cifar10'):
    with tf.variable_scope(scope, 'cifar10', [image]):
#        conv = slim.batch_norm(image, scale=True,activation_fn=None,scope='batch0', is_training=is_training, reuse=val)
#        vgg_m = sio.loadmat('/home/dmslsh/Documents/params.mat')['param'][0]
        
        conv = slim.conv2d(image, 96, [7, 7], 2, padding = 'VALID', activation_fn=None,
#                           weights_initializer = tf.constant_initializer(vgg_m[0]),
#                           biases_initializer  = tf.constant_initializer(vgg_m[1]),
                           scope='conv1', trainable=is_training, reuse=val)        
        conv = tf.nn.relu(conv)
        conv = tf.nn.lrn(conv)
        conv = slim.max_pool2d(conv, [3, 3], 2, scope='pool1')
        
        conv = slim.conv2d(conv, 256, [5, 5], 2, padding = 'VALID', activation_fn=None,
#                           weights_initializer = tf.constant_initializer(vgg_m[2]),
#                           biases_initializer  = tf.constant_initializer(vgg_m[3]),
                           scope='conv2', trainable=is_training, reuse=val)        
        conv = tf.nn.relu(conv)
        conv = tf.nn.lrn(conv)
        conv = slim.max_pool2d(conv, [3, 3], 2, scope='pool2')
        conv = slim.conv2d(conv, 512, [3, 3], 1, padding = 'VALID', activation_fn=None,
#                           weights_initializer = tf.constant_initializer(vgg_m[4]),
#                           biases_initializer  = tf.constant_initializer(vgg_m[5]),
                           scope='conv3', trainable=is_training, reuse=val)
        conv = tf.nn.relu(conv)
        
        conv = tf.contrib.layers.flatten(conv)
        fc1 = slim.fully_connected(conv, 512, activation_fn=tf.nn.relu, trainable=is_training, scope = 'full1', reuse = val)
        fc1 = slim.dropout(fc1,0.5,is_training=is_training)
        fc2 = slim.fully_connected(fc1, 512, activation_fn=tf.nn.relu, trainable=is_training, scope = 'full2', reuse = val)
        fc2 = slim.dropout(fc2,0.5,is_training=is_training)
        logits = slim.fully_connected(fc2, 100, activation_fn=None, trainable=is_training, scope = 'full3', reuse = val)
        
        
    end_points = {}
    end_points['Logits1'] = logits
    #end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return end_points
cifar10.default_image_size = 32
