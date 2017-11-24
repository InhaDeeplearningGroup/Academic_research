from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import selu

slim = tf.contrib.slim
#trunc_normal = lambda stddev: tf.xavier_initializer()

def imgnet64_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
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
    if act == True:
        conv = selu.selu(conv)
    return conv
        
def residual(x, y, training):
    with tf.variable_scope('residual'):
        resi = tf.add(y,x)
        conv = selu.selu(resi)
        return conv

def imgnet64(image, is_training=False, val = False, prediction_fn=slim.softmax,scope='imgnet64'):
    with tf.variable_scope(scope, 'imgnet64', [image]):
        ## img_size : 64 -> 32
        conv = CONV(image, 32, [3, 3], 'conv0_1', is_training=is_training, val = val)
        conv  = CONV(conv, 32, [3, 3], 'conv0_2', is_training=is_training, val = val)
        conv = CONV(conv, 32, [3, 3], 'conv0_3', is_training=is_training, val = val)
        conv = slim.max_pool2d(conv, [2, 2], 2, scope='pool0')
        ## img_size : 64 -> 32
        resi = CONV(conv, 64, [3, 3], 'conv1_1', is_training=is_training, val = val)
        conv = CONV(resi,  64, [3, 3], 'conv1_2', is_training=is_training, val = val)
        conv = CONV(conv,  64, [3, 3], 'conv1_3', is_training=is_training, act=False, val = val)
        conv = residual(conv, resi, is_training)
        conv = slim.max_pool2d(conv, [2, 2], 2, scope='pool1')
        ## img_size : 32 -> 16
        resi = CONV(conv, 128, [3, 3], 'conv2_1', is_training=is_training, val = val)
        conv = CONV(resi, 128, [3, 3], 'conv2_2', is_training=is_training, val = val)
        conv = CONV(conv, 128, [3, 3], 'conv2_3', is_training=is_training, act=False, val = val)
        conv = residual(conv, resi, is_training)
        conv = slim.max_pool2d(conv, [2, 2], 2, scope='pool2')
        ## img_size : 16 -> 8
        resi = CONV(conv, 128, [3, 3], 'conv3_1', is_training=is_training, val = val)
        conv = CONV(resi, 128, [3, 3], 'conv3_2', is_training=is_training, val = val)
        conv = CONV(conv, 128, [3, 3], 'conv3_3', is_training=is_training, act=False, val = val)
        conv = residual(conv, resi, is_training)
        conv = slim.max_pool2d(conv, [2, 2], 2, scope='pool3')
        ## img_size : 8 -> 8
        resi = CONV(conv, 256, [3, 3], 'conv4_1', is_training=is_training, val = val)
        conv = CONV(resi, 256, [3, 3], 'conv4_2', is_training=is_training, val = val)
        conv = CONV(conv, 256, [3, 3], 'conv4_3', is_training=is_training, act=False, val = val)
        conv = residual(conv, resi, is_training)
        ## img_size : 8 -> 8
        resi = CONV(conv, 256, [3, 3], 'conv5_1', is_training=is_training, val = val)
        conv = CONV(resi, 256, [3, 3], 'conv5_2', is_training=is_training, val = val)
        conv = CONV(conv, 256, [3, 3], 'conv5_3', is_training=is_training, act=False, val = val)
        conv = residual(conv, resi, is_training)
        ## img_size : 8 -> 4
        resi = CONV(conv, 256, [3, 3], 'conv6_1', is_training=is_training, val = val)
        conv = CONV(resi, 256, [3, 3], 'conv6_2', is_training=is_training, val = val)
        conv = CONV(conv, 256, [3, 3], 'conv6_3', is_training=is_training, act=False, val = val)
        conv = residual(conv, resi, is_training)
        conv = slim.max_pool2d(conv, [2, 2], 2, scope='pool6')
        ## img_size : 4 -> 4
        resi = CONV(conv, 512, [3, 3], 'conv7_1', is_training=is_training, val = val)
        conv = CONV(resi, 512, [3, 3], 'conv7_2', is_training=is_training, val = val)
        conv = CONV(conv, 512, [3, 3], 'conv7_3', is_training=is_training, act=False, val = val)
        conv = residual(conv, resi, is_training)
        ## img_size : 4 -> 4
        resi = CONV(conv, 512, [3, 3], 'conv8_1', is_training=is_training, val = val)
        conv = CONV(resi, 512, [3, 3], 'conv8_2', is_training=is_training, val = val)
        conv = CONV(conv, 512, [3, 3], 'conv8_3', is_training=is_training, act=False, val = val)
        conv = residual(conv, resi, is_training)
        ## img_size : 4 -> 2
        resi = CONV(conv, 512, [3, 3], 'conv9_1', is_training=is_training, val = val)
        conv = CONV(resi, 512, [3, 3], 'conv9_2', is_training=is_training, val = val)
        conv = CONV(conv, 512, [3, 3], 'conv9_3', is_training=is_training, act=False, val = val)
        conv = residual(conv, resi, is_training)
        conv = slim.max_pool2d(conv, [2, 2], 2, scope='pool9')
        ## img_size : 2 -> 1
        drop = selu.dropout_selu(conv, training = is_training)
        logits = CONV(drop, 1000, [1, 1], 'conv10', is_training=is_training, padding = 'VALID', batchnorm=False, act = False, val = val)
        logits = tf.contrib.layers.flatten(logits)
        
        
            
    end_points = {}
    end_points['Logits'] = logits
    #end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return end_points
imgnet64.default_image_size = 64
