from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import selu
import scipy.io as sio
slim = tf.contrib.slim
#trunc_normal = lambda stddev: tf.xavier_initializer()

def ECCV_small_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
        return arg_sc
def resi_block(x, depth, stride = 1, is_training= False, val = False, name=None):
    with tf.variable_scope('resi_%s'%name):
        conv = slim.conv2d(x, depth, [3, 3], stride, padding = 'SAME', activation_fn=None,
                           scope='%s/conv0'%name, trainable=is_training, reuse=val)
        conv = slim.batch_norm(conv, scale=True,activation_fn=tf.nn.relu,scope='%s/batch0'%name, is_training=is_training, reuse = val)
        conv = slim.conv2d(conv, depth, [3, 3], 1, padding = 'SAME', activation_fn=None,
                           scope='%s/conv1'%name, trainable=is_training, reuse=val)
        conv = slim.batch_norm(conv, scale=True,activation_fn=None,scope='%s/batch1'%name, is_training=is_training, reuse = val)
        
        if stride == 1:
            return tf.nn.relu(x+conv)
        else:
            return tf.nn.relu(conv)
    
def ECCV_small(image, is_training=False, val = False, lr = None, prediction_fn=slim.softmax,scope='ECCV_small'):
    with tf.variable_scope(scope, 'ECCV_small', [image]):
        image = slim.batch_norm(image, scale=True,activation_fn=None,scope='batch0', is_training=is_training, reuse = val)
        image = slim.conv2d(image, 64, [3, 3], 1, padding = 'SAME', activation_fn=None,
                               scope='conv0', trainable=is_training, reuse=val)
        image = slim.batch_norm(image, scale=True,activation_fn=tf.nn.relu,scope='batch1', is_training=is_training, reuse = val)
        image = slim.max_pool2d(image, [2, 2], 2, scope='pool')
        
        with tf.variable_scope('block0'):
            x = resi_block(image,128,2,is_training=is_training, val=val, name = 'conv0')
            x = resi_block(x,128,is_training=is_training, val=val, name = 'conv1')
#            x = slim.max_pool2d(x, [2, 2], 2, scope='pool')
            
        with tf.variable_scope('block1'):
            x = resi_block(x,256,2,is_training=is_training, val=val, name = 'conv0')
            x = resi_block(x,256,is_training=is_training, val=val, name = 'conv1')
#            x = slim.max_pool2d(x, [2, 2], 2, scope='pool')
            
        with tf.variable_scope('block2'):
            x = resi_block(x,512,2,is_training=is_training, val=val, name = 'conv0')
            x = resi_block(x,512,is_training=is_training, val=val, name = 'conv1')
#            x = slim.max_pool2d(x, [2, 2], 2, scope='pool')
        
        fc = tf.reduce_mean(x,[1,2])
#        fc = slim.avg_pool2d(x, [4, 4], 1, scope='pool')
#        fc = tf.contrib.layers.flatten(fc)
#        fc = slim.dropout(fc,0.5,is_training=is_training)
        logits = slim.fully_connected(fc, 10, activation_fn=None, trainable=is_training, scope = 'full3', reuse = val)
        
        
    end_points = {}
    end_points['Logits'] = logits
    #end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return end_points
ECCV_small.default_image_size = 32
