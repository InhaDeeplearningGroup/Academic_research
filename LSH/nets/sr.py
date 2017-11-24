from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
#trunc_normal = lambda stddev: tf.xavier_initializer()

def sr_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            with slim.arg_scope([slim.avg_pool2d], padding='SAME') as arg_sc:
                return arg_sc

def CONV(x, depth, shape, name, activation_fn = tf.nn.relu,reuse = None, is_training=False, batch_norm = True):
    if batch_norm == True:
        conv = slim.conv2d(x, depth, shape, scope=name,reuse = reuse,biases_initializer=None, activation_fn=None)
        conv = slim.batch_norm(conv,scale=True, activation_fn=activation_fn,scope=name+'/batch', reuse = reuse, is_training=is_training)
    else:
        conv = slim.conv2d(x, depth, shape, scope=name,reuse = reuse, activation_fn=activation_fn)
    return conv
        
def residual(x, y):
    with tf.variable_scope('residual'):
        resi = tf.add(x,y)
        resi = tf.nn.relu(resi)
        return resi
        
def sr(image, is_training=False,lr=None,prediction_fn=slim.softmax,scope='sr',val=False):
    with tf.variable_scope(scope, 'sr', [image]):
        
        conv = slim.conv2d(image, 32, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                           scope='conv0', trainable=is_training, reuse=val)
        for i in range(1,5):
            conv = slim.conv2d(conv, 64, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv%d'%i, trainable=is_training, reuse=val)
        conv = slim.conv2d(conv, 3, [3, 3], 1, padding = 'SAME', activation_fn=None,
                           scope='conv5', trainable=is_training, reuse=val)
        
        logits = conv+image
        
    end_points = {}
    end_points['Logits'] = logits
    #end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points
sr.default_image_size = 70
