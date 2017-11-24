from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import selu
slim = tf.contrib.slim
#trunc_normal = lambda stddev: tf.xavier_initializer()

def cd_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            with slim.arg_scope([slim.avg_pool2d], padding='SAME') as arg_sc:
                return arg_sc

def CONV(x, depth, shape,stride=1, name=None, is_training=False,batchnorm=True, padding = 'SAME',act = True, val = False):
    if val == True:
        reuse = True
    else:
        reuse = False
    conv = slim.conv2d(x, depth, shape, stride, padding = padding, activation_fn=None,
                       scope=name, trainable=is_training, reuse = reuse)
    if batchnorm==True:
        conv = slim.batch_norm(conv, scale=True,activation_fn=None,scope=name+'/batch', is_training=is_training, reuse = reuse)
    if act == True:
        conv = selu.selu(conv)
    return conv

def cd(image, is_training=False,lr=None,prediction_fn=slim.softmax,scope='cd',val=False):
    end_points = {}
    image = slim.batch_norm(image, scale=True,activation_fn=None,scope='batch', is_training=is_training, reuse = val)
    with tf.variable_scope('cd'):
        conv = CONV(image, 64, [3, 3],1, 'conv0', is_training=is_training, val = val)
        
        with tf.variable_scope('conv0'):
            c1 = CONV(conv, 32, [3, 3],1, 'conv0', is_training=is_training, val = val)
            c2 = CONV(c1, 32, [3, 3],1, 'conv1', is_training=is_training, val = val)
            c3 = CONV(tf.concat([c1,c2],3), 32, [3, 3],1, 'conv2', is_training=is_training, val = val)
            c4 = CONV(tf.concat([c1,c2,c3],3), 32, [3, 3],1, 'conv3', is_training=is_training, val = val)
            
        with tf.variable_scope('conv1'):
            c1 = CONV(c4, 64, [3, 3],2, 'conv0', is_training=is_training, val = val)
            c2 = CONV(c1, 64, [3, 3],1, 'conv1', is_training=is_training, val = val)
            c3 = CONV(tf.concat([c1,c2],3), 64, [3, 3],1, 'conv2', is_training=is_training, val = val)
            c4 = CONV(tf.concat([c1,c2,c3],3), 64, [3, 3],1, 'conv3', is_training=is_training, val = val)
            
        with tf.variable_scope('conv2'):
            c1 = CONV(c4, 128, [3, 3],1, 'conv0', is_training=is_training, val = val)
            c2 = CONV(c1, 128, [3, 3],1, 'conv1', is_training=is_training, val = val)
            c3 = CONV(tf.concat([c1,c2],3), 128, [3, 3],1, 'conv2', is_training=is_training, val = val)
            c4 = CONV(tf.concat([c1,c2,c3],3), 128, [3, 3],1, 'conv3', is_training=is_training, val = val)
            
        with tf.variable_scope('conv3'):
            c1 = CONV(c4, 256, [3, 3],2, 'conv0', is_training=is_training, val = val)
            c2 = CONV(c1, 256, [3, 3],1, 'conv1', is_training=is_training, val = val)
            c3 = CONV(tf.concat([c1,c2],3), 256, [3, 3],1, 'conv2', is_training=is_training, val = val)
            c4 = CONV(tf.concat([c1,c2,c3],3), 256, [3, 3],1, 'conv3', is_training=is_training, val = val)
            
        with tf.variable_scope('conv4'):
            c1 = CONV(c4, 512, [3, 3],2, 'conv0', is_training=is_training, val = val)
            c2 = CONV(c1, 512, [3, 3],1, 'conv1', is_training=is_training, val = val)
            c3 = CONV(tf.concat([c1,c2],3), 512, [3, 3],1, 'conv2', is_training=is_training, val = val)
            c4 = CONV(tf.concat([c1,c2,c3],3), 512, [3, 3],1, 'conv3', is_training=is_training, val = val)
            
        with tf.variable_scope('conv5'):
            c1 = CONV(c4, 512, [3, 3],2, 'conv0', is_training=is_training, val = val)
            c2 = CONV(c1, 512, [3, 3],1, 'conv1', is_training=is_training, val = val)
            c3 = CONV(tf.concat([c1,c2],3), 512, [3, 3],1, 'conv2', is_training=is_training, val = val)
            c4 = CONV(tf.concat([c1,c2,c3],3), 512, [3, 3],1, 'conv3', is_training=is_training, val = val)
        
        with tf.variable_scope('fc1'):
            conv = tf.contrib.layers.flatten(c4)
            fc1 = slim.fully_connected(conv, 512, activation_fn=tf.nn.relu, trainable=is_training, scope = 'full1', reuse = val)
            fc1 = slim.batch_norm(fc1, scale=True,activation_fn=tf.nn.relu,scope='batch0', is_training=is_training, reuse=val)
            fc1 = slim.dropout(fc1,is_training=is_training)
            
        with tf.variable_scope('fc2'):
            fc2 = slim.fully_connected(fc1, 512, activation_fn=tf.nn.relu, trainable=is_training, scope = 'full2', reuse = val)
            fc2 = slim.batch_norm(fc2, scale=True,activation_fn=tf.nn.relu,scope='batch0', is_training=is_training, reuse=val)
            fc2 = slim.dropout(fc2,is_training=is_training)
            
            logits = slim.fully_connected(fc2, 2, activation_fn=None, trainable=is_training, scope = 'full5', reuse = val)
    
    #end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    end_points['Logits'] = logits
    return end_points
cd.default_image_size = 70
