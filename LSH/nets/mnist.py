from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import selu

slim = tf.contrib.slim
#trunc_normal = lambda stddev: tf.xavier_initializer()

def mnist_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
        return arg_sc

def CONV(x, depth, shape, name, is_training=False,batchnorm=True,stride=1, padding = 'SAME'):
    conv = slim.conv2d(x, depth, shape, stride, padding = padding,
                       biases_initializer=None, activation_fn=None,
                       scope=name, trainable=is_training)
    if batchnorm==True:
        conv = slim.batch_norm(conv, scale=True,activation_fn=None,scope=name+'/batch', is_training=is_training)
    conv = selu.selu(conv)
    conv = selu.dropout_selu(conv)
    return conv
        
def residual(x, y,stride):
    with tf.variable_scope('residual'):
        y = slim.max_pool2d(y, [3, 3], stride, scope='pool')
        resi = tf.add(y,x)
        resi = tf.nn.relu(resi)
        return resi

def mnist(image, is_training=False,prediction_fn=slim.softmax,scope='mnist'):
    with tf.variable_scope(scope, 'mnist', [image]):
        std0 = CONV(image, 64, [3, 3], 'conv1',is_training=is_training,batchnorm=False)
        conv = slim.max_pool2d(std0, [2, 2], 2, scope='pool1')
        
        std1 = CONV(conv,  64, [3, 3], 'conv2'  ,is_training=is_training,batchnorm=False)
        conv = slim.max_pool2d(std1, [2, 2], 2, scope='pool2')
        
        std2 = CONV(conv,  128, [3, 3], 'conv3'  ,is_training=is_training,batchnorm=False, padding = 'VALID', stride = 2)
        
        conv = CONV(std2,  64, [1, 1], 'conv4_d',is_training=is_training)
        conv = CONV(conv,  64, [3, 3], 'conv4'  ,is_training=is_training, padding = 'VALID')
        conv = CONV(conv,  10, [1, 1], 'conv4_u',is_training=is_training)

        logits = tf.contrib.layers.flatten(conv)
        
        
            
    end_points = {}
    end_points['Logits'] = logits
    #end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points
mnist.default_image_size = 28
