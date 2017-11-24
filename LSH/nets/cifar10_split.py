from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import selu

slim = tf.contrib.slim
#trunc_normal = lambda stddev: tf.xavier_initializer()

def cifar10_split_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
        return arg_sc

def CONV(x, depth, shape, name, is_training=False, val = False,batchnorm=True,stride=1, padding = 'SAME',act = True):
    if val == True:
        reuse = True
    else:
        reuse = False
    conv = slim.conv2d(x, depth, shape, stride, padding = padding,
                       activation_fn=None, scope=name, trainable=is_training, reuse=reuse)
    if act == True:
#        conv = selu.selu(conv)
#        conv = tf.nn.elu(conv)
        conv = tf.nn.relu(conv)
    if batchnorm==True:
        conv = slim.batch_norm(conv, scale=True,activation_fn=None,scope=name+'/batch', is_training=is_training, reuse=reuse)
    return conv

def spilt_layer(x, n, name, is_training=False, val = False):
    x = tf.transpose(x,(3,0,1,2))
    
    in_shape = x.get_shape().as_list()[0]
    dv_shape = tf.to_int32(in_shape/n)
    param = list(range(in_shape))
    zeros = [-1]*in_shape
    with tf.variable_scope(name, reuse = val):
        kernel = tf.get_variable('kernel', in_shape, tf.float32,
                                 trainable=is_training,
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
        slim.layers._add_variable_to_collections(kernel, None, 'split_param')
        split_x = []
        ub = tf.reduce_max(kernel)+1.0
        for d in range(1,n+1):
            lb = tf.nn.top_k(kernel, k = d * dv_shape, name=None).values
            lb = tf.reduce_min(lb)
            split_idx = tf.where(ub > kernel, kernel, zeros)
            split_idx = tf.where(split_idx >= lb, param, zeros)
            split_idx = tf.reshape(tf.where(split_idx>-1),[dv_shape])
            split_x.append(tf.transpose(tf.gather(x, split_idx),(1,2,3,0)))
            ub = lb
    
    return split_x
        
def star4_layer(x, depth, split, name, mp=True, is_training = False, val = False):
    with tf.variable_scope(name+'/top'):
        top = CONV(x, depth, [3, 3], name+'/top', is_training=is_training, val = val)
        
    if split:
        with tf.variable_scope(name+'/sub_conv'):
            conv = slim.conv2d(top, depth*3, [3, 3], activation_fn=None, scope='sub', trainable=is_training, reuse=val)
            conv_split = spilt_layer(conv, 3, 'split',is_training=is_training, val = val)
            sub1 = conv_split[0]
            sub2 = conv_split[1]
            sub3 = conv_split[2]
    else:
        with tf.variable_scope(name+'/sub_conv'):
            sub1 = slim.conv2d(top, depth, [3, 3], activation_fn=None, scope=name+'/sub1', trainable=is_training, reuse=val)
            sub2 = slim.conv2d(top, depth, [3, 3], activation_fn=None, scope=name+'/sub2', trainable=is_training, reuse=val)
            sub3 = slim.conv2d(top, depth, [3, 3], activation_fn=None, scope=name+'/sub3', trainable=is_training, reuse=val)
    
    convs = [sub1-sub2, sub2-sub1, sub1-sub3, sub2-sub3]
    for i, conv in enumerate(convs):
        with tf.variable_scope(name+'/act/batch%d'%i):
#            conv1 = selu.selu(sub1-sub2)
            conv = tf.nn.elu(conv)
            conv = slim.batch_norm(conv, scale=True,activation_fn=None,scope=name+'/sub%d/batch'%i,
                                   is_training=is_training, reuse=val)
            conv = top + conv
        if mp == True:
            conv = slim.max_pool2d(conv, [2, 2], 2, scope=name+'/pool%d'%i)
            
        convs[i] = conv
    return convs

def star_layer(x, depth, split, name, mp=True, is_training = False, val = False):
    with tf.variable_scope(name+'/top'):
        top = CONV(x, depth, [3, 3], name+'/top', is_training=is_training, val = val)
        
    if split:
        with tf.variable_scope(name+'/sub_conv'):
            conv = slim.conv2d(top, depth*2, [3, 3], activation_fn=None, scope=name+'sub', trainable=is_training, reuse=val)
            conv_split = spilt_layer(conv, 2, 'split',is_training=is_training, val = val)
            sub1 = conv_split[0]
            sub2 = conv_split[1]
    else:
        with tf.variable_scope(name+'/sub_conv'):
            sub1 = slim.conv2d(top, depth, [1, 3], activation_fn=None, scope=name+'/sub1', trainable=is_training, reuse=val)
            sub2 = slim.conv2d(top, depth, [3, 1], activation_fn=None, scope=name+'/sub2', trainable=is_training, reuse=val)
        
    with tf.variable_scope(name+'/act/batch1'):
#        sub1 = selu.selu(sub1-sub2)
        sub1 = tf.nn.elu(sub1-sub2)
        conv1 = slim.batch_norm(sub1, scale=True,activation_fn=None,scope=name+'/sub1/batch', is_training=is_training, reuse=val)
        conv1 = top + conv1
    with tf.variable_scope(name+'/act/batch2'):
#        sub2 = selu.selu(sub2-sub1)
        sub2 = tf.nn.elu(sub2-sub1)
        conv2 = slim.batch_norm(sub2, scale=True,activation_fn=None,scope=name+'/sub2/batch', is_training=is_training, reuse=val)
        conv2 = top + conv2
    
    if mp == True:
        conv1 = slim.max_pool2d(conv1, [2, 2], 2, scope=name+'/pool1')
        conv2 = slim.max_pool2d(conv2, [2, 2], 2, scope=name+'/pool2')
        
    return [conv1, conv2]

def residual(x, y):
    with tf.variable_scope('residual'):
        resi = tf.add(y,x)
        resi = tf.nn.relu(resi)
        return resi

def cifar10_split(image, is_training=False, val = False, prediction_fn=slim.softmax,scope='cifar10'):
    with tf.variable_scope(scope, 'cifar10', [image]):
        ## img_size : 64 -> 32
        image = slim.batch_norm(image, scale=True,activation_fn=None,scope='batch0', is_training=is_training, reuse=val)
        
        layer0 = CONV(image, 32, [3, 3], 'conv0', is_training=is_training, val = val)
        layer0 = slim.max_pool2d(layer0, [2, 2], 2, scope='conv0_pool')
        
        layer0 = spilt_layer(layer0, 2, 'split',is_training=is_training, val = val)
        layer1 = []
        for n, conv in enumerate(layer0):
            c = CONV(conv, 64, [3, 3], 'conv1_%d_1'%n, is_training=is_training, val = val)
            c = CONV(c, 64, [3, 3], 'conv1_%d_2'%n, is_training=is_training, val = val)
            c = CONV(c, 128, [3, 3], 'conv1_%d_4'%n, is_training=is_training, val = val)
            c = slim.max_pool2d(c, [2, 2], 2, scope='conv1_pool%d'%n)
            layer1 += spilt_layer(c, 2, 'split1_%d'%n,is_training=is_training, val = val)
        layer2 = []
        for n, conv in enumerate(layer1):
            c = CONV(conv, 64, [3, 3], 'conv2_%d_1'%n, is_training=is_training, val = val)
            c = CONV(c, 64, [3, 3], 'conv2_%d_2'%n, is_training=is_training, val = val)
            c = CONV(c, 128, [3, 3], 'conv2_%d_3'%n, is_training=is_training, val = val)
            c = slim.max_pool2d(c, [2, 2], 2, scope='conv2_pool%d'%n)
            layer2 += spilt_layer(c, 2, 'split2_%d'%n,is_training=is_training, val = val)
        layer3 = []
        for n, conv in enumerate(layer2):
            c = CONV(conv, 64, [3, 3], 'conv3_%d_1'%n, is_training=is_training, val = val)
            c = CONV(c, 64, [3, 3], 'conv3_%d_2'%n, is_training=is_training, val = val)
            c = CONV(c, 128, [3, 3], 'conv3_%d_3'%n, is_training=is_training, val = val)
            layer3 += spilt_layer(c, 2, 'split3_%d'%n,is_training=is_training, val = val)
        
        
#        layer0 = star4_layer(image, 32, True, 'conv0', is_training = is_training, val = val)
#        
#        layer1 = []
#        for n,conv in enumerate(layer0):
#            layer1 += star_layer(conv, 64, True, 'conv1_%d'%n, is_training = is_training, val = val)
#        layer2 = []
#        for n,conv in enumerate(layer1):
#            layer2 += star_layer(conv, 64, True, 'conv2_%d'%n, is_training = is_training, val = val)
#        layer3 = []
#        for n,conv in enumerate(layer2):
#            layer3 += star_layer(conv, 64, True, 'conv3_%d'%n, mp = False, is_training = is_training, val = val)
            
#        for i in range(int(len(layer3))):
#            if i == 0:
#                layer4 = layer3[i]
#            else:
#                layer4 = tf.concat([layer4,layer3[i]], 3)
#        layer4 = slim.avg_pool2d(layer3, [4, 4], 1, scope='pool')
#        layer4 = tf.contrib.layers.flatten(layer4)
#        logits = slim.fully_connected(layer4, 20, activation_fn=None, trainable=is_training, scope = 'full', reuse = val)
        k=2
        for n in range(k):
            for i in range(int(len(layer3)/k)*n,int(len(layer3)/k)*(n+1)):
                if i == int(len(layer3)/k)*n:
                    layer4 = layer3[i]
                else:
                    layer4 = tf.concat([layer4,layer3[i]], 3)
            layer4 = slim.avg_pool2d(layer4, [4, 4], 1, scope='pool')
            layer4 = tf.contrib.layers.flatten(layer4)
            layer4 = slim.fully_connected(layer4, 5, activation_fn=None, trainable=is_training, scope = 'full%d'%n, reuse = val)
            if n == 0:
                logits = layer4
            else:
                logits = tf.concat([logits, layer4],1)
            
    end_points = {}
    end_points['Logits'] = logits
    #end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return end_points
cifar10_split.default_image_size = 32
