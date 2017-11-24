from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import selu

slim = tf.contrib.slim
#trunc_normal = lambda stddev: tf.xavier_initializer()

def imgnet64_std_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
        return arg_sc

def gram_matrix(x,y):
    dim_x = x.get_shape().as_list()
    x = tf.reshape(x, [dim_x[0], dim_x[1] * dim_x[2], dim_x[3]])
    
    dim_y = y.get_shape().as_list()
    y = tf.reshape(y, [dim_y[0], dim_y[1] * dim_y[2], dim_y[3]])
    
    return tf.matmul(x, y, transpose_a=True)/dim_y[1] * dim_y[2]

def spilt_layer(x, n, name):
    x = tf.transpose(x,(3,0,1,2))
    
    in_shape = x.get_shape().as_list()[3]
    dv_shape = in_shape/n
    param = list(range(in_shape))
    zeros = [-1]*in_shape
    with tf.variable_scope(name):
        kernel = tf.get_variable('kernel', in_shape[3], tf.float32,
                                  initializer=tf.contrib.layers.variance_scaling_initializer())
        slim._add_variable_to_collections(kernel, 'split_param')
        split_x = []
        ub = tf.reduced_max(kernel)+1.0
        for d in range(1,n+1):
            lb = tf.reduce_min(tf.nn.top_k(kernel, k = d * dv_shape,sorted=True,name=None))
            split_idx = tf.where( ub > kernel >= lb, param, zeros)
            split_idx = tf.where(split_idx>-1)
            split_x.append(tf.transpose(tf.gather(x, split_idx),(1,2,3,0)))
            ub = lb
    
    return split_x

def CONV(x, depth, shape, name, is_training=False,batchnorm=True,stride=1, padding = 'SAME',act = True, val = False):
    if val == True:
        reuse = True
    else:
        reuse = False
    conv = slim.conv2d(x, depth, shape, stride, padding = padding,
                       biases_initializer=None, activation_fn=None,
                       scope=name, trainable=is_training, reuse = reuse)
    if batchnorm==True:
        conv = slim.batch_norm(conv, scale=True,activation_fn=None,scope=name+'/batch', is_training=is_training, reuse = reuse)
    if act == True:
        conv = selu.selu(conv)
    if is_training == True:
        conv = selu.dropout_selu(conv)
    return conv
        
def residual(x, y, is_training = True):
    with tf.variable_scope('residual'):
        resi = tf.add(y,x)
        conv = selu.selu(resi)
        conv = selu.dropout_selu(conv, training = is_training)
        return resi
    
def imgnet64_std(image, is_training=False, val = False,prediction_fn=slim.softmax,scope='imgnet64_std'):
    with tf.variable_scope('imgnet64_std', 'imgnet64_std', [image]):
        ## img_size : 64 -> 32
        std1 = CONV(image, 64, [3, 3], 'conv1', is_training=is_training, val = val, batchnorm = False)
        pool1 = slim.max_pool2d(std1, [2, 2], 2, scope='pool1')
        ## img_size : 32 -> 16
        std2  = CONV(pool1, 128, [3, 3], 'conv2', is_training=is_training, val = val, batchnorm = False)
        pool2 = slim.max_pool2d(std2, [2, 2], 2, scope='pool2')
        ## img_size : 16 -> 8
        std3  = CONV(pool2, 128, [3, 3], 'conv3', is_training=is_training, val = val, batchnorm = False)
        pool3 = slim.max_pool2d(std3, [2, 2], 2, scope='pool3')
        ## img_size : 8 -> 4
        std4  = CONV(pool3, 256, [3, 3], 'conv4', is_training=is_training, val = val, batchnorm = False)
        pool4 = slim.max_pool2d(std4, [2, 2], 2, scope='pool5')
        ## img_size : 4 -> 2
        std5  = CONV(pool4, 512, [3, 3], 'conv5', is_training=is_training, val = val, batchnorm = False)
        pool5 = slim.max_pool2d(std5, [2, 2], 2, scope='pool5')
        ## img_size : 2 -> 1
        logits = CONV(pool5, 1000, [2, 2], 'conv6', is_training=is_training, val = val, padding = 'VALID', batchnorm=False, act = False)
        logits = tf.contrib.layers.flatten(logits)
    
    with tf.variable_scope('imgnet64', 'imgnet64', [image]):
        conv = CONV(image, 64, [3, 3], 'conv1_1', is_training=False, val = val)
        conv  = CONV(conv, 64, [3, 3], 'conv1_2', is_training=False, val = val)
        teach1 = CONV(conv, 64, [3, 3], 'conv1_3', is_training=False, val = val)
        part1 = slim.max_pool2d(teach1, [2, 2], 2, scope='pool1')
        ## img_size : 32 -> 16
        conv = CONV(part1,  128, [3, 3], 'conv2_1', is_training=False, val = val)
        conv = CONV(conv,   128, [3, 3], 'conv2_2', is_training=False, val = val)
        teach2 = CONV(conv, 128, [3, 3], 'conv2_3', is_training=False, val = val)
        part2 = slim.max_pool2d(teach2, [2, 2], 2, scope='pool2')
        ## img_size : 16 -> 8
        conv = CONV(part2, 128, [3, 3], 'conv3_1', is_training=False, val = val)
        conv = CONV(conv, 128, [3, 3], 'conv3_2', is_training=False, val = val)
        conv = CONV(conv, 128, [3, 3], 'conv3_3', is_training=False, act=False, val = val)
        teach3 = residual(conv, part2, False)
        part3 = slim.max_pool2d(teach3, [2, 2], 2, scope='pool3')
        ## img_size : 8 -> 8
        conv = CONV(part3, 256, [3, 3], 'conv4_1', is_training=False, val = val)
        conv = CONV(conv, 256, [3, 3], 'conv4_2', is_training=False, val = val)
        resi1 = CONV(conv, 256, [3, 3], 'conv4_3', is_training=False, val = val)
        ## img_size : 8 -> 4
        conv = CONV(resi1, 256, [3, 3], 'conv5_1', is_training=False, val = val)
        conv = CONV(conv, 256, [3, 3], 'conv5_2', is_training=False, val = val)
        conv = CONV(conv, 256, [3, 3], 'conv5_3', is_training=False, act=False, val = val)
        teach5 = residual(conv, resi1, False)
        part4 = slim.max_pool2d(teach5, [2, 2], 2, scope='pool5')
        ## img_size : 4 -> 4
        conv = CONV(part4, 512, [3, 3], 'conv6_1', is_training=False, val = val)
        conv = CONV(conv, 512, [3, 3], 'conv6_2', is_training=False, val = val)
        resi2 = CONV(conv, 512, [3, 3], 'conv6_3', is_training=False, val = val)
        ## img_size : 4 -> 2
        conv = CONV(resi2, 512, [3, 3], 'conv7_1', is_training=False, val = val)
        conv = CONV(conv, 512, [3, 3], 'conv7_2', is_training=False, val = val)
        conv = CONV(conv, 512, [3, 3], 'conv7_3', is_training=False, val = val, act=False)
        teach6 = residual(conv, resi2, False)
        conv = slim.max_pool2d(teach6, [2, 2], 2, scope='pool6')
        
    loss1 = tf.nn.l2_loss(gram_matrix(image, std1)-gram_matrix(image, teach1))
    loss2 = tf.nn.l2_loss(gram_matrix(pool1, std2)-gram_matrix(part1, teach2))
    loss3 = tf.nn.l2_loss(gram_matrix(pool2, std3)-gram_matrix(part2, teach3))
    loss4 = tf.nn.l2_loss(gram_matrix(pool3, std4)-gram_matrix(part3, teach5))
    loss5 = tf.nn.l2_loss(gram_matrix(pool4, std5)-gram_matrix(part4, teach6))
    N = float(3*64 + 64*128 + 128*128 + 128*256 + 256*512)
    end_points = {}
    end_points['Logits'] = logits
    end_points['Distillation'] = tf.log((loss1+loss2+loss3+loss4+loss5)/N)/tf.log(10.0)
    return end_points
imgnet64_std.default_image_size = 64
