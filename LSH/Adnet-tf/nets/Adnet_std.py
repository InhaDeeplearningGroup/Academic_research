from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import scipy.io as sio
import cv2
import numpy as np
slim = tf.contrib.slim

def vgg16_std_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
        return arg_sc

def active_map(x, y,name=None):
#    return [x], [y]
    sh = list(range(len(x.get_shape().as_list())))[1:]
    mean,var = tf.nn.moments(x,sh,keep_dims=True)
    x_norm = (x-mean)/tf.sqrt(var)
    mean,var = tf.nn.moments(y,sh,keep_dims=True)
    y_norm = (y-mean)/tf.sqrt(var)
    
    x_a = tf.nn.sigmoid(x_norm)
    y_a = tf.nn.sigmoid(y_norm)
    active = tf.stop_gradient(1+x_a+y_a)
    return [x*active], [y*active]



def removenan_pair(X,Y):
    for i in range(1):
        x = X[i]
        y = Y[i]
        isfin = tf.logical_and(tf.is_finite(x),tf.is_finite(y))
        
        sz = x.get_shape().as_list()
        sh = list(range(len(x.get_shape().as_list())))[1:]
        x_sum = tf.reduce_sum(x,sh,keep_dims=True)
        y_sum = tf.reduce_sum(y,sh,keep_dims=True)
        isfin_sum = tf.logical_and(tf.is_finite(x_sum),tf.is_finite(y_sum))
        isfin_sum = tf.cast(isfin_sum,tf.float32)
        mask = isfin_sum*sz[0]/(tf.reduce_sum(isfin_sum)+1e-16)
        
        x = tf.where(isfin, x,tf.zeros_like(x))
        y = tf.where(isfin, y,tf.zeros_like(y))
        
        x = (1 - mask) * tf.stop_gradient(x) + mask * x
        y = (1 - mask) * tf.stop_gradient(y) + mask * y
        
        X[i] = x
        Y[i] = y
        
    return tf.concat(X,2), tf.concat(Y,2)


def crop_removenan(x,scale = True):
    isfin = tf.is_finite(x)
    
    sz = x.get_shape().as_list()
    sh = list(range(len(x.get_shape().as_list())))[1:]
    x_sum = tf.reduce_sum(x,sh,keep_dims=True)
    isfin_sum = tf.is_finite(x_sum)
    isfin_sum = tf.cast(isfin_sum,tf.float32)
    mask = isfin_sum*sz[0]/(tf.reduce_sum(isfin_sum)+1e-16)
    
    x = tf.where(isfin, x,tf.zeros_like(x))
    x *= mask
    
    return x

def mmul(X):
    x = X[0]
    for i in range(1,len(X)):
        x = tf.matmul(x,X[i])
    return x
def msym(X):
    return (X+tf.matrix_transpose(X))/2
def mdiag(X,sz):
    X = tf.matrix_diag(X)
    if sz[2] > 0:
        return tf.concat([X,tf.zeros([sz[0],sz[2],sz[3]])],1)
    elif sz[2] < 0:
        return tf.concat([X,tf.zeros([sz[0],sz[1],abs(sz[2])])],2)
    else:
        return X
@tf.RegisterGradient('Svd')
def gradient_svd(op, ds, dU, dV):
    s, U, V = op.outputs
    u_sz = dU.get_shape().as_list()
    s_sz = ds.get_shape().as_list()
    v_sz = dV.get_shape().as_list()
    transpose = False
    if u_sz[1]<v_sz[1]:
        dV_temp = dV
        dV = dU
        dU = dV_temp
        
        V_temp = V
        V = U
        U = V_temp
        
        u_sz = dU.get_shape().as_list()
        v_sz = dV.get_shape().as_list()
        
        transpose = True
    
    U = crop_removenan(U)
    V = crop_removenan(V)
    s = crop_removenan(s)
    ds = crop_removenan(ds)
    
    sz = [s_sz[0],u_sz[1],u_sz[1]-s_sz[1],s_sz[1]]
    
    S = mdiag(s,sz)
    dS = mdiag(ds,sz)
    s_1 = crop_removenan(1.0/(s+1e-16))
    s_2 = tf.square(s)+1e-16
    
    k = crop_removenan(1.0/(tf.reshape(s_2,[s_sz[0],-1,1])-tf.reshape(s_2,[s_sz[0],1,-1])))
    KT = tf.matrix_transpose(tf.where(tf.eye(s_sz[-1],batch_shape=[s_sz[0]])==1.0, tf.zeros_like(k), k))
    
    s_1 = tf.matrix_diag(s_1)
    
    U1  = tf.slice( U,[0,0,0],[-1,-1,s_sz[-1]])
    U2  = tf.slice( U,[0,0,s_sz[-1]],[-1,-1,-1])
    dU1 = tf.slice(dU,[0,0,0],[-1,-1,s_sz[-1]])
    dU2 = tf.slice(dU,[0,0,s_sz[-1]],[-1,-1,-1])
    D = tf.matmul(dU1, s_1)-mmul([U2, tf.matrix_transpose(dU2),U1,s_1])
    
    DT = tf.matrix_transpose(D)
    UT = tf.matrix_transpose(U)
    VT = tf.matrix_transpose(V)
    
    grad = mmul([D + mmul([U,dS -mdiag(tf.matrix_diag_part(mmul([UT,D])),sz) + 2*mmul([S,msym(KT*mmul([VT,dV-mmul([V,DT,U,S])]))])]), VT])
    if transpose:
        grad = tf.matrix_transpose(grad)
    
    return [grad]

def SVD(X, dim = -1):
    sz = X[0].get_shape().as_list()
    V = []
    for i in range(1):
        x = tf.reshape(X[i],[sz[0],-1,sz[-1]])
        s,u,v = tf.svd(x,full_matrices=True)
#        V.append(v)
        V.append(tf.slice(v,[0,0,0],[-1,-1,1]))
        
    return s, u, V

def RBF_distillation(student, teacher, name = None, method = 'sum', dim = 1):
    st = student[0]
    sb = student[1]
    tt = teacher[0]
    tb = teacher[1]
        
    t_sz = st.get_shape().as_list()
    b_sz = sb.get_shape().as_list()
    
    st = tf.reshape(st,[t_sz[0],-1,1])
    sb = tf.reshape(sb,[b_sz[0],1,-1])
    
    tt = tf.reshape(tt,[t_sz[0],-1,1])
    tb = tf.reshape(tb,[b_sz[0],1,-1])

    s_sub = st-sb
    t_sub = tt-tb

    s_rbf = tf.contrib.layers.flatten(tf.exp(-tf.square(s_sub)/8))
    t_rbf = tf.contrib.layers.flatten(tf.exp(-tf.square(t_sub)/8))
    
    return (tf.nn.l2_loss(s_rbf-tf.stop_gradient(t_rbf)))

def fully_connected_layers(x, num_label, video_num, is_training = False, scope = None, reuse = False):
    y_s = []
    for i in range(1):
        y_s.append(slim.fully_connected(x, num_label, activation_fn=None, trainable=is_training, scope = 'full_act', reuse = reuse))
    y_s = tf.concat(y_s,2)
    y = tf.reduce_sum(y_s*video_num,2)
    return y
                
def vgg16_std(image, is_training=False, val = False, lr = None, prediction_fn=slim.softmax,scope='vgg13_std'):
    end_points = {}
    large = sio.loadmat('/home/dmsl/nas/backup1/personal_lsh/training/cifar100/vgg13.mat')
    with tf.variable_scope(scope, 'vgg16_std', [image]):
        image, video_num = image
        
        conv = slim.conv2d(image, 96, [5, 5], 2, padding = 'VALID', activation_fn=tf.nn.relu,
                           scope='conv0', trainable=is_training, reuse=val)
        conv = tf.nn.lrn(conv,5,2,1e-4,0.75)
        std0 = slim.max_pool2d(conv, [3, 3], 2, scope='pool0')
        
        conv = slim.conv2d(std0, 256, [3, 3], 2, padding = 'VALID', activation_fn=tf.nn.relu,
                           scope='conv1', trainable=is_training, reuse=val)
        conv = tf.nn.lrn(conv,5,2,1e-4,0.75)
        std1 = slim.max_pool2d(conv, [3, 3], 2, scope='pool1')
        
        std2 = slim.conv2d(std1, 512, [3, 3], 1, padding = 'VALID', activation_fn=tf.nn.relu,
                           scope='conv2', trainable=is_training, reuse=val)
        
        conv = tf.contrib.layers.flatten(conv)
        
        fcs = slim.fully_connected(conv, 512, activation_fn=tf.nn.relu, trainable=is_training, scope = 'full', reuse = val)
        fc = slim.dropout(fcs,0.5,is_training=is_training)
        
        act = fully_connected_layers(fc, 11, video_num, is_training = is_training, scope = 'full_act', reuse = val)
        conf = fully_connected_layers(fc, 2, video_num, is_training = is_training, scope = 'full_conf', reuse = val)
        
        
        
        if is_training:
            with tf.variable_scope('teacher'):
                conv = slim.conv2d(image, 96, [7, 7], 2, padding = 'VALID', activation_fn=tf.nn.relu,
                                   weights_initializer = tf.constant_initializer(large['conv0w']),
                                   biases_initializer  = tf.constant_initializer(large['conv0b']),
                                   scope='conv0', trainable=False, reuse=val)
                conv = tf.nn.lrn(conv,5,2,1e-4,0.75)
                teach0 = slim.max_pool2d(conv, [3, 3], 2, scope='pool0')
                
                conv = slim.conv2d(teach0, 256, [5, 5], 2, padding = 'VALID', activation_fn=tf.nn.relu,
                                   weights_initializer = tf.constant_initializer(large['conv1w']),
                                   biases_initializer  = tf.constant_initializer(large['conv1b']),
                                   scope='conv1', trainable=False, reuse=val)
                conv = tf.nn.lrn(conv,5,2,1e-4,0.75)
                teach1 = slim.max_pool2d(conv, [3, 3], 2, scope='pool1')
                
                teach2 = slim.conv2d(teach1, 512, [3, 3], 1, padding = 'VALID', activation_fn=tf.nn.relu,
                                     weights_initializer = tf.constant_initializer(large['conv2w']),
                                     biases_initializer  = tf.constant_initializer(large['conv2b']),
                                     scope='conv2', trainable=False, reuse=val)
                
                conv = tf.contrib.layers.flatten(teach2)
                
                fc = slim.fully_connected(conv, 512, activation_fn=tf.nn.relu,
                                          weights_initializer = tf.constant_initializer(large['fc0w']),
                                          biases_initializer  = tf.constant_initializer(large['fc0b']),
                                          trainable=False, scope = 'fullt0', reuse = val)
                fct = slim.fully_connected(fc, 512, activation_fn=tf.nn.relu,
                                           weights_initializer = tf.constant_initializer(large['fc1w']),
                                           biases_initializer  = tf.constant_initializer(large['fc1b']),
                                           trainable=False, scope = 'fullt1', reuse = val)
                
                
                with tf.variable_scope('rbf_grammian'):
                    std0, teach0 = active_map(std0, teach0)
                    std1, teach1 = active_map(std1, teach1)
                    std2, teach2 = active_map(std2, teach2)
                    fcs, fct = active_map(fcs, fct)
                    
                    with tf.variable_scope('SVD'):
                        _,_,sv0 = SVD(std0)
                        _,_,tv0 = SVD(teach0)
                        sv0,tv0 = removenan_pair(sv0,tv0)
                        
                        _,_,sv1 = SVD(std1)
                        _,_,tv1 = SVD(teach1)
                        sv1,tv1 = removenan_pair(sv1,tv1)
                        
                        _,_,sv2 = SVD(std2)
                        _,_,tv2 = SVD(teach2)
                        sv2,tv2 = removenan_pair(sv2,tv2)
                        
                        
                        sz = fcs[0].get_shape().as_list()[0]
                        fcs = tf.reshape(fcs,[sz,-1,1])
                        fct = tf.reshape(fct,[sz,-1,1])
                    v_loss = 0
                    u_loss = 0
                    fully_loss = 0

                    with tf.variable_scope('rbf0'):
                        v_loss = RBF_distillation([sv0,sv1],[tv0,tv1],'RBF0_v')
                    with tf.variable_scope('rbf1'):
                        v_loss += RBF_distillation([sv1,sv2],[tv1,tv2],'RBF1_v')
                    with tf.variable_scope('rbf2'):
                        v_loss += RBF_distillation([sv2,fcs],[tv2,fct],'RBF2_v')
                        
                    end_points['Dist'] = v_loss + u_loss + fully_loss/(512*512)
                    
        
    end_points['action'] = act
    end_points['conf'] = conf
    return end_points
vgg16_std.default_image_size = 32







