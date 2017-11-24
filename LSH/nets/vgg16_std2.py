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

def active_map(x, d = (2),name=None):
#    mean, var= tf.nn.moments(x,axes=d,keep_dims=True)
#    x = (x-mean)/tf.sqrt(var)
    x = tf.nn.sigmoid(x)
    return x

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
    
def removenan_pair(x,y):
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
        
    return x, y
   
def mmul(X):
    x = X[0]
    for i in range(1,len(X)):
        x = tf.matmul(x,X[i])
    return x
def And(X):
    x = X[0]
    for i in range(1,len(X)):
        x = tf.logical_and(x,X[i])
    return x
def msym(X):
    return (X+tf.matrix_transpose(X))/2
def mdiag(X,sz):
    X = tf.matrix_diag(X)
    if sz[2] > 0:
        return tf.concat([X,tf.zeros([sz[0],sz[2],sz[3]])],1)
    elif sz[2] < 0:
        return tf.concat([X,tf.zeros([sz[0],sz[1],abs(sz[2])])],2)
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
    
    U1  = tf.slice( U,[0,0,0],[-1,-1,s_sz[-1]])
    U2  = tf.slice( U,[0,0,s_sz[-1]],[-1,-1,-1])
    dU1 = tf.slice(dU,[0,0,0],[-1,-1,s_sz[-1]])
    dU2 = tf.slice(dU,[0,0,s_sz[-1]],[-1,-1,-1])
    
    sz = [s_sz[0],u_sz[1],u_sz[1]-s_sz[1],s_sz[1]]
    
    S = mdiag(s,sz)
    dS = mdiag(ds,sz)
    s_1 = crop_removenan(1.0/(s+1e-16))
    s_2 = tf.square(s)+1e-16
    
    k = crop_removenan(1.0/(tf.reshape(s_2,[s_sz[0],-1,1])-tf.reshape(s_2,[s_sz[0],1,-1])))
    KT = tf.matrix_transpose(tf.where(tf.eye(s_sz[-1],batch_shape=[s_sz[0]])==1.0, tf.zeros_like(k), k))
    
    s_1 = tf.matrix_diag(s_1)
    D = tf.matmul(dU1, s_1)-mmul([U2, tf.matrix_transpose(dU2),U1,s_1])
    
    DT = tf.matrix_transpose(D)
    UT = tf.matrix_transpose(U)
    VT = tf.matrix_transpose(V)
    
    grad = mmul([D + mmul([U,dS -mdiag(tf.matrix_diag_part(mmul([UT,D])),sz) + 2*mmul([S,msym(KT*mmul([VT,dV-mmul([V,DT,U,S])]))])]), VT])
    if transpose:
        grad = tf.matrix_transpose(grad)
    
    return [(grad)]

def SVD(x, dim = -1):
    ss = []
    uu = []
    vv = []
    sz = x.get_shape().as_list()
    x = tf.reshape(x,[sz[0],-1,sz[-1]])
    
    s,u,v = tf.svd(x,full_matrices=True)
    
#    s = tf.slice(s,[0,0],[-1,1])
#    u = tf.slice(u,[0,0,0],[-1,-1,1])
    v = tf.slice(v,[0,0,0],[-1,-1,1])
#    v = tf.matmul(u,tf.matmul(tf.matrix_diag(s),v,transpose_b=True))
#    v = tf.matmul(u,tf.matmul(mdiag(s,[sz[0],sz[1]*sz[2],sz[1]*sz[2]-sz[3],sz[3]]),v,transpose_b=True))
#    v = tf.matmul(mdiag(s,[sz[0],sz[1]*sz[2],sz[1]*sz[2]-sz[3],sz[3]]),v,transpose_b=True)
#    v = tf.matrix_transpose(v)
#    v = tf.matrix_transpose(tf.reduce_sum(v,1,keep_dims=True))
#    v = tf.reduce_sum(v,2,keep_dims=True)
    
    return ss, uu, v

def RBF_distillation(student, teacher, name = None, method = 'sum', dim = 1):
    st = student[0]
    sb = student[1]
    tt = teacher[0]
    tb = teacher[1]
        
    t_sz = st[0].get_shape().as_list()
    b_sz = sb[0].get_shape().as_list()
    
#    dim = int(np.sqrt(min(t_sz[1:]+b_sz[1:])))
    
#    st = tf.concat([st[0],st[1]],2)
#    sb = tf.concat([sb[0],sb[1]],2)
#    tt = tf.concat([tt[0],tt[1]],2)
#    tb = tf.concat([tb[0],tb[1]],2)
    
    st = tf.reshape(st[0],[t_sz[0],-1,1])
    sb = tf.reshape(sb[0],[b_sz[0],1,-1])
    
    tt = tf.reshape(tt[0],[t_sz[0],-1,1])
    tb = tf.reshape(tb[0],[b_sz[0],1,-1])

    s_sub = st-sb
    t_sub = tt-tb

    s_rbf = tf.contrib.layers.flatten(tf.exp(-tf.square(s_sub)/8))
    t_rbf = tf.contrib.layers.flatten(tf.exp(-tf.square(t_sub)/8))
    
#    return tf.reduce_sum(crop_removenan(tf.reduce_sum(tf.square(s_rbf-tf.stop_gradient(t_rbf)),1),scale=False))
    return (tf.nn.l2_loss(s_rbf-tf.stop_gradient(t_rbf)))

def Gram_distillation(student, teacher, name = None, method = 'sum', dim = 1):
    st = student[0]
    sb = student[1]
    tt = teacher[0]
    tb = teacher[1]
        
    s_gram = tf.matmul(st,sb,transpose_b = True)
    t_gram = tf.matmul(tt,tb,transpose_b = True)
    return tf.nn.l2_loss(s_gram-t_gram)
                
def vgg16_std(image, is_training=False, val = False, lr = None, prediction_fn=slim.softmax,scope='vgg13_std'):
    end_points = {}
    large = sio.loadmat('/home/dmsl/nas/backup1/personal_lsh/training/cifar100/vgg13.mat')
    with tf.variable_scope(scope, 'vgg16_std', [image]):
        with tf.variable_scope('block0'):
            std0 = slim.conv2d(image, 64, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv0', trainable=is_training, reuse=val)    
            std1 = slim.max_pool2d(std0, [2, 2], 2, scope='pool')
        with tf.variable_scope('block1'):
            std2 = slim.conv2d(std1, 128, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv0', trainable=is_training, reuse=val)   
            std3 = slim.max_pool2d(std2, [2, 2], 2, scope='pool')   
        with tf.variable_scope('block2'):
            std4 = slim.conv2d(std3, 256, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv0', trainable=is_training, reuse=val)  
            std5 = slim.max_pool2d(std4, [2, 2], 2, scope='pool')   
        with tf.variable_scope('block3'):
            std6 = slim.conv2d(std5, 512, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                               scope='conv0', trainable=is_training, reuse=val)   
            std7 = slim.max_pool2d(std6, [2, 2], 2, scope='pool')   
        
        fc = tf.contrib.layers.flatten(std7)
        fc1 = slim.fully_connected(fc, 1024, activation_fn=tf.nn.relu,
                                   trainable=is_training, scope = 'full1', reuse = val)
        fc = slim.dropout(fc1,0.5,is_training=is_training)
        logits = slim.fully_connected(fc, 100, activation_fn=None,
                                      trainable=is_training, scope = 'full3', reuse = val)
        
        
        
        if is_training:
            with tf.variable_scope('teacher'):
                with tf.variable_scope('block0'):
                    conv = slim.conv2d(image, 64, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                       weights_initializer = tf.constant_initializer(large['conv1w']),
                                       biases_initializer  = tf.constant_initializer(large['conv1b']),
                                       scope='conv0', trainable=False, reuse=val)        
                    teach0 = slim.conv2d(conv, 64, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                       weights_initializer = tf.constant_initializer(large['conv2w']),
                                       biases_initializer  = tf.constant_initializer(large['conv2b']),
                                       scope='conv1', trainable=False, reuse=val)     
                    teach1 = slim.max_pool2d(teach0, [2, 2], 2, scope='pool')   
                with tf.variable_scope('block1'):
                    conv = slim.conv2d(teach1, 128, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                       weights_initializer = tf.constant_initializer(large['conv3w']),
                                       biases_initializer  = tf.constant_initializer(large['conv3b']),
                                       scope='conv0', trainable=False, reuse=val)        
                    teach2 = slim.conv2d(conv, 128, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                       weights_initializer = tf.constant_initializer(large['conv4w']),
                                       biases_initializer  = tf.constant_initializer(large['conv4b']),
                                       scope='conv1', trainable=False, reuse=val)        
                    teach3 = slim.max_pool2d(teach2, [2, 2], 2, scope='pool')
                    
                with tf.variable_scope('block2'):
                    conv = slim.conv2d(teach3, 256, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                       weights_initializer = tf.constant_initializer(large['conv5w']),
                                       biases_initializer  = tf.constant_initializer(large['conv5b']),
                                       scope='conv0', trainable=False, reuse=val)        
                    conv = slim.conv2d(conv, 256, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                       weights_initializer = tf.constant_initializer(large['conv6w']),
                                       biases_initializer  = tf.constant_initializer(large['conv6b']),
                                       scope='conv1', trainable=False, reuse=val)        
                    teach4 = slim.conv2d(conv, 256, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                       weights_initializer = tf.constant_initializer(large['conv7w']),
                                       biases_initializer  = tf.constant_initializer(large['conv7b']),
                                       scope='conv2', trainable=False, reuse=val)        
                    teach5 = slim.max_pool2d(teach4, [2, 2], 2, scope='pool')
                    
                with tf.variable_scope('block3'):
                    conv = slim.conv2d(teach5, 512, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                       weights_initializer = tf.constant_initializer(large['conv8w']),
                                       biases_initializer  = tf.constant_initializer(large['conv8b']),
                                       scope='conv0', trainable=False, reuse=val)        
                    conv = slim.conv2d(conv, 512, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                       weights_initializer = tf.constant_initializer(large['conv9w']),
                                       biases_initializer  = tf.constant_initializer(large['conv9b']),
                                       scope='conv1', trainable=False, reuse=val)        
                    teach6 = slim.conv2d(conv, 512, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                       weights_initializer = tf.constant_initializer(large['conv10w']),
                                       biases_initializer  = tf.constant_initializer(large['conv10b']),
                                       scope='conv2', trainable=False, reuse=val)        
                    teach7 = slim.max_pool2d(teach6, [2, 2], 2, scope='pool')
#                    
                fc = tf.contrib.layers.flatten(teach7)
                fct1 = slim.fully_connected(fc, 1024, activation_fn=tf.nn.relu,
                                           weights_initializer = tf.constant_initializer(large['fc1w']),
                                           biases_initializer  = tf.constant_initializer(large['fc1b']),
                                           trainable=False, scope = 'fullt1', reuse = False)
                fct2 = slim.fully_connected(fct1, 1024, activation_fn=tf.nn.relu,
                                           weights_initializer = tf.constant_initializer(large['fc2w']),
                                           biases_initializer  = tf.constant_initializer(large['fc2b']),
                                           trainable=False, scope = 'fullt2', reuse = False)
                with tf.variable_scope('rbf_grammian'):
                    with tf.variable_scope('SVD'):
                        _,_,sv1 = SVD(std1)
                        _,_,tv1 = SVD(teach1)
                        sv1,tv1 = removenan_pair(sv1,tv1)
                        
                        _,_,sv2 = SVD(std2)
                        _,_,tv2 = SVD(teach2)
                        sv2,tv2 = removenan_pair(sv2,tv2)
                        
                        _,_,sv3 = SVD(std3)
                        _,_,tv3 = SVD(teach3)
                        sv3,tv3 = removenan_pair(sv3,tv3)
                        
                        _,_,sv4 = SVD(std4)
                        _,_,tv4 = SVD(teach4)
                        sv4,tv4 = removenan_pair(sv4,tv4)
                        
                        _,_,sv5 = SVD(std5)
                        _,_,tv5 = SVD(teach5)
                        sv5,tv5 = removenan_pair(sv5,tv5)
                        
                        _,_,sv6 = SVD(std6)
                        _,_,tv6 = SVD(teach6)
                        sv6,tv6 = removenan_pair(sv6,tv6)
                        
                        _,_,sv7 = SVD(std7)
                        _,_,tv7 = SVD(teach7)
                        sv7,tv7 = removenan_pair(sv7,tv7)
                        
                        fc1 = tf.reshape(fc1,[128,-1,1])
                        fc1 = [fc1,fc1,fc1]
                        fct2 = tf.reshape(fct2,[128,-1,1])
                        fct2 = [fct2,fct2,fct2]
                    v_loss = 0
                    u_loss = 0
                    fully_loss = 0
                    print (sv1.get_shape().as_list())
                    with tf.variable_scope('rbf0'):
                        v_loss0 = RBF_distillation([sv1,sv2],[tv1,tv2],'RBF01_v')
#                        v_loss0 = Gram_distillation([std1,std2],[teach1,teach2],'Gram0')
#                        v_loss0 += RBF_distillation([sv0,sv3],[tv0,tv3],'RBF03_v')/4
                        v_loss += v_loss0
#                        u_loss += RBF_distillation([su0,su1],[tu0,tu1],'RBF0_u')
                    with tf.variable_scope('rbf1'):
                        v_loss1 = RBF_distillation([sv3,sv4],[tv3,tv4],'RBF12_v')
#                        v_loss1 = Gram_distillation([std3,std4],[teach3,teach4],'Gram1')
                        v_loss += v_loss1
##                        u_loss += RBF_distillation([su1,su2],[tu1,tu2],'RBF0_u')
                    with tf.variable_scope('rbf2'):
                        v_loss += RBF_distillation([sv5,sv6],[tv5,tv6],'RBF2_v')
#                        v_loss += Gram_distillation([std5,std6],[teach5,teach6],'Gram2')
##                        u_loss += RBF_distillation([su2,su3],[tu2,tu3],'RBF0_u')
                    with tf.variable_scope('rbf3'):
                        fully_loss += RBF_distillation([sv7,fc1],[tv7, fct2],'RBF3_v')
#                        fully_loss+= RBF_distillation([su3,fc1],[tu3, fct2],'RBF3_u')
                        
                    end_points['Dist'] = v_loss + u_loss + fully_loss/(512*1024)
                    
        
    end_points['Logits'] = logits
#    end_points['fs'] = std0
#    end_points['ft'] = teach0
#    end_points['ms'] = map0
#    end_points['mt'] = mapt0
    return end_points
vgg16_std.default_image_size = 32



