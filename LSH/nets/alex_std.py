from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import scipy.io as sio
import cv2
import numpy as np
slim = tf.contrib.slim

def alex_std_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
        return arg_sc

def active_map(x, d = (2),name=None):
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
    u_sz = dU.get_shape().as_list()
    s_sz = ds.get_shape().as_list()
    v_sz = dV.get_shape().as_list()
    transpose = False
    if u_sz[1]<v_sz[1]:
        dV_temp = dV
        dV = dU
        dU = dV_temp
        
        s, V, U = op.outputs
        
        u_sz = dU.get_shape().as_list()
        v_sz = dV.get_shape().as_list()
        
        transpose = True
    else:
        s, U, V = op.outputs
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
    sz = x.get_shape().as_list()
    
    x = tf.reshape(x,[sz[0],-1,sz[-1]])
    
    s,u,v = tf.svd(x,full_matrices=True)
    
    v = tf.slice(v,[0,0,0],[-1,-1,1])
    
    return s, u, v

def RBF_distillation(student, teacher, name = None, method = 'sum', dim = 1):
    st = student[0]
    sb = student[1]
    tt = teacher[0]
    tb = teacher[1]
        
    t_sz = st.get_shape().as_list()
    b_sz = sb.get_shape().as_list()
    
#    dim = int(np.sqrt(min(t_sz[1:]+b_sz[1:])))
    
    st = tf.slice(st,[0,0,0],[-1,-1,dim])
    sb = tf.slice(sb,[0,0,0],[-1,-1,dim])
    tt = tf.slice(tt,[0,0,0],[-1,-1,dim])
    tb = tf.slice(tb,[0,0,0],[-1,-1,dim])
    
    st = tf.reshape(st,[t_sz[0],-1,1,dim])
    sb = tf.reshape(sb,[b_sz[0],1,-1,dim])
    
    tt = tf.reshape(tt,[t_sz[0],-1,1,dim])
    tb = tf.reshape(tb,[b_sz[0],1,-1,dim])

    s_sub = st-sb
    t_sub = tt-tb

    s_rbf = tf.contrib.layers.flatten(tf.exp(-tf.square(s_sub)/8))
    t_rbf = tf.contrib.layers.flatten(tf.exp(-tf.square(t_sub)/8))
    
    return (tf.nn.l2_loss(s_rbf-tf.stop_gradient(t_rbf)))

def Gram_distillation(student, teacher, name = None, method = 'sum', dim = 1):
    st = student[0]
    sb = student[1]
    tt = teacher[0]
    tb = teacher[1]
        
    t_sz = st.get_shape().as_list()
    b_sz = sb.get_shape().as_list()
    
#    dim = int(np.sqrt(min(t_sz[1:]+b_sz[1:])))
    
    st = tf.slice(st,[0,0,0],[-1,-1,dim])
    sb = tf.slice(sb,[0,0,0],[-1,-1,dim])
    tt = tf.slice(tt,[0,0,0],[-1,-1,dim])
    tb = tf.slice(tb,[0,0,0],[-1,-1,dim])
    
    st = tf.reshape(st,[t_sz[0],-1,1])
    sb = tf.reshape(sb,[b_sz[0],1,-1])
    
    tt = tf.reshape(tt,[t_sz[0],-1,1])
    tb = tf.reshape(tb,[b_sz[0],1,-1])

    s_rbf = tf.matmul(st,sb)
    t_rbf = tf.matmul(tt,tb)
    return (tf.nn.l2_loss(s_rbf-tf.stop_gradient(t_rbf)))
                
def alex_std(image, is_training=False, val = False, lr = None, prediction_fn=slim.softmax,scope='alex_std'):
    end_points = {}
    large = sio.loadmat('/home/dmsl/nas/backup1/personal_lsh/training/cifar100/alex.mat')
    with tf.variable_scope(scope, 'alex_std', [image]):
        std0 = slim.conv2d(image, 96, [3, 3], 2, padding = 'VALID', activation_fn=tf.nn.relu,
                           scope='conv0', trainable=is_training, reuse=val)        
        std1 = slim.conv2d(std0, 256, [3, 3], 2, padding = 'VALID', activation_fn=tf.nn.relu,
                           scope='conv1', trainable=is_training, reuse=val)     
        std2 = slim.conv2d(std1, 512, [3, 3], 2, padding = 'VALID', activation_fn=tf.nn.relu,
                           scope='conv2', trainable=is_training, reuse=val)     
        fc = tf.contrib.layers.flatten(std2)
        fc1 = slim.fully_connected(fc, 1024, activation_fn=tf.nn.relu,
                                   trainable=is_training, scope = 'full1', reuse = val)
        logits = slim.fully_connected(fc1, 100, activation_fn=None, 
                                      trainable=is_training, scope = 'full3', reuse = val)
        
        
        
        if is_training:
            with tf.variable_scope(scope, 'alex', [image]):
                teach0 = slim.conv2d(image, 96, [7, 7], 2, padding = 'VALID', activation_fn=tf.nn.relu,
                                     weights_initializer = tf.constant_initializer(large['conv1w']),
                                     biases_initializer  = tf.constant_initializer(large['conv1b']),
                                     scope='conv0', trainable=False, reuse=val)        
                teach1 = slim.conv2d(teach0, 256, [5, 5], 2, padding = 'VALID', activation_fn=tf.nn.relu,
                                     weights_initializer = tf.constant_initializer(large['conv2w']),
                                     biases_initializer  = tf.constant_initializer(large['conv2b']),
                                     scope='conv1', trainable=False, reuse=val)     
                teach2 = slim.conv2d(teach1, 512, [3, 3], 1, padding = 'VALID', activation_fn=tf.nn.relu,
                                     weights_initializer = tf.constant_initializer(large['conv3w']),
                                     biases_initializer  = tf.constant_initializer(large['conv3b']),
                                     scope='conv2', trainable=False, reuse=val)     
                fc = tf.contrib.layers.flatten(teach0)
                fct1 = slim.fully_connected(fc, 1024, activation_fn=tf.nn.relu,
                                            weights_initializer = tf.constant_initializer(large['fc1w']),
                                            biases_initializer  = tf.constant_initializer(large['fc1b']),
                                            trainable=False, scope = 'full1', reuse = val)
                fct2 = slim.fully_connected(fct1, 1024, activation_fn=tf.nn.relu,
                                            weights_initializer = tf.constant_initializer(large['fc2w']),
                                            biases_initializer  = tf.constant_initializer(large['fc2b']),
                                            trainable=False, scope = 'full2', reuse = val)
                
                logits = slim.fully_connected(fct2, 100, activation_fn=None, 
                                              weights_initializer = tf.constant_initializer(large['fc3w']),
                                              biases_initializer  = tf.constant_initializer(large['fc3b']),
                                              trainable=is_training, scope = 'full3', reuse = val)
                with tf.variable_scope('distillation'):
                    with tf.variable_scope('SVD'):
                        st_s, su0, sv0 = SVD(std0)
                        tt_s, tu0, tv0 = SVD(teach0)
                        sv0,tv0 = removenan_pair(sv0,tv0)
                        
                        st_s,su1,sv1 = SVD(std1)
                        tt_s,tu1,tv1 = SVD(teach1)
                        sv1,tv1 = removenan_pair(sv1,tv1)
                        
                        st_s,su2,sv2 = SVD(std2)
                        tt_s,tu2,tv2 = SVD(teach2)
                        sv2,tv2 = removenan_pair(sv2,tv2)
                        
                        fc1 = tf.reshape(fc1,[128,-1,1])
                        fct2 = tf.reshape(fct2,[128,-1,1])
                    
                    v_loss = 0
                    u_loss = 0
                    fully_loss = 0
                    with tf.variable_scope('rbf0'):
                        v_loss0 = RBF_distillation([sv0,sv1],[tv0,tv1],'RBF01_v')
#                        v_loss0 += RBF_distillation([sv0,sv2],[tv0,tv2],'RBF02_v')/4
#                        v_loss0 += RBF_distillation([sv0,sv3],[tv0,tv3],'RBF03_v')/16
                        v_loss += v_loss0
#                        u_loss += RBF_distillation([su0,su1],[tu0,tu1],'RBF0_u')
                    with tf.variable_scope('rbf1'):
                        v_loss1 = RBF_distillation([sv1,sv2],[tv1,tv2],'RBF12_v')
#                        v_loss1 += RBF_distillation([sv1,sv3],[tv1,tv3],'RBF13_v')/4
                        v_loss += v_loss1
##                        u_loss += RBF_distillation([su1,su2],[tu1,tu2],'RBF0_u')
                    with tf.variable_scope('rbf2'):
                        fully_loss += RBF_distillation([sv2,fc1],[tv2, fct2],'RBF3_v')
#                        fully_loss+= RBF_distillation([su3,fc1],[tu3, fct2],'RBF3_u')
                        
                    end_points['Dist'] = v_loss + u_loss + fully_loss/(512*1024)
                    
                    
                    
                    
        
    end_points['Logits'] = logits
#    end_points['fs'] = std0
#    end_points['ft'] = teach0
#    end_points['ms'] = map0
#    end_points['mt'] = mapt0
    return end_points
alex_std.default_image_size = 32



