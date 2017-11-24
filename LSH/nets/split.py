import tensorflow as tf
slim = tf.contrib.slim

def layer(x, n, name, is_training=False, val = False):
    out_dim = x.get_shape().as_list()
    dv_shape = out_dim[-1]//n
    x = tf.transpose(x,(2,0,1))
    
    with tf.variable_scope(name, reuse = val):
        param = tf.get_variable('param', out_dim[-1], tf.int32,
                                 trainable=is_training,
                                 initializer=tf.constant_initializer(0))
        slim.layers._add_variable_to_collections(param, tf.GraphKeys.TRAINABLE_VARIABLES, 'split_param')
        tf.summary.histogram('%s/param'%name, param)
        
        if is_training:
            kernel = tf.get_variable('kernel', out_dim[-1], tf.float32,
                                     trainable=is_training,
                                     initializer=tf.contrib.layers.variance_scaling_initializer(),
                                     regularizer = slim.l2_regularizer(5e-4))
            slim.layers._add_variable_to_collections(kernel, tf.GraphKeys.TRAINABLE_VARIABLES, 'split_kernel')
            tf.summary.histogram('%s/kernel'%name, kernel)
            
            weight = tf.get_variable('weight', out_dim[-1], tf.float32,
                                     trainable=is_training,
                                     initializer=tf.contrib.layers.variance_scaling_initializer(),
                                     regularizer = slim.l2_regularizer(5e-4))
            slim.layers._add_variable_to_collections(weight, tf.GraphKeys.TRAINABLE_VARIABLES, 'split_weight')
            tf.summary.histogram('%s/weight'%name, weight)
            
            bias = tf.get_variable('bias', out_dim[-1], tf.float32,
                                     trainable=is_training,
                                     initializer=tf.contrib.layers.variance_scaling_initializer())
            slim.layers._add_variable_to_collections(bias, tf.GraphKeys.TRAINABLE_VARIABLES, 'split_bias')
            tf.summary.histogram('%s/bias'%name, bias)
        
            ker = kernel*weight+bias
            ker = tf.nn.sigmoid(ker)
            param = tf.assign(param,tf.nn.top_k(ker, k=out_dim[-1], sorted=True).indices)
            loss = tf.reduce_mean(tf.matmul(tf.reshape(ker,[-1,1]),tf.reshape(ker,[1,-1])))\
                  +tf.reduce_mean(tf.square(ker))
        else:
            loss = 0.0
            
        split_x = tf.gather(x, param)
        split_x = tf.transpose(split_x,(1,2,0))
        split_x = tf.reshape(split_x,[out_dim[0], -1, dv_shape])
        
        
    return split_x, loss

def fc(x, d, name = None, is_training=False, val = False):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name, reuse = val):
        weight = tf.get_variable('%s/weight'%name, [in_shape[-2],in_shape[-1],d], tf.float32,
                                 trainable=is_training,
                                 initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 regularizer = slim.l2_regularizer(5e-4))
        slim.layers._add_variable_to_collections(weight, tf.GraphKeys.TRAINABLE_VARIABLES, 'weight')
        tf.summary.histogram('%s/weight'%name, weight)
        
        bias = tf.get_variable('%s/bias'%name, [in_shape[1], d], tf.float32,
                                 trainable=is_training,
                                 initializer=tf.constant_initializer(0))
        slim.layers._add_variable_to_collections(bias, tf.GraphKeys.TRAINABLE_VARIABLES, 'bias')
        tf.summary.histogram('%s/bias'%name, bias)
        
        x = tf.reshape(x,[in_shape[0],in_shape[1],in_shape[2],1])
        y = tf.reduce_sum(x*weight,2)+bias
    return y

def logits(x, d1, d2, name = None, reshape = True, is_training=False, val = False):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name, reuse = val):
        weight = tf.get_variable('%s/weight'%name, [in_shape[1], in_shape[2]], tf.float32,
                                 trainable=is_training,
                                 initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 regularizer = slim.l2_regularizer(5e-4))
        slim.layers._add_variable_to_collections(weight, tf.GraphKeys.TRAINABLE_VARIABLES, 'weight')
        tf.summary.histogram('%s/weight'%name, weight)
        
        bias = tf.get_variable('%s/bias'%name, d2, tf.float32,
                                 trainable=is_training,
                                 initializer=tf.constant_initializer(0))
        slim.layers._add_variable_to_collections(bias, tf.GraphKeys.TRAINABLE_VARIABLES, 'bias')
        tf.summary.histogram('%s/bias'%name, bias)
        
        y = tf.reduce_sum(tf.multiply(x,weight),2)+bias
    
    return y

def dropout(x, p=0.5, is_training=False):
    with tf.variable_scope('drop'):
        if is_training:
            x_shape = x.get_shape().as_list()
            x_split = []
            for i in range(x_shape[1]):
                x_split += [slim.dropout(tf.slice(x,[0,i,0],[x_shape[0],1,x_shape[-1]]),keep_prob=p,is_training=is_training)]
            return tf.concat(x_split,1)
        else:
            return x
        
                
            
            
        
        
        
        
        
        
        
        