import tensorflow as tf
# this gradient based on https://arxiv.org/abs/1509.07838
# and i modify to compute truncated SVD
# this code compute gradient BxMxN matrix and it doesn't matter with shape

def crop_removenan(x,scale = True):
    isfin = tf.is_finite(x)
    sh = list(range(len(x.get_shape().as_list())))[1:]
    isfin_batch = tf.reduce_min(tf.cast(isfin,tf.float32),sh,keep_dims=True)
    
    x = tf.where(isfin, x,tf.zeros_like(x))
    x *= isfin_batch
    
    return x

def mmul(X):
    x = X[0]
    for i in range(1,len(X)):
        x = tf.matmul(x,X[i])
    return x
def msym(X):
    return (X+tf.matrix_transpose(X))/2

@tf.RegisterGradient('Svd')
def gradient_svd(op, ds, dU, dV):
    s, U, V = op.outputs
    u_sz = dU.get_shape().as_list()
    s_sz = ds.get_shape().as_list()
    v_sz = dV.get_shape().as_list()
    
    transpose = False
    if u_sz[1]<v_sz[1]:
        (dU, dV) = (dV, dU)
        (U, V) = (V, U)
        
        transpose = True

    S = tf.matrix_diag(s)
    s_1 = tf.matrix_diag(1/s)
    s_2 = tf.square(s)
    
    k = crop_removenan(1.0/(tf.reshape(s_2,[s_sz[0],-1,1])-tf.reshape(s_2,[s_sz[0],1,-1])))
    KT = tf.matrix_transpose(tf.where(tf.eye(s_sz[-1],batch_shape=[s_sz[0]])==1.0, tf.zeros_like(k), k))
    
    D = tf.matmul(dU, s_1)
    
    grad = tf.matmul(D,V,transpose_b=True)\
         + tf.matmul(tf.matmul(U,tf.matrix_diag(ds-tf.matrix_diag_part(tf.matmul(U,D,transpose_a=True)))), V,transpose_b=True)\
         + tf.matmul(2*mmul([U, S, msym(KT*tf.matmul(V,dV-mmul([tf.matmul(V,D,transpose_b=True),U,S]),transpose_a=True))]), V,transpose_b=True)
         
    if transpose:
        grad = tf.matrix_transpose(grad)
    
    return [crop_removenan(grad)]
