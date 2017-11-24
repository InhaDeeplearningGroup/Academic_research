import tensorflow as tf
slim = tf.contrib.slim
from datasets import dataset_factory
from nets import nets_factory
from tensorflow.python.training.saver import latest_checkpoint
from tensorflow.python.training.saver import Saver
from tensorflow.python.training       import supervisor
from tensorflow import Session
from tensorflow import ConfigProto

import time
import numpy as np
import scipy.io as sio
import cv2
#train_dir   =  '/home/dmsl/nas/share/personal_lsh/training/cifar100/vanila/vgg13'
train_dir   =  '/home/dmsl/Documents/tf/svd/teacher2'
dataset_dir = '/home/dmsl/Documents/data/tf/cifar100'
dataset_name = 'cifar10'
model_name   = 'vgg16'
batch_size = 200
tf.logging.set_verbosity(tf.logging.INFO)

def convolve_np(x,f):
    x_sz = list(x.shape)
    x = x.reshape(x_sz+[1])
    f_sz = list(f.shape)
    f = f.reshape(f_sz[:2]+[1]+f_sz[2:])
    output = np.zeros((x_sz[0]+f_sz[0]-1,x_sz[0]+f_sz[0]-1,x_sz[2],f_sz[3]))
    x = np.pad(x,(((f_sz[0]-1),(f_sz[0]-1)),((f_sz[0]-1),(f_sz[0]-1)),(0,0),(0,0),(0,0)),'constant')
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i,j] = np.sum(x[i:i+f_sz[0],j:j+f_sz[1]]*f,(0,1,3))
    return output

def pooling_np(x,f_sz):
    x_sz = list(x.shape)
    output = np.zeros((x_sz[0]+f_sz[0]-1,x_sz[0]+f_sz[0]-1,x_sz[2],x_sz[3]))
    x = np.pad(x,(((f_sz[0]-1),(f_sz[0]-1)),((f_sz[0]-1),(f_sz[0]-1)),(0,0),(0,0)),'constant')
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i,j] = np.mean(x[i:i+f_sz[0],j:j+f_sz[1]],(0,1))
    return output

def dist_conv_init(filters, shape=[1,1]):
    num_in = len(filters)
    
    f = filters[0]
    for n in range(1,num_in):
        if len(filters[n].shape)==4:
            f = convolve_np(f,np.float32(filters[n]))
        else:
            f = pooling_np(f,np.float32(filters[n]))
        
    output = np.zeros([shape[0],shape[1],f.shape[2],f.shape[3]],dtype=np.float32)
    for n in range(output.shape[-1]):
        output[:,:,:,n] = cv2.resize(f[:,:,:,n], (shape[0],shape[1]),interpolation=cv2.INTER_CUBIC)
        
    if shape[0] == 1:
        output = np.squeeze(output)
    return output

def resizing(f,shape=[1,1]):
#    output = np.zeros([shape[0],shape[1],f.shape[2],f.shape[3]],dtype=np.float32)
#    for n in range(output.shape[-1]):
#        output[:,:,:,n] = cv2.resize(f[:,:,:,n], (shape[0],shape[1]),interpolation=cv2.INTER_CUBIC)
#        
#    if shape[0] == 1:
#        output = np.squeeze(output)
    sz = f.shape
    output = f[sz[0]//2-shape[0]//2:sz[0]//2+shape[0]//2+1,sz[0]//2-shape[0]//2:sz[0]//2+shape[0]//2+1]
    return output

def dist_bias_init(filters):
    num_in = len(filters)
    f = filters[0]
    for n in range(1,num_in,2):
        f = convolve_np(f,np.float32(filters[n]))+np.float32(filters[n+1])
    f = np.sum(np.sum(f,axis=0,keepdims=True),axis=1,keepdims=True)
    f = np.squeeze(f)
    return f
def relu_map_convolve(x,y):
    c1 = np.maximum(convolve_np(x,y),0)
    c2 = np.maximum(convolve_np(x,-y),0)
    return c1-c2
    
with tf.Graph().as_default():
    ## Load Dataset
    dataset = dataset_factory.get_dataset(dataset_name, 'test', dataset_dir)
    with tf.device('/device:CPU:0'):
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                                  shuffle=False,
                                                                  num_readers = 1,
                                                                  common_queue_capacity=200 * batch_size,
                                                                  common_queue_min=100 * batch_size)
        images, labels = provider.get(['image', 'label'])
    
    images = tf.to_float(images)
    images = tf.concat([(tf.slice(images,[0,0,0],[32,32,1])-112.4776)/70.4587,
                            (tf.slice(images,[0,0,1],[32,32,1])-124.1058)/65.4312,
                            (tf.slice(images,[0,0,2],[32,32,1])-129.3773)/68.2094],2)
#    images = tf.image.resize_bicubic(tf.reshape(images,[1,32,32,3]),[112,112])
#    images = tf.squeeze(images)
    batch_images, batch_labels = tf.train.batch([images, labels],
                                            batch_size = batch_size,
                                            num_threads = 1,
                                            capacity = 200 * batch_size)
    
    batch_queue = slim.prefetch_queue.prefetch_queue([batch_images, batch_labels], capacity=50*batch_size)
    img, lb = batch_queue.dequeue()
    ## Load Model
    network_fn = nets_factory.get_network_fn(model_name)
    end_points = network_fn(img, is_training=False)
    print (end_points)
    task1 = tf.to_int32(tf.argmax(end_points['Logits'], 1))
    
    training_accuracy1 = slim.metrics.accuracy(task1, tf.to_int32(lb))
    
    variables_to_restore = slim.get_variables_to_restore()
    checkpoint_path = latest_checkpoint(train_dir)
    saver = Saver(variables_to_restore)
    config = ConfigProto()
    config.gpu_options.allow_growth=True
    sess = Session(config=config)
    sv = supervisor.Supervisor(logdir=checkpoint_path,
                               summary_op=None,
                               summary_writer=None,
                               global_step=None,
                               saver=None)
    correct = 0
    predict = 0
    with sv.managed_session(master='', start_standard_services=False, config=config) as sess:
        saver.restore(sess, checkpoint_path)
        optim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        layer = {}
        name = ['conv1w','conv1b',
                'conv2w','conv2b',
                'conv3w','conv3b',
                'conv4w','conv4b',
                'conv5w','conv5b',
                'conv6w','conv6b',
                'conv7w','conv7b',
                'conv8w','conv8b',
                'conv9w','conv9b',
                'conv10w','conv10b',
                'fc1w','fc1b','fc2w','fc2b','fc3w','fc3b']
#        print (optim_vars)
        for i in range(0,len(optim_vars)):
            p = sess.run(optim_vars[i])
#            if len(list(p.shape)) ==2:
#                p = p.reshape([1,1,p.shape[0],p.shape[1]])
#            if (len(list(p.shape)) ==1)&(name[i][:4]=='conv'):
#                p = p.reshape([1,1,1,p.shape[0]])
            layer[name[i]] = p
#                
        t = time.time()
        predict = np.array([0,0], dtype = float)
        sv.start_queue_runners(sess)
        l = 0
        for i in range(50):
#        for i in range(1):
            p1, l1, task = sess.run([task1, lb, training_accuracy1])
            predict += task
            correct += np.sum(np.where(p1 == l1, 1,0))
#            l += sess.run(end_points['m'])
        print (time.time()-t)
#        fs = sess.run(end_points['fs'])
#        ft = sess.run(end_points['ft'])
#        ms = sess.run(end_points['ms'])
#        mt = sess.run(end_points['mt'])
        
    accuracy = correct/(dataset.num_samples)
    print (accuracy)
    
    sess.close()
sio.savemat('/home/dmsl/nas/backup1/personal_lsh/training/cifar100/vgg13_noaug.mat',layer)
#small2 = sio.loadmat('/home/dmsl/nas/share/training/cifar100/vgg13to6.mat')
#small = {}
#small['w1'] = resizing(relu_map_convolve(layer['conv1w'],layer['conv2w']),[3,3])
#small['b1'] = resizing(relu_map_convolve(layer['conv1b'],layer['conv2w'])+layer['conv2b'])
#
#small['w2'] = resizing(relu_map_convolve(layer['conv3w'],layer['conv4w']),[3,3])
#small['b2'] = resizing(relu_map_convolve(layer['conv3b'],layer['conv4w'])+layer['conv4b'])
#
#small['w3'] = resizing(relu_map_convolve(relu_map_convolve(layer['conv5w'],layer['conv6w']),layer['conv7w']),[3,3])
#small['b3'] = resizing(relu_map_convolve(relu_map_convolve(layer['conv5b'],layer['conv6w'])+layer['conv6b'],layer['conv7w'])+layer['conv7b'])
#
#small['w4'] = resizing(relu_map_convolve(relu_map_convolve(layer['conv8w'],layer['conv9w']),layer['conv10w']),[3,3])
#small['b4'] = resizing(relu_map_convolve(relu_map_convolve(layer['conv8b'],layer['conv9w'])+layer['conv9b'],layer['conv10w'])+layer['conv10b'])
#
#small['w5'] = np.squeeze(np.maximum(np.dot(layer['fc1w'],layer['fc2w']),0)-np.maximum(np.dot(layer['fc1w'],-layer['fc2w']),0))
#small['b5'] = np.maximum(np.dot(layer['fc1b'],layer['fc2w']),0)-np.maximum(np.dot(layer['fc1b'],-layer['fc2w']),0)+layer['fc2b']
#
#small['w6'] = layer['fc3w']
#small['b6'] = layer['fc3b']
#sio.savemat('/home/dmsl/nas/share/personal_lsh/training/cifar100/vgg13.mat',layer)
