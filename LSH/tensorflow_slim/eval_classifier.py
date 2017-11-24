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

with tf.Graph().as_default():
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
        for i in range(0,len(optim_vars)):
            p = sess.run(optim_vars[i])
            layer[name[i]] = p
#                
        t = time.time()
        predict = np.array([0,0], dtype = float)
        sv.start_queue_runners(sess)
        l = 0
        for i in range(50):
            p1, l1, task = sess.run([task1, lb, training_accuracy1])
            predict += task
            correct += np.sum(np.where(p1 == l1, 1,0))
        print (time.time()-t)
        
    accuracy = correct/(dataset.num_samples)
    print (accuracy)
    
    sess.close()
sio.savemat('/home/dmsl/nas/backup1/personal_lsh/training/cifar100/vgg13_noaug.mat',layer)
