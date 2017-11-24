from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2
import os
import sys
import random
import numpy as np
import scipy.io as sio
from six.moves import xrange

LABELS_FILENAME = 'labels.txt'

def _get_output_filename(dataset_dir, split_name):
    return '%s/stereo%s.tfrecord' % (dataset_dir, split_name)
  
def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
  
def image_to_tfexample(image_data, image_format, class_id, height, width):
    return tf.train.Example(features=tf.train.Features(feature={
                                                                'image/encoded': bytes_feature(image_data),
                                                                'image/format ': bytes_feature(str.encode(image_format)),
                                                                'image/class/label': int64_feature(class_id),
                                                                'image/height': int64_feature(height),
                                                                'image/width': int64_feature(width),}))
  
def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))
      
### main
def run(dataset_dir):
    from random import shuffle
    dataset_type = 'png'
    
    # Load data & make feeder    
    dataset_length = 0
    for n in xrange(10):
        depth = sio.loadmat('/home/dmsl/Documents/data/middle/fisheye/depth%d.mat'%n) ['dist_map']
        dataset_length += len(np.where(depth[:,:,0]*depth[:,:,1])[0])
    
    training_filename = _get_output_filename(dataset_dir, '_train')
    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(training_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    
    crop = []
    left = []
    right = []
    for n in xrange(2):
        left.append(cv2.imread('/home/dmsl/Documents/data/middle/fisheye/left%d.bmp'%n))
        right.append(cv2.imread('/home/dmsl/Documents/data/middle/fisheye/right%d.bmp'%n))
        depth = sio.loadmat('/home/dmsl/Documents/data/middle/fisheye/depth%d.mat'%n) ['dist_map'].astype(int)
        left_blur = cv2.GaussianBlur(left[n],(5,5),0)
        right_blur = cv2.GaussianBlur(right[n],(5,5),0)
        for i in range(1000,1200):
            for j in range(1000,1200):
#        for i in range(5,left[n].shape[0]-6):
#            for j in range(5,left[n].shape[0]-6):
                if depth[i,j,0]*depth[i,j,1] > 0:
                    ii = depth[i,j,0]
                    jj = depth[i,j,1]
                    left_patch = left[n][i-5:i+6,j-5:j+6]
                    right_patch = right[n][ii-5:ii+6,jj-5:jj+6]
                    if (np.sum(abs(left_patch - left_blur[i-5:i+6,j-5:j+6] )) > 100 ) & (np.sum(abs(right_patch - right_blur[ii-5:ii+6,jj-5:jj+6] )) > 100):
                        ni = ii
                        nj = jj
                        while (abs(ni - ii) < 5) | (abs(nj - jj) < 5):
                            ni = random.randrange(ii-11,ii+11)
                            nj = random.randrange(jj-11,jj+11)
                        crop.append([n, i,j, ii,jj, 1])
                        crop.append([n, i,j, ni,nj, 0])
    shuffle(crop)                                    
    
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        with tf.Graph().as_default():
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_image = tf.image.encode_png(image_placeholder)
            sys.stdout.write('\n\rimageconvering')
            sys.stdout.flush()
            with tf.Session('') as sess:
                with tf.device("/cpu:0"):
                    count = 0
                    for n in xrange(len(crop)):
                        index, i, j, ii, jj, label = crop[n]
                        left_patch = left[index][i-5:i+6,j-5:j+6]
                        right_patch = right[index][ii-5:ii+6,jj-5:jj+6]
                        image = np.hstack((left_patch,right_patch))
                        image_string = sess.run(encoded_image,
                                              feed_dict={image_placeholder: image})
                        example = image_to_tfexample(image_string, dataset_type, label, 11, 22)
                        tfrecord_writer.write(example.SerializeToString())
                        
                        count += 1
                        sys.stdout.write('\r>>Reading pairs : %d/%d saved pairs : %d height:%d width:%d' 
                                         % (n , len(crop), count,  image.shape[0], image.shape[1]))
                        sys.stdout.flush()
                        if image.shape[1] != 22 or image.shape[0] != 11:
                            sys.stdout.write('\n\r>> %d  %d' 
                                         % (image.shape[0],image.shape[1]))
                            sys.stdout.flush()
                            break

    print('\nFinished converting the dataset!')
