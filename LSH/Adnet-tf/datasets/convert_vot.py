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
    training_filename = _get_output_filename(dataset_dir, '_train')
    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(training_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        with tf.Graph().as_default():
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_image = tf.image.encode_png(image_placeholder)
            with tf.Session('') as sess:
                dataset = sio.loadmat('/home/dmsl/Documents/train_db.mat')['train_db']
                data = []
                num_vd = dataset.shape[1]
                count = 0
                for v in range(dataset.shape[1]):
            #for v in range(1):
                    print ('video : %d/%d'%(v+1, dataset.shape[1]))
                    video = dataset[0,v][0]
                    for i in range(video.shape[0]):
                        img_path = video[i][0][0]
                        img = cv2.imread(img_path)
                        
                        bbox = (video[i][2]-1).astype(int)
                        bbox[:,:2] = np.where(bbox[:,(1,0)]<0,0,bbox[:,(1,0)])
                        bbox[:,2] = np.where(bbox[:,2]>img.shape[0],img.shape[0],bbox[:,2])
                        bbox[:,3] = np.where(bbox[:,3]>img.shape[1],img.shape[1],bbox[:,3])
                        bbox[:,2:] = bbox[:,(3,2)]
                        
                        action_lable = np.argmax(np.hstack((video[i][3],np.zeros([11,50]))),0)
                        score_lable = video[i][4]-1
                        
                        for bb in range(bbox.shape[0]):
                            crop_img = img[bbox[bb,0]:bbox[bb,0]+bbox[bb,2],
                                           bbox[bb,1]:bbox[bb,1]+bbox[bb,3]]
                            crop_img = cv2.resize(crop_img,(112,112),cv2.INTER_CUBIC)
                            data.append([crop_img, action_lable[bb], score_lable[bb,0], v])
                    
                    if (((v+1)%5) == 0)&(v != 0):
                        shuffle(data)
                        dataset_length = len(data)//2
                        pre = count
                        for i in xrange(dataset_length):
                            image, al, sl, v = data[i]
                            if sl == 1:
                                al = 11
                            label = al*num_vd + v
                            image_string = sess.run(encoded_image,
                                                  feed_dict={image_placeholder: image})
                            example = image_to_tfexample(image_string, dataset_type, label, 112, 112)
                            tfrecord_writer.write(example.SerializeToString())
                            
                            count += 1
                            sys.stdout.write('\r>>saved img : %d/%d ' 
                                             % (dataset_length+pre, count))
                            sys.stdout.flush()
                        sys.stdout.write('\r\n ')
                        sys.stdout.flush()
                        data = data[dataset_length:]
                    if v+1 == num_vd:
                        shuffle(data)
                        dataset_length = len(data)
                        pre = count
                        for i in xrange(dataset_length):
                            image, al, sl, v = data[i]
                            if sl == 1:
                                al = 11
                            label = al*num_vd + v
                            image_string = sess.run(encoded_image,
                                                  feed_dict={image_placeholder: image})
                            example = image_to_tfexample(image_string, dataset_type, label, 112, 112)
                            tfrecord_writer.write(example.SerializeToString())
                            
                            count += 1
                            sys.stdout.write('\r>>saved img : %d/%d ' 
                                             % (dataset_length+pre, count))
                            sys.stdout.flush()
                        sys.stdout.write('\r\n ')
                        sys.stdout.flush()
    print('\nFinished converting the dataset!')
