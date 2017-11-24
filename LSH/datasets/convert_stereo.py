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
    import re
    from random import shuffle
    dataset_type = 'png'
    
    # Load data & make feeder    
    dataset = open('/home/dmsl/Documents/data/kitti/training/set.txt')
    data = [None]*200
    seed = range(200)
    for i in xrange(200):
        name = re.findall('\S+',dataset.readline())[0]
        data[i] = cv2.imread('/home/dmsl/Documents/data/kitti/training/disp_occ_1/%s'%name,0)
        name = re.findall('\S+',dataset.readline())[0]
    height = 370
    width = 1200
    pixels = [[] for i in xrange((height-10)*(width-10-255))]
    for i in xrange(height-10):
        for j in xrange(width-10-255):
            y = i + 5
            x = j + 5 + 255            
            pixels[i*(width-10-255)+j] = [y,x]
    sets = [[] for i in range(256)]
    sys.stdout.write('\r\ncapture feature with original image')
    sys.stdout.flush()
    temp = list(zip(data, seed))
    random.shuffle(temp)
    data, seed = zip(*temp)
    for step in xrange(200):
        dis = data[step]
        shuffle(pixels)
        for i in xrange(len(pixels)):
            y,x = pixels[i]
            disparity = dis[y,x]
            if disparity != 0 and len(sets[disparity]) < 50000:
                sets[disparity].append([seed[step],y,x,disparity])
    sys.stdout.write('    done')
    sys.stdout.flush()
    for n in range(len(sets)):
        if len(sets[n]) > 1000:
            seed = [i for i in range(len(sets[n]))]
            sets[n] = np.array(sets[n])[np.random.choice(seed,1000,False)].tolist()
            
            
    croped = []
    for n in xrange(10):
        disp = sio.loadmat('/home/dmsl/Documents/data/middle/disp%d.mat'%n) ['img']
        disp = disp.astype(int)
        for i in range(1000):
            disp[np.where(i-disp[:,i]-20<0),i] = 0
        y,x = np.where(disp[20:-20,20:-20] != 0)
        d = disp[np.where(disp[20:-20,20:-20] != 0)]
        y += 20
        x += 20
        name = np.array([200+n]*len(x))
        
        croped += np.transpose(np.vstack((name,y,x,d))).tolist()
    
    seed = [i for i in range(len(croped))]
    croped = np.array(croped)[np.random.choice(seed,256*2000,False)].tolist()
    for n in xrange(256):
        croped += sets[n]    
    random.shuffle(croped)
    dataset_length = len(croped)
    training_filename = _get_output_filename(dataset_dir, '_train')
    
    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(training_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        with tf.Graph().as_default():
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_image = tf.image.encode_png(image_placeholder)
            l  = [None]*210
            l2  = [None]*210
            r  = [None]*210
            sys.stdout.write('\n\rimageconvering')
            sys.stdout.flush()
            for i in xrange(200):
                l[i] = cv2.imread('/home/dmsl/Documents/data/kitti/left_denoise/left%d.bmp'%(i+1))
                l2[i] = cv2.GaussianBlur(l[i],(5,5),0)
                r[i] = cv2.imread('/home/dmsl/Documents/data/kitti/right_denoise/right%d.bmp'%(i+1))
            for i in xrange(10):
                l[200+i] = cv2.imread('/home/dmsl/Documents/data/middle/left%d.png'%i)
                l2[200+i] = cv2.GaussianBlur(l[200+i],(5,5),0)
                r[200+i] = cv2.imread('/home/dmsl/Documents/data/middle/right%d.png'%i)
            sys.stdout.write('\n\rdone')
            sys.stdout.flush()
                
            with tf.Session('') as sess:
                count = 0
                for i in xrange(dataset_length):
                    name, y,x,label = croped[i]
                    left = l[name][y-5:y+6,x-5:x+6,:]
                    if np.sum(abs(left.astype(int) - l2[name][y-5:y+6,x-5:x+6,:].astype(int))) > 400:
                        n = label
                        while n == label:
                            n = random.randrange(5,11)
                        positive = r[name][y-5:y+6,x-label-5:x-label+6,:]
                        negative = r[name][y-5:y+6,x-label+n-5:x-label+n+6,:]
                        image = np.hstack((left,positive))
                        image_string = sess.run(encoded_image,
                                              feed_dict={image_placeholder: image})
                        example = image_to_tfexample(image_string, dataset_type, 1, 11, 22)
                        tfrecord_writer.write(example.SerializeToString())
                        
                        image = np.hstack((left,negative))
                        image_string = sess.run(encoded_image,
                                              feed_dict={image_placeholder: image})
                        example = image_to_tfexample(image_string, dataset_type, 0, 11, 22)
                        tfrecord_writer.write(example.SerializeToString())
                        count += 1
                        sys.stdout.write('\r>>Reading pairs : %d/%d saved pairs : %d height:%d width:%d' 
                                         % (i , dataset_length, count,  image.shape[0], image.shape[1]))
                        sys.stdout.flush()
                        if image.shape[1] != 22 or image.shape[0] != 11:
                            sys.stdout.write('\n\r>> %d  %d' 
                                         % (image.shape[0],image.shape[1]))
                            sys.stdout.flush()
                            break
    
    print('\nFinished converting the dataset!')
