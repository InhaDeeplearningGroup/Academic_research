from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import tensorflow as tf
import os, glob
import sys
import cv2
import numpy as np
from random import shuffle
from six.moves import xrange
LABELS_FILENAME = 'labels.txt'

def _get_output_filename(dataset_dir, split_name):
    return '%s/cd%s.tfrecord' % (dataset_dir, split_name)
  
def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
  
def image_to_tfexample(image_data, image_format, class_id, height, width):
    return tf.train.Example(features=tf.train.Features(feature={
                                                                'image/encoded': bytes_feature(image_data),
                                                                'image/format ': bytes_feature(image_format),
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
    dataset_type = 'png'
    imagesets = glob.glob(os.path.join('/home/dmsl/Documents/data/cd/train', '*'))
    imagesets.sort()
    training_filename = _get_output_filename(dataset_dir, '_train')
    test_filename = _get_output_filename(dataset_dir, '_test')
    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(training_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    with tf.device('/cpu:0'):
        with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_image = tf.image.encode_png(image_placeholder)
            with tf.Session('',config=tf.ConfigProto(log_device_placement=True)) as sess:
                seed = np.array(range(12500))
                shuffle(seed)
                for n in xrange(int(len(seed)*0.9)):
                    cat = cv2.imread('%s'%imagesets[seed[n]])
                    dog = cv2.imread('%s'%imagesets[seed[n]+12500])
                    cat = cv2.resize(cat,(32,32),cv2.INTER_LANCZOS4)
                    dog = cv2.resize(dog,(32,32),cv2.INTER_LANCZOS4)
                    image_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: cat})
                    example = image_to_tfexample(image_string, str.encode(dataset_type), int(0), 32, 32)
                    tfrecord_writer.write(example.SerializeToString())
                    image_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: dog})
                    example = image_to_tfexample(image_string, str.encode(dataset_type), int(1), 32, 32)
                    tfrecord_writer.write(example.SerializeToString())
                    sys.stdout.write('\r>> Reading dataset images %d/%d' 
                                     % (n , len(imagesets)))
        with tf.python_io.TFRecordWriter(test_filename) as tfrecord_writer:
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_image = tf.image.encode_png(image_placeholder)
            with tf.Session('',config=tf.ConfigProto(log_device_placement=True)) as sess:
                for n in xrange(int(len(seed)*0.9),len(seed)):
                    cat = cv2.imread('%s'%imagesets[seed[n]])
                    dog = cv2.imread('%s'%imagesets[seed[n]+12500])
                    cat = cv2.resize(cat,(32,32),cv2.INTER_LANCZOS4)
                    dog = cv2.resize(dog,(32,32),cv2.INTER_LANCZOS4)
                    image_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: cat})
                    example = image_to_tfexample(image_string, str.encode(dataset_type), int(0), 32, 32)
                    tfrecord_writer.write(example.SerializeToString())
                    image_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: dog})
                    example = image_to_tfexample(image_string, str.encode(dataset_type), int(1), 32, 32)
                    tfrecord_writer.write(example.SerializeToString())
                    sys.stdout.write('\r>> Reading dataset images %d/%d' 
                                     % (n , len(imagesets)))
        print('\nFinished converting the dataset! %d'%len(imagesets))
        print('\nFinished converting the dataset! %d'%len(imagesets))
