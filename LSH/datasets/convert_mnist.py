from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import tensorflow as tf
import os
import sys
import numpy as np
from six.moves import xrange
LABELS_FILENAME = 'labels.txt'

def _get_output_filename(dataset_dir, split_name):
    return '%s/mnist%s.tfrecord' % (dataset_dir, split_name)
  
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
    dataset_len =10000
    dataset_type = 'png'
    data_filename = os.path.join('/home/dmsl/Documents/data/mnist/', 't10k-images-idx3-ubyte.gz')
    label_filename = os.path.join('/home/dmsl/Documents/data/mnist/', 't10k-labels-idx1-ubyte.gz')
    with gzip.open(data_filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(
            28 * 28 * dataset_len * 1)
        data = np.frombuffer(buf, dtype=np.uint8)
        images = data.reshape(dataset_len, 28, 28, 1)
    
    with gzip.open(label_filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * dataset_len)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    
    
    training_filename = _get_output_filename(dataset_dir, '_test')
    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(training_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    with tf.device('/cpu:0'):
        with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_image = tf.image.encode_png(image_placeholder)
            with tf.Session('',config=tf.ConfigProto(log_device_placement=True)) as sess:
                for n in xrange(dataset_len):
                    image = images[n]
                    label = labels[n]
                    image_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: image})
                    example = image_to_tfexample(image_string, str.encode(dataset_type), int(label), 28, 28)
                    tfrecord_writer.write(example.SerializeToString())
                    sys.stdout.write('\r>> Reading dataset images %d/%d' 
                                     % (n , dataset_len))
                data = []
                    
                        
        print('\nFinished converting the dataset! %d'%dataset_len)
