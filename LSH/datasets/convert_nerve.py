from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2
import os
import sys
import numpy as np
from six.moves import xrange

def _get_output_filename(dataset_dir, split_name):
    return '%s/nerve%s.tfrecord' % (dataset_dir, split_name)
  
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
def run(dataset_dir):
    import glob
    from random import shuffle
    dataset_type = 'png'

    training_filename = _get_output_filename(dataset_dir, '_train')
    
    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(training_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        with tf.Graph().as_default():
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_image = tf.image.encode_png(image_placeholder)
            dataset_length = 0
            data = []
            with tf.Session('') as sess:
                train_paths = glob.glob(os.path.join('/home/dmsl/Documents/data/nerve_train/imag', '*.tif'))
                for i in xrange(5635):
                    img = train_paths[i]
                    mask = img[:38]+'mask'+img[42:]
                    image = cv2.imread(img)
                    label = cv2.imread(mask)
                
                    index = np.where(label>0)
                    if index[0].shape[0]>0:
                        box_pts = [np.min(index[0]),np.max(index[0]),np.min(index[1]),np.max(index[1])]
                        if (box_pts[0]+box_pts[1])/2-178 < 0:                        
                            box_pts[0] = max(0,int((box_pts[0]+box_pts[1])/2)-178)
                            box_pts[1] = box_pts[0]+356
                            
                        else:
                            box_pts[1] = min(image.shape[0],int((box_pts[0]+box_pts[1])/2)+178)
                            box_pts[0] = box_pts[1]-356
                            
                        if (box_pts[2]+box_pts[3])/2-178 < 0:                        
                            box_pts[2] = max(0,int((box_pts[2]+box_pts[3])/2)-178)
                            box_pts[3] = box_pts[2]+356
                            
                        else:
                            box_pts[3] = min(image.shape[1],int((box_pts[2]+box_pts[3])/2)+178)
                            box_pts[2] = box_pts[3]-356
                        
                        
                        box_img = image[box_pts[0]:box_pts[1],box_pts[2]:box_pts[3]]
                        box_mask = label[box_pts[0]:box_pts[1],box_pts[2]:box_pts[3]]
                        data.append(np.hstack((box_img,box_mask)))
                        
                    
                shuffle(data)
                for n in xrange(len(data)):
                    image = data[n]
                    image_string = sess.run(encoded_image, feed_dict={image_placeholder: image})
                    example = image_to_tfexample(image_string, str.encode(dataset_type), int(0), 356, 712)
                    tfrecord_writer.write(example.SerializeToString())
                    sys.stdout.write('\r>> Reading dataset images %d/%d' 
                                     % (n , dataset_length))
                    if ((image.shape[0] != 356) or (image.shape[1] != 712)):
                        print (image.shape)
                        break
                    
    print('\nFinished converting the dataset! %d'%dataset_length)
