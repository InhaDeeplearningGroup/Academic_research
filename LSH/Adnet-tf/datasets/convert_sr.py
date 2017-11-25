import tensorflow as tf
import cv2
import os
import sys
import numpy as np
import glob
from six.moves import xrange
from random import shuffle
LABELS_FILENAME = 'labels.txt'

def _get_output_filename(dataset_dir, split_name):
    return '%s/sr%s.tfrecord' % (dataset_dir, split_name)
  
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

    training_filename = _get_output_filename(dataset_dir, '_train')
    test_filename = _get_output_filename(dataset_dir, '_test')
    
    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(training_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    image_placeholder = tf.placeholder(dtype=tf.uint8)
    encoded_image = tf.image.encode_png(image_placeholder)
    with tf.Session('') as sess:
        dataset_length = 0
        data = []
        imgs_paths = glob.glob(os.path.join('/home/dmsl/Documents/data/Urban100', '*.jpg'))
#        for n in xrange(len(imgs_paths)):
        for n in xrange(5):
            print ('%d'%n)
            img = cv2.imread('%s'%imgs_paths[n])
            height, width,_ = img.shape
            #img = img[:,:,1]
            laplacian = abs(cv2.Laplacian(img[:,:,2],cv2.CV_64F))
            
            limg = cv2.resize(img,(width//2, height//2), cv2.INTER_CUBIC)
            limg = cv2.resize(img,(width, height), cv2.INTER_CUBIC)
            
            for i in xrange(height-25):
                for j in xrange(width-25):
                    if (np.max(laplacian[i+6:i+14,j+6:j+14]) > 200):
                        data.append([limg[i:i+21,j:j+21],img[i:i+21,j:j+21]])
            
        shuffle(data)
        with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
            
            for n in xrange(int(len(data)*0.9)):
                #image = np.stack((data[n],data[n],data[n]),2)
                image = data[n][0]
                mask = data[n][1]
                image = np.hstack((image,mask))
                
                if image.shape[0] != 21 or image.shape[1] != 42:
                        sys.stdout.write('\n\r>> %d  %d' 
                                     % (image.shape[0],image.shape[1]))
                        sys.stdout.flush()
                        break
                image_string = sess.run(encoded_image,
                                  feed_dict={image_placeholder: image})
                example = image_to_tfexample(image_string, str.encode(dataset_type), 0, 21,42)
                tfrecord_writer.write(example.SerializeToString())
                sys.stdout.write('\r>> Reading dataset images %d/%d' 
                                 % (n , len(data)))
                
        with tf.python_io.TFRecordWriter(test_filename) as tfrecord_writer:
            for n in xrange(int(len(data)*0.9),len(data)):
                #image = np.stack((data[n],data[n],data[n]),2)
                image = data[n][0]
                mask = data[n][1]
                image = np.hstack((image,mask))
                
                if image.shape[0] != 21 or image.shape[1] != 42:
                        sys.stdout.write('\n\r>> %d  %d' 
                                     % (image.shape[0],image.shape[1]))
                        sys.stdout.flush()
                        break
                image_string = sess.run(encoded_image,
                                  feed_dict={image_placeholder: image})
                example = image_to_tfexample(image_string, str.encode(dataset_type), 0, 21,42)
                tfrecord_writer.write(example.SerializeToString())
                sys.stdout.write('\r>> Reading dataset images %d/%d' 
                                 % (n , len(data)))
            
                    
    print('\nFinished converting the dataset! %d'%dataset_length)
