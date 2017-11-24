from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os, glob
import sys, cv2, re
import numpy as np
from six.moves import xrange
from random import shuffle
import scipy.io as sio
LABELS_FILENAME = 'labels.txt'

def _get_output_filename(dataset_dir, split_name):
    return '%s/OTB100%s.tfrecord' % (dataset_dir, split_name)
  
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
    with tf.device('/cpu:0'):
        test_dataset = []
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_png(image_placeholder)
        with tf.Session('',config=tf.ConfigProto(log_device_placement=True)) as sess:
            dataset_len = 0
            training_len = 0
            test_len = 0
            imgset = {}
            with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
                dataset = []
                labels = np.zeros(23)
                video = glob.glob(os.path.join('/home/dmsl/Documents/data/OTB100', '*'))
                for vd in range(100):
                    ground_truth = open('%s/groundtruth_rect.txt'%video[vd])
                    name = video[vd][33:]
                    sys.stdout.write('\n\r>> Reading video_num%d %s'
                                     % (vd, name))
                    video_img_paths = glob.glob(os.path.join('%s/img'%video[vd],'*.jpg'))
                    video_img_paths.sort()
                    imgset['%s'%name] = []
                    pos =   sio.loadmat('/home/dmsl/Documents/data/OTB100/ann/%s_pos.mat'%name,squeeze_me=True)['pos']
                    label = sio.loadmat('/home/dmsl/Documents/data/OTB100/ann/%s_label.mat'%name,squeeze_me=True)['label']
                    neg =   sio.loadmat('/home/dmsl/Documents/data/OTB100/ann/%s_neg.mat'%name,squeeze_me=True)['neg']
                    for f, fr in enumerate(video_img_paths):
                        frame = cv2.imread(fr)
                        gt = ground_truth.readline()
                        gt = re.findall('\d+',gt)
                        for i,b in enumerate(gt):
                            gt[i] = int(b)
                        h,w,d = frame.shape
                        frame = cv2.resize(frame,(w*64//gt[2],h*64//gt[3]))
                        h,w,d = frame.shape
                        f_pos = pos[:,:,f]
                        f_label = label[:,f]-1
                        f_neg = neg[:,:,f]
                        
                        centers = boxes_(:,1:2) + 0.5 * boxes_(:,3:4);
                        wh = boxes_(:,3:4) * 1.4;
                        boxes_ = [centers, centers] + [-0.5 * wh, +0.5 * wh];
                        boxes_(:,1) = max(1, boxes_(:,1));
                        boxes_(:,2) = max(1, boxes_(:,2));
                        boxes_(:,3) = min(size(im,2), boxes_(:,3));
                        boxes_(:,4) = min(size(im,1), boxes_(:,4));
                        
                        
                        
                        f_pos[:,:2] = np.where(f_pos[:,:2]<0,0,f_pos[:,:2])
                        f_neg[:,:2] = np.where(f_neg[:,:2]<0,0,f_neg[:,:2])
                        f_pos[:,2] = np.where(f_pos[:,0]+f_pos[:,2]>w,w-f_pos[:,0],f_pos[:,2])
                        f_neg[:,2] = np.where(f_neg[:,0]+f_neg[:,2]>w,w-f_neg[:,0],f_neg[:,2])
                        f_pos[:,3] = np.where(f_pos[:,1]+f_pos[:,3]>h,h-f_pos[:,1],f_pos[:,3])
                        f_neg[:,3] = np.where(f_neg[:,1]+f_neg[:,3]>h,h-f_neg[:,1],f_neg[:,3])
                                                
                        x1 =min(np.min(f_pos[:,0]),np.min(f_neg[:,0]))
                        x2 =max(np.max(f_pos[:,0]+f_pos[:,2]),np.max(f_neg[:,0]+f_neg[:,2]))
                        y1 =min(np.min(f_pos[:,1]),np.min(f_neg[:,1]))
                        y2 =max(np.max(f_pos[:,1]+f_pos[:,3]),np.max(f_neg[:,1]+f_neg[:,3]))
                        
                        for i in range(f_pos.shape[2]):
                            dataset.append(['%s'%name, f, f_pos[i], f_label[i]*2 + 1])
                            labels[f_label[i]] +=1
                        for i in range(f_neg.shape[2]):
                            dataset.append(['%s'%name, f, f_neg[i], 0])
                                
                        cropped = frame[y1:y2, x1:x2]
                        imgset['%s'%video[vd][33:]].append(cropped)
                            
                sys.stdout.write('   labels %s'%labels)
                        
                shuffle(dataset)
                dataset_len += len(dataset)
                sys.stdout.write('\n\r')
                
                for i, data in enumerate(dataset):
                    frame = imgset[data[0]][data[1]]
                    nst = data[2]
                    image = frame[nst[1]:nst[1]+nst[3],
                                  nst[0]:nst[0]+nst[2]]
                    image = cv2.resize(image,(64,64),cv2.INTER_LANCZOS4)
                    label = data[3]
                    image_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: image})
                    example = image_to_tfexample(image_string, str.encode(dataset_type), int(label), 64, 64)
                    tfrecord_writer.write(example.SerializeToString())
                    training_len += 1
                    sys.stdout.write('\r>> Reading dataset images %d/%d training : %d test : %d '
                                     % (dataset_len , i, training_len, test_len))
                    if i+1 == int(len(dataset)*0.9):
                        test_dataset = test_dataset + dataset[int(len(dataset)*0.9):]
                        break
                        
            with tf.python_io.TFRecordWriter(test_filename) as tfrecord_writer:
                for i, data in enumerate(test_dataset):
                    frame = imgset[data[0]][data[1]]
                    nst = data[2]
                    image = frame[nst[1]:nst[1]+nst[3],
                                  nst[0]:nst[0]+nst[2]]
                    image = cv2.resize(image,(64,64),cv2.INTER_LANCZOS4)
                    label = data[3]
                    image_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: image})
                    example = image_to_tfexample(image_string, str.encode(dataset_type), int(label), 112, 112)
                    tfrecord_writer.write(example.SerializeToString())
                    test_len += 1
                    sys.stdout.write('\r>> Reading dataset images %d/%d training : %d test : %d '
                                 % (dataset_len , i, training_len, test_len))
                    
                            
        print('\nFinished converting the dataset! %d'%dataset_len)
