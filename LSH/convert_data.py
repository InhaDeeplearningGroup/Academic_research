import tensorflow as tf

from datasets import convert_mnist
from datasets import convert_cifar10
from datasets import convert_cifar100

import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_name',None,
                           'The name of the dataset to convert.')

tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The directory where the output TFRecords and temporary files are saved.')
def main(_):
    if not FLAGS.dataset_name:
        raise ValueError('You must supply the dataset name with --dataset_name')
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    if not os.path.exists(FLAGS.dataset_dir):
        os.makedirs(FLAGS.dataset_dir)
    elif FLAGS.dataset_name == 'mnist':
        convert_mnist.run(FLAGS.dataset_dir)
    elif FLAGS.dataset_name == 'cifar10':
        convert_cifar10.run(FLAGS.dataset_dir)
    elif FLAGS.dataset_name == 'cifar100':
        convert_cifar100.run(FLAGS.dataset_dir)
  
if __name__ == '__main__':
    tf.app.run()

