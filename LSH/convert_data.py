# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts a particular dataset.

Usage:
```shell

$ python download_and_convert_data.py \
    --dataset_name=mnist \
    --dataset_dir=/tmp/mnist

$ python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=/tmp/cifar10

$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir=/tmp/flowers
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import convert_sr
from datasets import convert_cd
from datasets import convert_stereo2
from datasets import convert_mnist
from datasets import convert_imgnet64
#from datasets import convert_OTB100
from datasets import convert_cifar10
from datasets import convert_cifar100

import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataset_name',
    None,
    'The name of the dataset to convert, one of "cifar10", "flowers", "mnist".')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')
def main(_):
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if not os.path.exists(FLAGS.dataset_dir):
    os.makedirs(FLAGS.dataset_dir)
  if FLAGS.dataset_name == 'sr':
    convert_sr.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'stereo2':
    convert_stereo2.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'mnist':
      convert_mnist.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'imgnet64':
      convert_imgnet64.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'cd':
      convert_cd.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'OTB100':
      convert_OTB100.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'cifar10':
      convert_cifar10.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'cifar100':
      convert_cifar100.run(FLAGS.dataset_dir)
  
      
if __name__ == '__main__':
  tf.app.run()

