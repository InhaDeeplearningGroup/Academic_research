# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities for preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
def random_rotate(x, deg):
    theta = np.pi/180 * tf.random_uniform([1],-deg, deg)
    x = tf.contrib.image.rotate(x, theta, interpolation="NEAREST")
    return x
def preprocess_image(image, is_training):
    with tf.variable_scope('preprocessing'):
        image = tf.to_float(image)
        tf.summary.image('image', tf.expand_dims(image, 0))
#        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_flip_left_right(image)
        image = random_rotate(image, 15)
#        image = tf.contrib.keras.preprocessing.image.random_rotation(image, 15, 0,1,2)
#        image = tf.image.random_flip_up_down(image)
        image = tf.pad(image, [[4,4],[4,4],[0,0]], 'SYMMETRIC')
        image = tf.random_crop(image,[32,32,3])
        tf.summary.image('aug_image', tf.expand_dims(image, 0))
        return image
