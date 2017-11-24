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
slim = tf.contrib.slim


def preprocess_image(image, is_training):
    with tf.variable_scope('preprocessing'):
        image = tf.to_float(image)
        tf.summary.image('image', tf.expand_dims(image, 0))
        left_image = tf.slice(image,[0,0,0],[11,11,3])
        right_image = tf.slice(image,[0,11,0],[11,11,3])
        if is_training is True:
            with tf.variable_scope('distort_image'):
                right_image = tf.image.random_hue(right_image, max_delta=0.1)
                right_image = tf.image.random_saturation(right_image, lower=0.9, upper=1.1)
                right_image = tf.image.random_contrast(right_image, lower=0.9, upper=1.1)
                right_image = tf.image.random_brightness(right_image, max_delta=0.1)
        
        tf.summary.image('left_image', tf.expand_dims(left_image, 0))
        tf.summary.image('right_image', tf.expand_dims(right_image, 0))
        return left_image, right_image
