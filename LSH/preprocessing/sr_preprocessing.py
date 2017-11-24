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
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        image = tf.image.random_flip_up_down(image)
        
        hr = tf.slice(image,[0,21,0],[21,21,3])
        lr = tf.slice(image,[0,0,0],[21,21,3])
        
        tf.summary.image('lr_image', tf.expand_dims(lr, 0))
        tf.summary.image('hr_image', tf.expand_dims(hr, 0))

        return [lr, hr]
