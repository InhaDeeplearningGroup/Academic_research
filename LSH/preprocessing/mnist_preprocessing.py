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
        #image = tf.reshape(image,[28,28,1])
#        image = tf.pad(image, [[3,3],[3,3],[0,0]], mode='CONSTANT')
#        image = tf.random_crop(image,[29,29,1])
        return image
