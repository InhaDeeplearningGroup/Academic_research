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
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import stereo
from datasets import cd
from datasets import nerve
from datasets import mnist
from datasets import cifar10
from datasets import cifar100
from datasets import imgnet64
from datasets import OTB100_ori

datasets_map = {
    'stereo':stereo,
    'cd':cd,
    'nerve':nerve,
    'mnist':mnist,
    'cifar10':cifar10,
    'cifar100':cifar100,
    'imgnet64':imgnet64,
    'OTB100_ori':OTB100_ori,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name].get_split(split_name, dataset_dir,
                                        file_pattern, reader)
