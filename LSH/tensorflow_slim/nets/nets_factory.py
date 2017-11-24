import functools

import tensorflow as tf

from nets import vgg16
from nets import vgg16_small
from nets import vgg16_std
#from nets import vgg16_std2

from nets import ECCV
from nets import ECCV_small
#from nets import alex_std

slim = tf.contrib.slim

networks_map   = {
                'vgg16':vgg16.vgg16,
                'vgg16_small':vgg16_small.vgg16_small,
                'vgg16_std':vgg16_std.vgg16_std,
#                'vgg16_std2':vgg16_std2.vgg16_std,
                'ECCV':ECCV.ECCV,
                'ECCV_small':ECCV_small.ECCV_small,
#                'alex_std':alex_std.alex_std,
                 }

arg_scopes_map = {
                  'vgg16':vgg16.vgg16_arg_scope,
                  'vgg16_small':vgg16_small.vgg16_small_arg_scope,
                  'vgg16_std':vgg16_std.vgg16_std_arg_scope,
#                  'vgg16_std2':vgg16_std2.vgg16_std_arg_scope,
                  'ECCV':ECCV.ECCV_arg_scope,
                  'ECCV_small':ECCV_small.ECCV_small_arg_scope,
#                  'alex_std':alex_std.alex_std_arg_scope,
                 }


def get_network_fn(name, weight_decay=5e-4):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        logits, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
    
  arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
  func = networks_map[name]
  @functools.wraps(func)
  def network_fn(images, is_training, lr = None, val = False):
    with slim.arg_scope(arg_scope):
      return func(images, is_training=is_training, lr = lr, val = val)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn

