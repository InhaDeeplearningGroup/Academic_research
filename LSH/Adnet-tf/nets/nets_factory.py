import functools

import tensorflow as tf

from nets import Adnet
from nets import Adnet_small
from nets import Adnet_std
slim = tf.contrib.slim

networks_map   = {
                'Adnet':Adnet.Adnet,
                'Adnet_small':Adnet_small.Adnet_small,
                'Adnet_std':Adnet_std.Adnet_std,
                 }

arg_scopes_map = {
                  'Adnet':Adnet.Adnet_arg_scope,
                  'Adnet_small':Adnet_small.Adnet_small_arg_scope,
                  'Adnet_std':Adnet_std.Adnet_std_arg_scope,
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

