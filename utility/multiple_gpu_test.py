__author__ = "KimDaeHa"
# -*- coding: utf-8 -*-
# %%
import argparse
app = argparse.ArgumentParser()
app.add_argument("-o", "--output", required=True,
                 help="path to output plot")
app.add_argument("-g", "--gpus", type=int, default=1,
                 help="# of GPUs to use for training")
args = vars(app.parse_args())

G = args["gpus"]

# %%
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(G)
set_session(tf.Session(config=config))

print()
print("[INFO] training with {} GPUs ...".format(G))
print()
