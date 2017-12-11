import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

# %matplotlib inline

# 1 - Transfer Learning
# pretrained VGG-16 CNN with 16 layer
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
print(model)

# 2- Neural Style Transfer
# 1）Computing the content cost

def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(tf.transpose(a_C), shape=(n_W * n_H, n_C))
    a_G_unrolled = tf.reshape(tf.transpose(a_G), shape=(n_W * n_H, n_C))

    J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content