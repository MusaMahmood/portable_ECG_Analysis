# MUSA MAHMOOD - Copyright 2018
# Python 3.6.3
# TF 1.8.0

# Imports:
import os
import numpy as np
import tf_shared as tfs
import tensorflow as tf

from scipy.io import savemat
from keras import regularizers
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Reshape
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.utils.generic_utils import Progbar
from keras.optimizers import Adam
from keras.backend import tensorflow_backend as tf_backend
from keras.layers import Bidirectional, CuDNNLSTM, Conv1D, LeakyReLU, Flatten, Activation, Input

# Setup: TODO:
