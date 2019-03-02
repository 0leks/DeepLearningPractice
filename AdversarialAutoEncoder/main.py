import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import time
import os

import data_utils

print('Tensorflow version:', tf.__version__)
assert tf.__version__ >= "2.0" # TensorFlow ≥ 2.0 required


train_dataset, validate_dataset = data_utils.getMNISTdatasets()

