import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg') # Allows generating plots without popup. Must call before importing pyplot.
import matplotlib.pyplot as plt

import utils.general
import utils.data
import utils.models

print('Loading fashion mnist data from file')
(xtrain, ytrain), (xvalidate, yvalidate) = utils.data.loadCSVImageDataset('../datasets/fashionmnist/fashion-mnist_train.csv')

plt.imshow(xtrain[0, :, :, 0],  cmap='Greys')
plt.colorbar()
plt.savefig('image0_.png')

#https://medium.com/@connectwithghosh/simple-autoencoder-example-using-tensorflow-in-python-on-the-fashion-mnist-dataset-eee63b8ed9f1
#https://blog.keras.io/building-autoencoders-in-keras.html
