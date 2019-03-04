import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

print('tensorflow version:', tf.__version__)

(train_images, train_labels), (XTest, YTest) = keras.datasets.fashion_mnist.load_data()

train_images = (train_images.astype('float32') - 127.5) / 127.5
XTest = (XTest.astype('float32') - 127.5) / 127.5

XTraining, XValidation, YTraining, YValidation = train_test_split(train_images,train_labels,stratify=train_labels,test_size=0.167)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('training set X shape:', XTraining.shape)
print('training set Y shape:', YTraining.shape)
print('validation set X shape:', XValidation.shape)
print('validation set Y shape:', YValidation.shape)
print('test set X shape:', XTest.shape)
print('test set Y shape:', YTest.shape)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


numEpochs = 100
trainLoss = [0] * numEpochs
validationLoss = [0] * numEpochs

for i in range(numEpochs):
  # no batches here, just doing entire dataset
  model.fit(XTraining, YTraining, epochs=1)

  train_loss, train_acc = model.evaluate(XTraining, YTraining)
  trainLoss[i] = train_loss

  validation_loss, validation_acc = model.evaluate(XValidation, YValidation)
  validationLoss[i] = validation_loss

#TODO add title and save figure as png instead of sidplaying plot
#TODO create local library for saving images, loading data
plt.figure()
plt.plot(trainLoss, label='Train Set')
plt.plot(validationLoss, label='Validation Set')
plt.xlabel('Epoch')
plt.ylabel('Loss (sparse_categorical_crossentropy')
plt.legend()
plt.show()
