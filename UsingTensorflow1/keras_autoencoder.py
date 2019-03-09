import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim

import matplotlib
matplotlib.use('Agg') # Allows generating plots without popup. Must call before importing pyplot.
import matplotlib.pyplot as plt

import utils.general
import utils.data
import utils.models

print('Loading fashion mnist data from file')
#(xtrain, ytrain), (xvalidate, yvalidate) = utils.data.loadCSVImageDataset('../datasets/fashionmnist/fashion-mnist_train.csv')
#(xtrain, ytrain), (xvalidate, yvalidate) = utils.data.getFashionMNISTdatasetClassification(validation_ratio=1/6)
(xtrain, ytrain), (xvalidate, yvalidate) = utils.data.getMNISTdatasetClassification(validation_ratio=1/6)

#https://blog.keras.io/building-autoencoders-in-keras.html

input_layer = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
net = input_layer
net = slim.flatten(net)
net = slim.fully_connected(net, 512, activation_fn=tf.nn.elu)
net = slim.fully_connected(net, 128, activation_fn=tf.nn.elu)
net = slim.fully_connected(net, 32, activation_fn=tf.nn.elu)
latent_layer = net
net = slim.fully_connected(net, 128, activation_fn=tf.nn.elu)
net = slim.fully_connected(net, 512, activation_fn=tf.nn.elu)
net = slim.fully_connected(net, 784, activation_fn=tf.nn.sigmoid)
net = tf.reshape(net, [-1, 28, 28, 1])
output_layer = net

meansq = tf.reduce_mean(tf.square(output_layer - input_layer))

learn_rate = 0.1   # how fast the model should learn
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)

# initialising stuff and starting the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# defining batch size, number of epochs and learning rate
batch_size = 100  # how many images to use together for training
hm_epochs = 20    # how many times to go through the entire dataset
# running the model for a 1000 epochs taking 100 images in batches
# total improvement is printed out after each epoch
training_losses = []
validation_losses = []
numBatches = int(xtrain.shape[0]/batch_size)
for epoch in range(hm_epochs):
    epoch_loss = 0    # initializing error as 0
    for i in range(numBatches):
        epoch_x = xtrain[ i*batch_size : (i+1)*batch_size ]
        _, c = sess.run([optimizer, meansq], feed_dict={input_layer: epoch_x})
        epoch_loss += c

    epoch_loss = epoch_loss/numBatches
    validate_loss = sess.run([meansq], feed_dict={input_layer: xvalidate})
    training_losses.append(epoch_loss)
    validation_losses += validate_loss
    print('Epoch', epoch, '/', hm_epochs, 'training loss:',epoch_loss, ', validation loss:', validate_loss)

# pick any image
any_image = xtrain[999]
# run it though the autoencoder
output_any_image = sess.run(output_layer, feed_dict={input_layer:[any_image]})
# run it though just the encoder
encoded_any_image = sess.run(latent_layer, feed_dict={input_layer:[any_image]})
# print the original image

plt.imshow(np.reshape(any_image, [28, 28]),  cmap='Greys')
plt.savefig("input.png")
# print the encoding
print(encoded_any_image)

plt.imshow(np.reshape(output_any_image, [28, 28]),  cmap='Greys')
plt.savefig("output.png")

print(training_losses)
print(validation_losses)

utils.general.plotLosses([training_losses, validation_losses], ['Train Set', 'Validation Set'], 'keras_autoencoder', 'keras_autoencoder_losses.png')
