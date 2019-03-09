import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg') # Allows generating plots without popup. Must call before importing pyplot.
import matplotlib.pyplot as plt

import utils.general
import utils.data
import utils.models

print('Loading fashion mnist data from file')
#(xtrain, ytrain), (xvalidate, yvalidate) = utils.data.loadCSVImageDataset('../datasets/fashionmnist/fashion-mnist_train.csv')
(xtrain, ytrain), (xvalidate, yvalidate) = utils.data.getFashionMNISTdatasetClassification(validation_ratio=1/6)

plt.imshow(xtrain[0, :, :, 0],  cmap='Greys')
plt.colorbar()
plt.savefig('image0_.png')

#https://medium.com/@connectwithghosh/simple-autoencoder-example-using-tensorflow-in-python-on-the-fashion-mnist-dataset-eee63b8ed9f1
#https://blog.keras.io/building-autoencoders-in-keras.html


xtrain = np.reshape(xtrain, [xtrain.shape[0], 784])
xvalidate = np.reshape(xvalidate, [xvalidate.shape[0], 784])

# Deciding how many nodes wach layer should have
n_nodes_inpl = 784  #encoder
n_nodes_hl1  = 32  #encoder
n_nodes_hl2  = 32  #decoder
n_nodes_outl = 784  #decoder

# first hidden layer has 784*32 weights and 32 biases
hidden_1_layer_vals = {
    'weights':tf.Variable(tf.random_normal([n_nodes_inpl,n_nodes_hl1])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))
}
# second hidden layer has 32*784 weights and 784 biases
output_layer_vals = {
    'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_outl])),
    'biases':tf.Variable(tf.random_normal([n_nodes_outl]))
}

# image with shape 784 goes in
input_layer = tf.placeholder('float', [None, 784])
# multiply output of input_layer wth a weight matrix and add biases
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer,hidden_1_layer_vals['weights']), hidden_1_layer_vals['biases']))
# multiply output of layer_2 wth a weight matrix and add biases
output_layer = tf.matmul(layer_1,output_layer_vals['weights']) + output_layer_vals['biases']
# define our cost function
meansq = tf.reduce_mean(tf.square(output_layer - input_layer))
# define our optimizer
learn_rate = 0.1   # how fast the model should learn
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)

# initialising stuff and starting the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# defining batch size, number of epochs and learning rate
batch_size = 100  # how many images to use together for training
hm_epochs = 10    # how many times to go through the entire dataset
# running the model for a 1000 epochs taking 100 images in batches
# total improvement is printed out after each epoch
training_losses = []
validation_losses = []
for epoch in range(hm_epochs):
    epoch_loss = 0    # initializing error as 0
    for i in range(int(xtrain.shape[0]/batch_size)):
        epoch_x = xtrain[ i*batch_size : (i+1)*batch_size ]
        _, c = sess.run([optimizer, meansq], feed_dict={input_layer: epoch_x})
        epoch_loss += c

    validate_loss = sess.run([meansq], feed_dict={input_layer: xvalidate})
    training_losses.append(epoch_loss)
    validation_losses += validate_loss
    print('Epoch', epoch, '/', hm_epochs, 'training loss:',epoch_loss, ', validation loss:', validate_loss)

# pick any image
any_image = xtrain[999]
# run it though the autoencoder
output_any_image = sess.run(output_layer, feed_dict={input_layer:[any_image]})
# run it though just the encoder
encoded_any_image = sess.run(layer_1, feed_dict={input_layer:[any_image]})
# print the original image

plt.imshow(np.reshape(any_image, [28, 28]),  cmap='Greys')
plt.savefig("input.png")
# print the encoding
print(encoded_any_image)

plt.imshow(np.reshape(output_any_image, [28, 28]),  cmap='Greys')
plt.savefig("output.png")

print(training_losses)
print(validation_losses)

utils.general.plotLosses([training_losses], ['Train Set'], 'simple_autoencoder', 'losses_train.png')
utils.general.plotLosses([validation_losses], ['Validation Set'], 'simple_autoencoder', 'losses_validate.png')
