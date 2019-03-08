#https://skymind.ai/wiki/generative-adversarial-network-gan
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import cv2

import matplotlib
matplotlib.use('Agg') # Allows generating plots without popup. Must call before importing pyplot.
import matplotlib.pyplot as plt

import utils.data
import utils.general


# Load the dataset
#(xtrain, xtrainlabels), (xvalidate, xvalidatelabels) = utils.data.getMNISTdatasetClassification(validation_ratio=1/6)
(xtrain, xtrainlabels), (xvalidate, xvalidatelabels) = utils.data.getFashionMNISTdatasetClassification(validation_ratio=1/6)

ytrain = utils.data.getCannyEdgeDetectionFromData(xtrain, 100, 200)
yvalidate = utils.data.getCannyEdgeDetectionFromData(xvalidate, 100, 200)

input_shape = (28, 28, 1)
output_dim = np.prod(input_shape)
latent_dim = 2
latent_shape = (latent_dim,)

outerLayerDim = 4
innerLayerDim = 8
useConvEncoder = True
useConvDecoder = True


def build_encoder():
    model = tf.keras.Sequential()

    if useConvEncoder:
        model.add(tf.keras.layers.Conv2D(outerLayerDim, (3,3), padding='same', activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(innerLayerDim, (3,3), padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
    else:
        model.add(tf.keras.layers.Flatten(input_shape=input_shape))

    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(latent_dim))
    model.summary()

    img = tf.keras.Input(shape=input_shape)
    encoded = model(img)

    return tf.keras.Model(img, encoded)


def build_decoder():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(256, input_shape=latent_shape, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))

    if useConvDecoder:
        model.add(tf.keras.layers.Dense(7*7*innerLayerDim, activation='relu'))
        model.add(tf.keras.layers.Reshape((7,7,innerLayerDim)))
        model.add(tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear'))
        model.add(tf.keras.layers.Conv2D(outerLayerDim, (3,3), padding='same', activation='relu'))
        model.add(tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear'))
        model.add(tf.keras.layers.Conv2D(1, (3,3), padding='same', activation='sigmoid'))

    else:
        model.add(tf.keras.layers.Dense(np.product(input_shape), activation='sigmoid'))

    model.add(tf.keras.layers.Reshape(input_shape))
    model.summary()

    latent = tf.keras.Input(shape=latent_shape)
    decoded = model(latent)

    return tf.keras.Model(latent, decoded)


optimizer = tf.train.AdamOptimizer(0.0002)

x_encoder = build_encoder()
x_encoder.compile(loss='binary_crossentropy', optimizer=optimizer)

x_decoder = build_decoder()
x_decoder.compile(loss='binary_crossentropy', optimizer=optimizer)

y_encoder = build_encoder()
y_encoder.compile(loss='binary_crossentropy', optimizer=optimizer)

y_decoder = build_decoder()
y_decoder.compile(loss='binary_crossentropy', optimizer=optimizer)

# reconstructor takes input_shape and outputs input_shape
x = tf.keras.Input(shape=input_shape)
x_latent = x_encoder(x)
x_hat = x_decoder(x_latent)

x_x_reconstructor = tf.keras.Model(x, x_hat)
x_x_reconstructor.compile(loss='binary_crossentropy', optimizer=optimizer)
x_x_reconstructor.summary()

y = tf.keras.Input(shape=input_shape)
y_latent = y_encoder(y)
y_hat = y_decoder(y_latent)

y_y_reconstructor = tf.keras.Model(y, y_hat)
y_y_reconstructor.compile(loss='binary_crossentropy', optimizer=optimizer)
y_y_reconstructor.summary()

x2 = tf.keras.Input(shape=input_shape)
x2_latent = x_encoder(x2)
y2_hat = y_decoder(x2_latent)

x_y_reconstructor = tf.keras.Model(x2, y2_hat)
x_y_reconstructor.compile(loss='binary_crossentropy', optimizer=optimizer)
x_y_reconstructor.summary()

y2 = tf.keras.Input(shape=input_shape)
y2_latent = y_encoder(y2)
x2_hat = x_decoder(y2_latent)

y_x_reconstructor = tf.keras.Model(y2, x2_hat)
y_x_reconstructor.compile(loss='binary_crossentropy', optimizer=optimizer)
y_x_reconstructor.summary()


batch_size = 100
#half_batch = int(batch_size / 2)
epochs = 100

losses = np.asarray([[], [], [], []])

numBatches = int(xtrain.shape[0]/batch_size)
for epoch in range(epochs):
    losses = np.append(losses, [[0], [0], [0], [0]], axis=1)
    for i in range(numBatches):
        epoch_x = xtrain[i*batch_size : (i+1)*batch_size]
        epoch_y = ytrain[i*batch_size : (i+1)*batch_size]

        losses[0][-1] += x_x_reconstructor.train_on_batch(epoch_x, epoch_x)
        losses[1][-1] += y_y_reconstructor.train_on_batch(epoch_y, epoch_y)
        losses[2][-1] += x_y_reconstructor.train_on_batch(epoch_x, epoch_y)
        losses[3][-1] += y_x_reconstructor.train_on_batch(epoch_y, epoch_x)

    # losses[0][-1] /= numBatches
    # losses[1][-1] /= numBatches
    # losses[2][-1] /= numBatches
    # losses[3][-1] /= numBatches

    losses[:, -1] /= numBatches
    # Plot the progress
    print ("%d [x_x loss: %f] [y_y_loss: %f] [x_y_loss: %f] [y_x_loss: %f]" % (epoch, losses[0][-1], losses[1][-1], losses[2][-1], losses[3][-1]))
    utils.general.plotLosses(losses, ['x_x', 'y_y', 'x_y', 'y_x'], 'losses', 'losses.png')
    #print ("%d [D loss: %f, acc.: %.2f%%] [C loss: %f] [R loss: %f]" % (epoch, total_d_loss, 100*total_acc_loss, total_c_loss, total_r_loss))

    ###### Test the trained model: continuous latent space ######
    n = 40
    minRange = -4
    maxRange = 4
    x = np.linspace(minRange, maxRange, n)
    y = np.linspace(minRange, maxRange, n)

    I_latent_x = np.empty((28 * n, 28 * n))
    I_latent_y = np.empty((28 * n, 28 * n))
    num_extra_latent = latent_dim - 2
    extra_dim = []
    if num_extra_latent > 0:
        extra_dim = [0] * num_extra_latent
    for i, yi in enumerate(x):
        for j, xi in enumerate(y):
            # dim = [xi, yi] * int(LATENT_DIM/2)
            dim = np.concatenate(([xi, yi], extra_dim))
            z = np.array([dim])
            x_hat = x_decoder.predict(z)
            I_latent_x[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_hat[0].reshape(28, 28)
            y_hat = y_decoder.predict(z)
            I_latent_y[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = y_hat[0].reshape(28, 28)

    fig = plt.figure(num=None, figsize=(6.4, 4.8), dpi=int(n * 28 / 2))
    plt.imshow(I_latent_x, vmin=0, vmax=1, cmap="Greys")
    plt.xticks((0, n * 28), (minRange, maxRange))
    plt.yticks((0, n * 28), (maxRange, minRange))
    plt.xlabel(r'$\mu_0$')
    plt.ylabel(r'$\mu_1$')
    plt.title('Sampling latent space across first two variables (others set to 0)')
    plt.savefig('latentx' + str(epoch) + '.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(num=None, figsize=(6.4, 4.8), dpi=int(n * 28 / 2))
    plt.imshow(I_latent_y, vmin=0, vmax=1, cmap="Greys")
    plt.xticks((0, n * 28), (minRange, maxRange))
    plt.yticks((0, n * 28), (maxRange, minRange))
    plt.xlabel(r'$\mu_0$')
    plt.ylabel(r'$\mu_1$')
    plt.title('Sampling latent space across first two variables (others set to 0)')
    plt.savefig('latenty' + str(epoch) + '.png', bbox_inches='tight')
    plt.close(fig)

    ###### Test the trained model: transformation plot only mu ######
    z = x_encoder.predict(xvalidate)
    if latent_dim > 2:
        print('doing PCA')
        z = PCA(n_components=2).fit_transform(z)

    fig, ax = plt.subplots()
    s = ax.scatter(z[:, 0], z[:, 1], s=4, c=xvalidatelabels, alpha=0.5, cmap='tab10')
    ax.set_xlabel(r'$\mu_0$')
    ax.set_ylabel(r'$\mu_1$')
    ax.set_title(r'Latent $\mu_0$ and $\mu_1$ for 10000 validation images')
    fig.colorbar(s, ax=ax)
    ax.grid()
    fig.savefig('transformedx' + str(epoch) + '.png', bbox_inches='tight')
    plt.close(fig)

    z = y_encoder.predict(yvalidate)
    if latent_dim > 2:
        print('doing PCA')
        z = PCA(n_components=2).fit_transform(z)

    fig, ax = plt.subplots()
    s = ax.scatter(z[:, 0], z[:, 1], s=3, c=xvalidatelabels, alpha=0.5, cmap='tab10')
    ax.set_xlabel(r'$\mu_0$')
    ax.set_ylabel(r'$\mu_1$')
    ax.set_title(r'Latent $\mu_0$ and $\mu_1$ for 10000 validation images')
    fig.colorbar(s, ax=ax)
    ax.grid()
    fig.savefig('transformedy' + str(epoch) + '.png', bbox_inches='tight')
    plt.close(fig)



