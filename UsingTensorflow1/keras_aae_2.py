import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import time
import matplotlib
matplotlib.use('Agg') # Allows generating plots without popup. Must call before importing pyplot.
import matplotlib.pyplot as plt

# turns off interactive mode for pyplot
plt.ioff()

import utils.data
import utils.general

useFashionDataset = True
useConvEncoder = False
useConvDecoder = useConvEncoder
latent_dim = 32
outerConvLayerDim = 16
innerConvLayerDim = 32
outerDenseLayerDim = 512
innerDenseLayerDim = 256
outerDenseLayerDim = 128
innerDenseLayerDim = 64
batch_size = 250
epochs = 200

runTitle = 'AAE_V2_'
runTitle += ('Conv_' if useConvEncoder else '')
runTitle += ('Fashion' if useFashionDataset else 'Digits') + '_'
runTitle += 'Latent' + str(latent_dim)

debugPath = utils.general.setupDebugPath('test_runs', runTitle)

# Load the dataset
if useFashionDataset:
    (xtrain, xtrainlabels), (xvalidate, xvalidatelabels) = utils.data.getFashionMNISTdatasetClassification(validation_ratio=1/6)
else:
    (xtrain, xtrainlabels), (xvalidate, xvalidatelabels) = utils.data.getMNISTdatasetClassification(validation_ratio=1/6)

input_shape = xtrain[0].shape
output_dim = np.prod(input_shape)
latent_shape = (latent_dim,)


def build_encoder():
    model = tf.keras.Sequential()

    if useConvEncoder:
        model.add(tf.keras.layers.Conv2D(outerConvLayerDim, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.Conv2D(outerConvLayerDim, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(innerConvLayerDim, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(innerConvLayerDim, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Flatten())
    else:
        model.add(tf.keras.layers.Flatten(input_shape=input_shape))

    model.add(tf.keras.layers.Dense(outerDenseLayerDim, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(innerDenseLayerDim, activation='relu'))
    model.add(tf.keras.layers.Dense(latent_dim))
    model.summary()

    img = tf.keras.Input(shape=input_shape)
    encoded = model(img)

    return tf.keras.Model(img, encoded)


def build_decoder():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(innerDenseLayerDim, activation='relu', input_shape=latent_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(outerDenseLayerDim, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    if useConvDecoder:
        model.add(tf.keras.layers.Dense(7 * 7 * innerConvLayerDim, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Reshape((7, 7, innerConvLayerDim)))
        model.add(tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear'))
        model.add(tf.keras.layers.Conv2D(innerConvLayerDim, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(innerConvLayerDim, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear'))
        model.add(tf.keras.layers.Conv2D(outerConvLayerDim, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(outerConvLayerDim, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid'))
    else:
        model.add(tf.keras.layers.Dense(np.product(input_shape), activation='sigmoid'))

    model.add(tf.keras.layers.Reshape(input_shape))
    model.summary()

    latent = tf.keras.Input(shape=latent_shape)
    decoded = model(latent)

    return tf.keras.Model(latent, decoded)


def build_discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(outerDenseLayerDim, activation='relu', input_shape=latent_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(innerDenseLayerDim, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.summary()

    latent = tf.keras.Input(shape=latent_shape)
    disc = model(latent)

    return tf.keras.Model(latent, disc)


optimizer = tf.train.AdamOptimizer(0.0002)
loss_function = 'binary_crossentropy'

x_encoder = build_encoder()
x_encoder.compile(loss='binary_crossentropy', optimizer=optimizer)

x_decoder = build_decoder()
x_decoder.compile(loss='binary_crossentropy', optimizer=optimizer)

x_discriminator = build_discriminator()
x_discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
x_discriminator.trainable = False

# reconstructor takes input_shape and outputs input_shape
x = tf.keras.Input(shape=input_shape)

x_latent = x_encoder(x)

x_hat = x_decoder(x_latent)

x_x_disc = x_discriminator(x_latent)

x_x_reconstructor = tf.keras.Model(x, x_hat)
x_x_reconstructor.compile(loss=loss_function, optimizer=optimizer)

x_x_discriminator = tf.keras.Model(x, x_x_disc)
x_x_discriminator.compile(loss=loss_function, optimizer=optimizer)

losses = np.asarray([[], [], [], [], [], []])
numBatches = int(xtrain.shape[0]/batch_size)
for epoch in range(epochs):
    losses = np.append(losses, [[0], [0], [0], [0], [0], [0]], axis=1)
    startTime_s = time.time()
    for i in range(numBatches):
        epoch_x = xtrain[i*batch_size : (i+1)*batch_size]

        # train reconstruction
        losses[0][-1] += x_x_reconstructor.train_on_batch(epoch_x, epoch_x)

        if epoch > 5:
            # train generator
            # The generator wants the discriminator to label the generated samples as valid (ones)
            c_loss = x_x_discriminator.train_on_batch(epoch_x, np.ones((batch_size, 1)))
            losses[5][-1] += c_loss

            if epoch%2 == 0: # train disc half the time
                # train discriminator
                generated_latent_x = x_encoder.predict(epoch_x)
                true_latent_x = np.random.normal(0, 1, (batch_size, latent_dim))
                d_loss_real = x_discriminator.train_on_batch(true_latent_x, np.ones((batch_size, 1)))
                d_loss_fake = x_discriminator.train_on_batch(generated_latent_x, np.zeros((batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                losses[4][-1] += d_loss

    deltaTime_s = (time.time() - startTime_s)

    losses[:, -1] /= numBatches
    # Plot the progress
    hat = '\u0302'
    print ('%d [x_x loss: %f] [y_y_loss: %f] [x_y_loss: %f] [y_x_loss: %f] [%.1f seconds] [x v x\u0302 disc loss: %f] [x_x_gen loss: %f]' % (epoch, losses[0][-1], losses[1][-1], losses[2][-1], losses[3][-1], deltaTime_s, losses[4][-1], losses[5][-1]))
    utils.general.plotLosses(losses, ['x_x', 'y_y', 'x_y', 'y_x'], 'losses', debugPath + 'losses.png')

    #utils.general.saveLatentSamplingImage(x_decoder, latent_dim, debugPath + 'samplingX' + str(epoch) + '.png', title='X Sampling latent space', n=40, minRange=-4, maxRange=4)
    #utils.general.saveLatentSamplingImage(y_decoder, latent_dim, debugPath + 'samplingY' + str(epoch) + '.png', title='Y Sampling latent space', n=40, minRange=-4, maxRange=4)

    doPCA = latent_dim > 2
    x_z = x_encoder.predict(xvalidate)
    if doPCA:
        print('Doing PCA')
        x_z = PCA(n_components=2).fit_transform(x_z)

    utils.general.saveLatentSpaceImage(x_z, xvalidatelabels, debugPath + 'latentSpaceX' + str(epoch) + '.png', title=r'Latent $\mu_0$ and $\mu_1$ for ' + str(xvalidate.shape[0]) + ' X validation images', doPCA=doPCA)

    n = 5
    x_batch = xvalidate[:n, :, :, :]
    idx = np.random.randint(xvalidate.shape[0], size=n)
    x_batch2 = xvalidate[idx, :, :, :]

    x_batch = np.concatenate((x_batch, x_batch2), axis=0)

    x_x_reconstructed = x_x_reconstructor.predict(x_batch)

    utils.general.saveReconstructionImages((
        x_batch,
        x_x_reconstructed,
    ), debugPath + 'reconstructed' + str(epoch) + '.png')

    utils.general.saveLatentSpaceTraverse(20, xvalidate, x_encoder, x_decoder, debugPath + 'latentTraverse' + str(epoch) + '.png')




