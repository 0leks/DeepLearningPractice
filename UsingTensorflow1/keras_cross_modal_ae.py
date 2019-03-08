import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA


import utils.data
import utils.general

useFashionDataset = True
useConvEncoder = True
useConvDecoder = useConvEncoder
latent_dim = 2
outerLayerDim = 8
innerLayerDim = 16
batch_size = 100
epochs = 100

runTitle = 'XModal_'
runTitle += ('Conv_' if useConvEncoder else '')
runTitle += ('Fashion' if useFashionDataset else 'Digits') + '_'
runTitle += 'Latent' + str(latent_dim)

debugPath = utils.general.setupDebugPath('test_runs', runTitle)

# Load the dataset
if useFashionDataset:
    (xtrain, xtrainlabels), (xvalidate, xvalidatelabels) = utils.data.getFashionMNISTdatasetClassification(validation_ratio=1/6)
else:
    (xtrain, xtrainlabels), (xvalidate, xvalidatelabels) = utils.data.getMNISTdatasetClassification(validation_ratio=1/6)

ytrain = utils.data.getCannyEdgeDetectionFromData(xtrain, 100, 200)
yvalidate = utils.data.getCannyEdgeDetectionFromData(xvalidate, 100, 200)
for i in range(20):
    combined = np.append(xtrain[i], ytrain[i], axis=0)
    utils.general.saveImage(combined[:, :, 0], debugPath + 'randomImage' + str(i) + '.png')

input_shape = xtrain[0].shape
output_dim = np.prod(input_shape)
latent_shape = (latent_dim,)


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

    model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=latent_shape))
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
loss_function = 'binary_crossentropy'

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
y = tf.keras.Input(shape=input_shape)

x_latent = x_encoder(x)
y_latent = y_encoder(y)

x_hat = x_decoder(x_latent)
y_hat = y_decoder(y_latent)
x_y_hat = y_decoder(x_latent)
y_x_hat = x_decoder(y_latent)

x_x_reconstructor = tf.keras.Model(x, x_hat)
x_x_reconstructor.compile(loss=loss_function, optimizer=optimizer)

y_y_reconstructor = tf.keras.Model(y, y_hat)
y_y_reconstructor.compile(loss=loss_function, optimizer=optimizer)

x_y_reconstructor = tf.keras.Model(x, x_y_hat)
x_y_reconstructor.compile(loss=loss_function, optimizer=optimizer)

y_x_reconstructor = tf.keras.Model(y, y_x_hat)
y_x_reconstructor.compile(loss=loss_function, optimizer=optimizer)

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

    losses[:, -1] /= numBatches
    # Plot the progress
    print ("%d [x_x loss: %f] [y_y_loss: %f] [x_y_loss: %f] [y_x_loss: %f]" % (epoch, losses[0][-1], losses[1][-1], losses[2][-1], losses[3][-1]))
    utils.general.plotLosses(losses, ['x_x', 'y_y', 'x_y', 'y_x'], 'losses', debugPath + 'losses.png')

    utils.general.saveLatentSamplingImage(x_decoder, latent_dim, debugPath + 'samplingX' + str(epoch) + '.png', title='X Sampling latent space', n=40, minRange=-4, maxRange=4)
    utils.general.saveLatentSamplingImage(y_decoder, latent_dim, debugPath + 'samplingY' + str(epoch) + '.png', title='Y Sampling latent space', n=40, minRange=-4, maxRange=4)

    doPCA = latent_dim > 2
    x_z = x_encoder.predict(xvalidate)
    y_z = y_encoder.predict(yvalidate)
    if doPCA:
        print('Doing PCA')
        x_z = PCA(n_components=2).fit_transform(x_z)
        y_z = PCA(n_components=2).fit_transform(y_z)

    min1 = min(np.amin(x_z[:,0]), np.amin(y_z[:,0]))
    max1 = max(np.amax(x_z[:,0]), np.amax(y_z[:,0]))
    min2 = min(np.amin(x_z[:,1]), np.amin(y_z[:,1]))
    max2 = max(np.amax(x_z[:,1]), np.amax(y_z[:,1]))

    border = 0.1
    limits = [min1 - border, max1 + border, min2 - border, max2 + border]

    utils.general.saveLatentSpaceImage(x_z, xvalidatelabels, debugPath + 'latentSpaceX' + str(epoch) + '.png', title=r'Latent $\mu_0$ and $\mu_1$ for ' + str(xvalidate.shape[0]) + ' X validation images', doPCA=doPCA, limits=limits)
    utils.general.saveLatentSpaceImage(y_z, xvalidatelabels, debugPath + 'latentSpaceY' + str(epoch) + '.png', title=r'Latent $\mu_0$ and $\mu_1$ for ' + str(yvalidate.shape[0]) + ' Y validation images', doPCA=doPCA, limits=limits)
