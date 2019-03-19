#https://skymind.ai/wiki/generative-adversarial-network-gan
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg') # Allows generating plots without popup. Must call before importing pyplot.
import matplotlib.pyplot as plt

import utils.data
import utils.general


useDeepNetwork = True

useFashionDataset = False
latent_dim = 8
batch_size = 100
epochs = 200

runTitle = 'AAE_tests_'
runTitle += ('Fashion' if useFashionDataset else 'Digits') + '_'
runTitle += 'Latent' + str(latent_dim)

debugPath = utils.general.setupDebugPath('test_runs', runTitle)

if useFashionDataset:
    (xtrain, xtrainlabels), (xvalidate, xvalidatelabels) = utils.data.getFashionMNISTdatasetClassification(validation_ratio=1/6)
else:
    (xtrain, xtrainlabels), (xvalidate, xvalidatelabels) = utils.data.getMNISTdatasetClassification(validation_ratio=1/6)

# x to 1000 relu to 1000 relu to 8 linear z
# z to 1000 relu to 1000 relu to 28x28 sigmoid output x_hat
# z to 1000 relu to 1000 relu to 1 sigmoid D

input_shape = (28, 28, 1)
output_dim = np.prod(input_shape)
latent_shape = (latent_dim,)

def build_encoder():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    if useDeepNetwork:
        model.add(tf.keras.layers.Dense(1000, activation='relu'))
        model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dense(latent_dim))
    model.summary()

    img = tf.keras.Input(shape=input_shape)
    encoded = model(img)

    return tf.keras.Model(img, encoded)


def build_decoder():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(1000, activation='relu', input_shape=latent_shape))
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    if useDeepNetwork:
        model.add(tf.keras.layers.Dense(1000, activation='relu'))
        model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dense(np.product(input_shape), activation='sigmoid'))
    model.add(tf.keras.layers.Reshape(input_shape))
    model.summary()

    latent = tf.keras.Input(shape=latent_shape)
    decoded = model(latent)

    return tf.keras.Model(latent, decoded)


def build_discriminator():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(1000, activation='relu', input_shape=latent_shape))
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    if useDeepNetwork:
        model.add(tf.keras.layers.Dense(1000, activation='relu'))
        model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    latent = tf.keras.Input(shape=latent_shape)
    validity = model(latent)

    return tf.keras.Model(latent, validity)

# TODO try out different learning rates
# TODO maybe also different optimizer
optimizer = tf.train.AdamOptimizer(0.0002)

discriminator = build_discriminator()
# TODO is binary cross entropy the right loss function for this?
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

encoder = build_encoder()
encoder.compile(loss='binary_crossentropy', optimizer=optimizer)

decoder = build_decoder()
decoder.compile(loss='binary_crossentropy', optimizer=optimizer)

# reconstructor takes input_shape and outputs input_shape
x = tf.keras.Input(shape=input_shape)
latent = encoder(x)

x_hat = decoder(latent)

reconstructor = tf.keras.Model(x, x_hat)
reconstructor.summary()
# TODO the paper uses euclidian distance
reconstructor.compile(loss='binary_crossentropy', optimizer=optimizer)

discriminator.trainable = False
valid = discriminator(latent)

combined = tf.keras.Model(x, valid)
combined.summary()
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

d_losses = []
c_losses = []
r_losses = []
numBatches = int(xtrain.shape[0]/batch_size)
for epoch in range(epochs):
    total_d_loss = 0
    total_acc_loss = 0
    total_c_loss = 0
    total_r_loss = 0
    for i in range(numBatches):
        epoch_x = xtrain[i*batch_size : (i+1)*batch_size]

        # ---------------------
        #  Train Discriminator
        # ---------------------

        z = encoder.predict(epoch_x)
        # TODO allow using different distribution here.
        # paper uses normal with deviation 5 for MNIST 
        noise = np.random.normal(0, 5, (batch_size, latent_dim))
        # The noise signal should be marked valid (1) by the discriminator
        d_loss_real = discriminator.train_on_batch(noise, np.ones((batch_size, 1)))
        # The encoded samples should be marked fake (0) by the discriminator
        d_loss_fake = discriminator.train_on_batch(z, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Encode / Disc
        # ---------------------

        # The generator wants the discriminator to label the generated samples valid (1)
        # combined model is x -> z -> D
        c_loss = combined.train_on_batch(epoch_x, np.ones((batch_size, 1)))

        # ---------------------
        #  Train Encode / Decode
        # ---------------------
        r_loss = reconstructor.train_on_batch(epoch_x, epoch_x)

        total_d_loss += d_loss[0]
        total_acc_loss += d_loss[1]
        total_c_loss += c_loss
        total_r_loss += r_loss

    total_d_loss /= numBatches
    total_acc_loss /= numBatches
    total_c_loss /= numBatches
    total_r_loss /= numBatches

    d_losses.append(total_d_loss)
    c_losses.append(total_c_loss)
    r_losses.append(total_r_loss)
    # Plot the progress
    print ("%d [Disc loss: %f, acc.: %.2f%%] [Gen loss: %f] [Recon loss: %f]" % (epoch, total_d_loss, 100*total_acc_loss, total_c_loss, total_r_loss))

    utils.general.plotLosses([d_losses], ['d_loss'], 'AAE losses', debugPath + 'AAE_d_losses' + str(epoch) + '.png')
    utils.general.plotLosses([c_losses], ['c_loss'], 'AAE losses', debugPath + 'AAE_c_losses' + str(epoch) + '.png')
    utils.general.plotLosses([r_losses], ['r_loss'], 'AAE losses', debugPath + 'AAE_r_losses' + str(epoch) + '.png')
    utils.general.plotLosses([d_losses, c_losses, r_losses], ['Disc loss', 'Gen loss', 'Recon loss'], 'AAE losses', debugPath + 'AAE_losses' + str(epoch) + '.png')

    doPCA = latent_dim > 2
    z = encoder.predict(xvalidate)
    # TODO Shireen recommended some algo other than PCA because it preserves neighborhoods
    z = PCA(n_components=2).fit_transform(z) if doPCA else z

    utils.general.saveLatentSpaceImage(z, xvalidatelabels, debugPath + 'latentSpace' + str(epoch) + '.png', title=r'Latent $\mu_0$ and $\mu_1$ for ' + str(xvalidate.shape[0]) + ' X validation images', doPCA=doPCA)
    
    # TODO implement find nearest image in training data (using euclidian distance)

    # TODO demonstrate image reconstruction with x, x_hat, and nearest to x in training data
    