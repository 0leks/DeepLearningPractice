#https://skymind.ai/wiki/generative-adversarial-network-gan
import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg') # Allows generating plots without popup. Must call before importing pyplot.
import matplotlib.pyplot as plt

import utils.data
import utils.general

input_shape = (28, 28, 1)

def build_discriminator():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    img = tf.keras.Input(shape=input_shape)
    validity = model(img)

    return tf.keras.Model(img, validity)


def build_generator():
    noise_shape = (100,)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(256, input_shape=noise_shape))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Dense(np.prod(input_shape), activation='tanh'))
    model.add(tf.keras.layers.Reshape(input_shape))

    model.summary()

    noise = tf.keras.Input(shape=noise_shape)
    img = model(noise)

    return tf.keras.Model(noise, img)

optimizer = tf.train.AdamOptimizer(0.0002)


# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Build and compile the generator
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

# The generator takes noise as input and generated imgs
z = tf.keras.Input(shape=(100,))
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The valid takes generated images as input and determines validity
valid = discriminator(img)

# The combined model  (stacked generator and discriminator) takes
# noise as input => generates images => determines validity
combined = tf.keras.Model(z, valid)
combined.summary()
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# Load the dataset
#(xtrain, ytrain), (xvalidate, yvalidate) = utils.data.getMNISTdatasetClassification(validation_ratio=1/6)
(xtrain, ytrain), (xvalidate, yvalidate) = utils.data.getFashionMNISTdatasetClassification(validation_ratio=1/6)

batch_size = 100
half_batch = int(batch_size / 2)
epochs = 100

d_losses = []
g_losses = []
numBatches = int(xtrain.shape[0]/batch_size)
testNoise = np.random.normal(0, 1, (5, 100))
for epoch in range(epochs):
    total_d_loss = 0
    total_acc_loss = 0
    total_g_loss = 0
    for i in range(numBatches):
        epoch_x = xtrain[ i*batch_size : (i+1)*batch_size ]
        # ---------------------
        #  Train Discriminator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, 100))

        # Generate a half batch of new images
        gen_imgs = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(epoch_x, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size*2, 100))

        # The generator wants the discriminator to label the generated samples
        # as valid (ones)
        valid_y = np.array([1] * batch_size*2)

        # Train the generator
        g_loss = combined.train_on_batch(noise, valid_y)
        total_d_loss += d_loss[0]
        total_acc_loss += d_loss[1]
        total_g_loss += g_loss

    total_d_loss /= numBatches
    total_acc_loss /= numBatches
    total_g_loss /= numBatches

    d_losses.append(total_d_loss)
    g_losses.append(total_g_loss)
    # Plot the progress
    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, total_d_loss, 100*total_acc_loss, total_g_loss))

    testNoise2 = np.random.normal(0, 1, (5, 100))
    gen_imgs = generator.predict(testNoise)
    gen_imgs2 = generator.predict(testNoise2)

    for i in range(5):
        plt.figure()
        plt.imshow(gen_imgs[i, :, :, 0], cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.savefig('images/agenerated' + str(epoch) + '_' + str(i) + '.png', bbox_inches='tight')
        plt.close()
    for i in range(5):
        plt.figure()
        plt.imshow(gen_imgs2[i, :, :, 0], cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.savefig('images/agenerated' + str(epoch) + '_' + str(i+5) + '.png', bbox_inches='tight')
        plt.close()

    utils.general.plotLosses([d_losses], ['d_loss'], 'keras_gan', 'keras_gan_d_losses.png')
    utils.general.plotLosses([g_losses], ['g_loss'], 'keras_gan', 'keras_gan_g_losses.png')

