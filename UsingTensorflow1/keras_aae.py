#https://skymind.ai/wiki/generative-adversarial-network-gan
import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg') # Allows generating plots without popup. Must call before importing pyplot.
import matplotlib.pyplot as plt

import utils.data
import utils.general


input_shape = (28, 28, 1)
output_dim = np.prod(input_shape)
latent_dim = 2
latent_shape = (latent_dim,)

def build_encoder():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dense(latent_dim))
    #model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.summary()

    img = tf.keras.Input(shape=input_shape)
    encoded = model(img)

    return tf.keras.Model(img, encoded)


def build_decoder():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(256, input_shape=latent_shape))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dense(output_dim, activation='sigmoid'))
    model.add(tf.keras.layers.Reshape(input_shape))
    model.summary()

    latent = tf.keras.Input(shape=latent_shape)
    decoded = model(latent)

    return tf.keras.Model(latent, decoded)


def build_discriminator():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=latent_shape))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    latent = tf.keras.Input(shape=latent_shape)
    validity = model(latent)

    return tf.keras.Model(latent, validity)


optimizer = tf.train.AdamOptimizer(0.0002)

discriminator = build_discriminator()
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
reconstructor.compile(loss='binary_crossentropy', optimizer=optimizer)

discriminator.trainable = False
valid = discriminator(latent)

combined = tf.keras.Model(x, valid)
combined.summary()
combined.compile(loss='binary_crossentropy', optimizer=optimizer)


# Load the dataset
#(xtrain, ytrain), (xvalidate, yvalidate) = utils.data.getMNISTdatasetClassification(validation_ratio=1/6)
(xtrain, ytrain), (xvalidate, yvalidate) = utils.data.getFashionMNISTdatasetClassification(validation_ratio=1/6)

batch_size = 100
half_batch = int(batch_size / 2)
epochs = 100

d_losses = []
c_losses = []
r_losses = []
numBatches = int(xtrain.shape[0]/batch_size)
testNoise = np.random.normal(0, 1, (5, latent_dim))
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
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(z, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(noise, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Encode / Disc
        # ---------------------

        # The generator wants the discriminator to label the generated samples
        # as valid (ones)
        valid_y = np.array([0] * batch_size)

        # Train the generator
        c_loss = combined.train_on_batch(epoch_x, valid_y)

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
    print ("%d [D loss: %f, acc.: %.2f%%] [C loss: %f] [R loss: %f]" % (epoch, total_d_loss, 100*total_acc_loss, total_c_loss, total_r_loss))


    testNoise2 = np.random.normal(0, 1, (3, latent_dim))
    gen_imgs = decoder.predict(testNoise)
    gen_imgs2 = decoder.predict(testNoise2)

    for i in range(5):
        plt.figure()
        plt.imshow(gen_imgs[i, :, :, 0], vmin=0, vmax=1, cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.savefig('images3/agenerated' + str(epoch) + '_' + str(i) + '.png', bbox_inches='tight')
        plt.close()
    for i in range(3):
        plt.figure()
        plt.imshow(gen_imgs2[i, :, :, 0], vmin=0, vmax=1, cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.savefig('images3/agenerated' + str(epoch) + '_' + str(i+5) + '.png', bbox_inches='tight')
        plt.close()

    utils.general.plotLosses([d_losses], ['d_loss'], 'keras_gan', 'keras_gan_d_losses.png')
    utils.general.plotLosses([c_losses], ['c_loss'], 'keras_gan', 'keras_gan_c_losses.png')
    utils.general.plotLosses([r_losses], ['r_loss'], 'keras_gan', 'keras_gan_r_losses.png')

    ###### Test the trained model: continuous latent space ######
    n = 40
    minRange = -4
    maxRange = 4
    x = np.linspace(minRange, maxRange, n)
    y = np.linspace(minRange, maxRange, n)

    I_latent = np.empty((28 * n, 28 * n))
    num_extra_latent = latent_dim - 2
    extra_dim = []
    if num_extra_latent > 0:
        extra_dim = [0] * num_extra_latent
    for i, yi in enumerate(x):
        for j, xi in enumerate(y):
            # dim = [xi, yi] * int(LATENT_DIM/2)
            dim = np.concatenate(([xi, yi], extra_dim))
            z = np.array([dim])
            x_hat = decoder.predict(z)
            I_latent[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_hat[0].reshape(28, 28)

    fig = plt.figure(num=None, figsize=(6.4, 4.8), dpi=int(n * 28 / 2))

    plt.imshow(I_latent, vmin=0, vmax=1, cmap="Greys")
    plt.xticks((0, n * 28), (minRange, maxRange))
    plt.yticks((0, n * 28), (maxRange, minRange))
    plt.xlabel(r'$\mu_0$')
    plt.ylabel(r'$\mu_1$')
    plt.title('Sampling latent space across first two variables (others set to 0)')
    plt.savefig('latent' + str(epoch) + '.png', bbox_inches='tight')
    plt.close(fig)

    ###### Test the trained model: transformation plot only mu ######
    # batch = mnist.test.next_batch(3000)
    batch = xvalidate
    z = encoder.predict(batch)
    fig, ax = plt.subplots()
    colorValues = yvalidate#np.argmax(yvalidate, 1)  # * 9 / 8 - 0.5
    s = ax.scatter(z[:, 0], z[:, 1], s=4, c=colorValues, alpha=0.5, cmap='tab10')
    # s = ax.scatter(z[:, 0], z[:, 1], c=np.argmax(batch[1], 1), alpha=0.3)
    ax.set_xlabel(r'$\mu_0$')
    ax.set_ylabel(r'$\mu_1$')
    ax.set_title(r'Latent $\mu_0$ and $\mu_1$ for 10000 validation images')
    cbar = fig.colorbar(s, ax=ax)
    ax.grid()
    fig.savefig('transformed' + str(epoch) + '.png', bbox_inches='tight')
    plt.close(fig)

