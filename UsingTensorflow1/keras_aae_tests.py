#https://skymind.ai/wiki/generative-adversarial-network-gan
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg') # Allows generating plots without popup. Must call before importing pyplot.

import utils.data
import utils.general
import utils.models


useDeepNetwork = False
useFashionDataset = False
useCircleGaussian = False
num_gaussian = 10
circle_gaussian_radius = 10
gaussian_stddev = 5
latent_dim = 2
batch_size = 100
epochs = 200


runTitle = 'AAE_tests_'
runTitle += ('Fashion' if useFashionDataset else 'Digits') + '_'
runTitle += ('Deep_' if useDeepNetwork else '')
runTitle += ('CircleGaussian' + str(num_gaussian) + 'R' + str(circle_gaussian_radius) + '_' if useCircleGaussian else '')
runTitle += 'Latent' + str(latent_dim)

debugPath = utils.general.setupDebugPath('test_runs', runTitle)

if useFashionDataset:
    (xtrain, xtrainlabels), (xvalidate, xvalidatelabels) = utils.data.getFashionMNISTdatasetClassification(validation_ratio=1/6)
else:
    (xtrain, xtrainlabels), (xvalidate, xvalidatelabels) = utils.data.getMNISTdatasetClassification(validation_ratio=1/6)

input_shape = xtrain[0].shape
output_dim = np.prod(input_shape)

# TODO try out different learning rates
# TODO maybe also different optimizer
# TODO is binary cross entropy the right loss function for this?
# TODO the paper uses euclidian distance for reconstruction loss
optimizer = tf.train.AdamOptimizer(0.0002)

layer_sizes = [1000, 1000, 1000, 1000] if useDeepNetwork else [1000, 1000]

encoder, decoder, discriminator, reconstructor, combined = utils.models.makeAdversarialAutoEncoder(input_shape, layer_sizes, latent_dim, optimizer, 'binary_crossentropy')

d_losses = []
c_losses = []
r_losses = []
numBatches = int(xtrain.shape[0]/batch_size)
val_losses = []*3
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
        noise = np.random.normal(0, gaussian_stddev, (batch_size, latent_dim))
        if useCircleGaussian:
            noise = utils.general.sampleFromCircleGaussians(batch_size, num_gaussian, circle_gaussian_radius, gaussian_stddev)
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
    utils.general.plotLosses([d_losses, c_losses, r_losses], ['Disc loss', 'Gen loss', 'Recon loss'], 'AAE losses', debugPath + 'AAE_losses' + str(epoch) + '.png')

    doPCA = latent_dim > 2
    z = encoder.predict(xvalidate)
    # TODO Shireen recommended some algo other than PCA because it preserves neighborhoods
    z = PCA(n_components=2).fit_transform(z) if doPCA else z

    maximum = circle_gaussian_radius + gaussian_stddev + 5
    utils.general.saveLatentSpaceImage(z,
                                       xvalidatelabels,
                                       debugPath + 'latentSpace' + str(epoch) + '.png',
                                       title=r'Latent $\mu_0$ and $\mu_1$ for ' + str(xvalidate.shape[0]) + ' X validation images',
                                       doPCA=doPCA,
                                       limits=[-maximum, maximum, -maximum, maximum],
                                       mnist_fashion=useFashionDataset)
    
    # TODO implement find nearest image in training data (using euclidian distance)

    # TODO demonstrate image reconstruction with x, x_hat, and nearest to x in training data

    n = 5
    x_batch = xvalidate[:n, :, :, :]
    idx = np.random.randint(xvalidate.shape[0], size=n)
    x_batch2 = xvalidate[idx, :, :, :]

    x_batch = np.concatenate((x_batch, x_batch2), axis=0)

    x_reconstructed = reconstructor.predict(x_batch)

    utils.general.saveReconstructionImages((
        x_batch,
        x_reconstructed,
    ), debugPath + 'reconstructed' + str(epoch) + '.png')

    utils.general.saveLatentSpaceTraverse(20, xvalidate, encoder, decoder, debugPath + 'latentTraverse' + str(epoch) + '.png')