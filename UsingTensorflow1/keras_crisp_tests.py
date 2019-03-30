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
useConvAutoEncoder = False
gaussian_stddev = 5
latent_dim = 2
batch_size = 64
epochs = 200
recon_loss = 'binary_crossentropy'
#recon_loss = 'mean_squared_error'
disc_loss = 'binary_crossentropy'

runTitle = 'AEGAN_'
runTitle += ('Fashion' if useFashionDataset else 'Digits') + '_'
runTitle += ('Deep_' if useDeepNetwork else '')
runTitle += ('Conv_' if useConvAutoEncoder else '')
runTitle += 'Latent' + str(latent_dim)

debugPath = utils.general.setupDebugPath('test_runs', runTitle)
print('Running: ' + debugPath)

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

encoder, decoder, discriminator, reconstructor, combined = utils.models.makeVAE_GAN(input_shape, layer_sizes, latent_dim, optimizer, recon_loss, disc_loss, use_conv=useConvAutoEncoder)
print('Saving to directory:', debugPath)


d_losses = []
c_losses = []
r_losses = []
accuracies_train = []
accuracies_val = []
val_reconstruction = []
val_accuracy = []
val_disc_loss = []
numBatches = int(xtrain.shape[0]/batch_size)
print('Num Batches per epoch:', numBatches)

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

        x_hat = reconstructor.predict(epoch_x)
        # train to mark actual data as 1
        d_loss_real = discriminator.train_on_batch(epoch_x, np.ones((batch_size, 1)))
        # train to mark reconstructed data as 0
        d_loss_fake = discriminator.train_on_batch(x_hat, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Encode / Disc
        # ---------------------

        # train generator(encoder + decoder) to trick decoder into marking reconstructed data as 1
        c_loss = combined.train_on_batch(epoch_x, np.ones((batch_size, 1)))

        # ---------------------
        #  Train Encode / Decode
        # ---------------------
        r_loss = reconstructor.train_on_batch(epoch_x, epoch_x)

        total_d_loss += d_loss[0]/batch_size
        total_acc_loss += d_loss[1]
        total_c_loss += c_loss/batch_size
        total_r_loss += r_loss/batch_size

    # average total losses over number of batches
    total_d_loss /= numBatches
    total_acc_loss /= numBatches
    total_c_loss /= numBatches
    total_r_loss /= numBatches

    d_losses.append(total_d_loss)
    c_losses.append(total_c_loss)
    r_losses.append(total_r_loss)
    accuracies_train.append(total_acc_loss)

    # evaluate on validation data
    val_eval = reconstructor.evaluate(xvalidate, xvalidate, verbose=0)
    reconstructedXVal = reconstructor.predict(xvalidate)
    val_disc_eval, val_disc_acc = discriminator.evaluate(reconstructedXVal, np.zeros((xvalidate.shape[0], 1)), verbose=0)
    val_reconstruction.append(val_eval)
    val_disc_loss.append(val_disc_eval)
    accuracies_val.append(val_disc_acc)

    print(debugPath + " epoch %d [Disc loss: %f, acc.: %.2f%%] [Gen loss: %f] [Recon loss: %f]" % (epoch, total_d_loss, 100*total_acc_loss, total_c_loss, total_r_loss))
    print(debugPath + " epoch %d Validation [Disc loss: %f] [Disc acc: %.2f%%] [Recon loss: %f]" % (epoch, val_disc_eval, 100*val_disc_acc, val_eval))
    # Plot the progress
    if epoch > 0:
        utils.general.plotLosses([d_losses, c_losses, r_losses], ['Disc loss', 'Gen loss', 'Recon loss'], 'AEGAN losses', debugPath + 'AEGAN_losses' + str(epoch) + '.png')
        utils.general.plotLosses([val_reconstruction, val_disc_loss], ['Recon loss', 'Disc loss'], 'AEGAN validation losses', debugPath + 'AEGAN_validation_losses' + str(epoch) + '.png')
        utils.general.plotLosses([accuracies_train, accuracies_val], ['train acc', 'val acc'], 'AEGAN accuracies', debugPath + 'AEGAN_accuracies' + str(epoch) + '.png')

    doPCA = latent_dim > 2
    z = encoder.predict(xvalidate)
    # TODO Shireen recommended some algo other than PCA because it preserves neighborhoods
    z = PCA(n_components=2).fit_transform(z) if doPCA else z

    maximum = gaussian_stddev + 5
    utils.general.saveLatentSpaceImage(z,
                                       xvalidatelabels,
                                       debugPath + 'latentSpace' + str(epoch) + '.png',
                                       title=r'Latent $\mu_0$ and $\mu_1$ for ' + str(
                                           xvalidate.shape[0]) + ' X validation images',
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

    utils.general.saveLatentSpaceTraverse(20, xvalidate, encoder, decoder,
                                          debugPath + 'latentTraverse' + str(epoch) + '.png')
