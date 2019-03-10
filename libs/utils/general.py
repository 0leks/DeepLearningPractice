import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Allows generating plots without popup. Must call before importing pyplot.
import matplotlib.pyplot as plt
# turns off interactive mode for pyplot
plt.ioff()


def createDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)


def setupDebugPath(testDir, runName):
    createDirectory(testDir)
    imagePath = testDir + '/' + runName
    counter = 0
    while os.path.exists(imagePath + '_' + str(counter)):
        counter = counter + 1
    imagePath = imagePath + '_' + str(counter) + '/'
    createDirectory(imagePath)
    return imagePath


def plotLosses(lossArrays, lossLabels, title, filePath):
    plt.figure()
    for loss, label in zip(lossArrays, lossLabels):
        plt.plot(loss, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.savefig(filePath)
    plt.close()


def saveImage(image, filePath):
    plt.figure()
    plt.imshow(image, vmin=0, vmax=1, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filePath, bbox_inches='tight')
    plt.close()


def saveLatentSpaceImage(z, labels, filePath, title='Latent Space', doPCA=False, limits=None):
    fig, ax = plt.subplots()
    if limits is not None:
        plt.xlim(limits[0], limits[1])
        plt.ylim(limits[2], limits[3])
    s = ax.scatter(z[:, 0], z[:, 1], s=1, c=labels, alpha=0.7, cmap='tab10')
    ax.set_xlabel(r'$\mu_0$')
    ax.set_ylabel(r'$\mu_1$')
    ax.set_title(title)
    ticks = np.arange(10)*8/9 + 0.5
    cbar = fig.colorbar(s, ax=ax, ticks=ticks)
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    cbar.ax.set_yticklabels(class_names)
    ax.grid()
    fig.savefig(filePath, bbox_inches='tight')
    plt.close(fig)


def saveLatentSamplingImage(decoder, latent_dim, filePath, title='Sampling latent space acrod first two variables (others set to 0)', n=40, minRange=-4, maxRange=4):
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
    plt.imshow(I_latent, vmin=0, vmax=1, cmap="gray")
    plt.xticks((0, n * 28), (minRange, maxRange))
    plt.yticks((0, n * 28), (maxRange, minRange))
    plt.xlabel(r'$\mu_0$')
    plt.ylabel(r'$\mu_1$')
    plt.title(title)
    plt.savefig(filePath, bbox_inches='tight')
    plt.close(fig)


def saveReconstructionImages(images, filePath):
    columns = len(images)
    rows = images[0].shape[0]
    lines = np.concatenate(images, axis=2)
    reconstructed = np.reshape(lines, (rows*28,columns*28))

    fig = plt.figure(num=None, figsize=(6.4, 4.8), dpi=int(max(rows, columns) * 28))
    plt.imshow(reconstructed, cmap='gray')
    hat = '\u0302'
    arrow = '\u2192'
    labels = (
            'x', 
            'x' + arrow + 'x' + hat, 
            'y' + arrow + 'x' + hat, 
            'y', 
            'x' + arrow + 'y' + hat, 
            'y' + arrow + 'y' + hat)
    plt.xticks((np.arange(6) + 0.5)*28, labels, rotation=-90)
    plt.yticks([])
    plt.title('Reconstructed validation images')
    plt.savefig(filePath)
    plt.close(fig)


def saveLatentSpaceTraverse(n, data, encoder, decoder, filePath, data2=None, decoder2=None):
    numSamples = int(n/2) if decoder2 is not None else n
    idx = np.random.randint(data.shape[0], size=numSamples * 2)
    randomImages = data[idx, :, :, :]
    randomImages2 = data2[idx, :, :, :] if data2 is not None else randomImages
    interpValues = np.arange(0, 1 + 1 / n, 1 / n)
    final_output = []
    for i in range(numSamples):
        start = randomImages[2 * i:2 * i + 1, :, :, :]
        end = randomImages[2 * i + 1:2 * i + 2, :, :, :]

        start_z = encoder.predict(start)
        end_z = encoder.predict(end)

        interp = [np.multiply(start_z, 1 - t) + np.multiply(end_z, t) for t in interpValues]
        interp = np.concatenate(interp)
        output = decoder.predict(interp)
        output = np.concatenate((start, output, end), axis=0)
        output = np.reshape(output, output.shape[:-1])
        line = np.concatenate(output, axis=1)
        final_output.append(line)
        if decoder2 is not None:
            start2 = randomImages2[2 * i:2 * i + 1, :, :, :]
            end2 = randomImages2[2 * i + 1:2 * i + 2, :, :, :]
            output = decoder2.predict(interp)
            output = np.concatenate((start2, output, end2), axis=0)
            output = np.reshape(output, output.shape[:-1])
            line = np.concatenate(output, axis=1)
            final_output.append(line)

    final_output = np.asarray(final_output)
    final_output = np.concatenate(final_output, axis=0)
    fig = plt.figure(num=None, figsize=(6.4, 4.8), dpi=int(n * 28))
    plt.imshow(final_output, vmin=0, vmax=1, cmap="gray")
    labels = ['%.2f' % value for value in interpValues]
    labels = np.concatenate((['Original'], labels, ['Original']), axis=0)
    plt.xticks((np.arange(n + 3) + 0.5)*28, labels, rotation=-90)
    plt.yticks([])
    plt.title('Interpolation in Latent Space z')
    plt.savefig(filePath, bbox_inches='tight')
    plt.close(fig)