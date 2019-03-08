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
    plt.imshow(image, vmin=0, vmax=1, cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filePath, bbox_inches='tight')
    plt.close()

def saveLatentSpaceImage(z, labels, filePath, title='Latent Space', doPCA=False, limits=None):
    fig, ax = plt.subplots()
    if limits is not None:
        plt.xlim(limits[0], limits[1])
        plt.ylim(limits[2], limits[3])
    s = ax.scatter(z[:, 0], z[:, 1], s=3, c=labels, alpha=0.5, cmap='tab10')
    ax.set_xlabel(r'$\mu_0$')
    ax.set_ylabel(r'$\mu_1$')
    ax.set_title(title)
    fig.colorbar(s, ax=ax)
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
    plt.imshow(I_latent, vmin=0, vmax=1, cmap="Greys")
    plt.xticks((0, n * 28), (minRange, maxRange))
    plt.yticks((0, n * 28), (maxRange, minRange))
    plt.xlabel(r'$\mu_0$')
    plt.ylabel(r'$\mu_1$')
    plt.title(title)
    plt.savefig(filePath, bbox_inches='tight')
    plt.close(fig)