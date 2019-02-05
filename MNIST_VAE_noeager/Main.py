# adapted from https://github.com/shaohua0116/VAE-Tensorflow

import vae_util

import numpy as np
import matplotlib
matplotlib.use('Agg') # Allows generating plots without popup. Must call before importing pyplot.
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# turns off interactive mode
plt.ioff()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True, validation_size=10000)
num_sample = mnist.train.num_examples
print('train shape: ', mnist.train.images.shape)
print('test shape: ', mnist.test.images.shape)
print('validate shape: ', mnist.validation.images.shape)
input_dim = mnist.train.images[0].shape[0]
w = h = 28

BATCH_SIZE = 200
NUM_EPOCHS = 10
LATENT_DIM = 2

# resets tensorflow graph, use if training multiple graphs in same session
#tf.reset_default_graph()

if(NUM_EPOCHS):
  print('Training model with', LATENT_DIM, 'latent dimensions and', NUM_EPOCHS, 'epochs')
  # Train the model with 2d latent space
  model, trainLoss, valLoss, testLoss = vae_util.trainer(input_dim, num_sample, mnist, learning_rate=1e-4, batch_size=BATCH_SIZE, num_epoch=NUM_EPOCHS, n_z=LATENT_DIM)

  # plot losses
  fig, ax = plt.subplots()
  t = np.arange(0, NUM_EPOCHS, 1)
  ax.plot(t, trainLoss, label='Training Set')
  ax.plot(t, valLoss, label='Validation Set')
  ax.plot(t, testLoss, label='Test Set')
  ax.set(xlabel='epoch', ylabel='loss', title='Losses')
  legend = ax.legend()
  ax.grid()
  fig.savefig("losses.png")

  # Test the trained model: transformation
  batch = mnist.test.next_batch(3000)[0]
  z = model.transformer(batch)
  fig, ax = plt.subplots()
  plt.scatter(z[:, 0], z[:, 1], c=np.argmax(batch[1], 1), alpha=0.5)
  ax.set_xlabel(r'$\mu$')
  ax.set_ylabel(r'$\sigma$')
  ax.set_title(r'Latent $\mu$ and $\sigma$ for 3000 random test images')
  plt.colorbar()
  plt.grid()
  plt.savefig('I_transformed.png')
  plt.close(fig)

  # Test the trained model: continuous latent space
  n = 20
  x = np.linspace(-2, 2, n)
  y = np.linspace(-2, 2, n)

  I_latent = np.empty((h*n, w*n))
  for i, yi in enumerate(x):
    for j, xi in enumerate(y):
      z = np.array([[xi, yi]] * model.batch_size)
      x_hat = model.generator(z)
      I_latent[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = x_hat[0].reshape(28, 28)

  fig = plt.figure()
  plt.imshow(I_latent, cmap="gray")
  plt.savefig('I_latent.png')
  plt.close(fig)

  # Test the trained model: generation
  # Sample noise vectors from N(0, 1)
  z = np.random.normal(size=[model.batch_size, model.n_z])
  x_generated = model.generator(z)

  n = np.sqrt(model.batch_size).astype(np.int32)
  I_generated = np.empty((h*n, w*n))
  for i in range(n):
    for j in range(n):
      I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = x_generated[i*n+j, :].reshape(28, 28)

  fig = plt.figure()
  plt.imshow(I_generated, cmap='gray')
  plt.savefig('I_generated.png')
  plt.close(fig)

  # Test the trained model: reconstruction
  n = 10
  batch = mnist.test.next_batch(n*n)[0]
  x_reconstructed = model.reconstructor(batch)
  I_reconstructed = np.empty((h*n, 2*w*n))
  for i in range(n):
    for j in range(n):
      x = np.concatenate(
        (x_reconstructed[i*n+j, :].reshape(h, w),
         batch[0][i*n+j, :].reshape(h, w)),
        axis=1
      )
      I_reconstructed[i*h:(i+1)*h, j*2*w:(j+1)*2*w] = x

  fig = plt.figure()
  plt.imshow(I_reconstructed, cmap='gray')
  plt.savefig('I_reconstructed.png')
  plt.close(fig)
