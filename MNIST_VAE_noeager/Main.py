# from https://github.com/shaohua0116/VAE-Tensorflow


import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vae_util
from vae_util import Test
from vae_util import VariantionalAutoencoder


test = Test()
print(test.a)
print(test.b)

plt.ioff()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
num_sample = mnist.train.num_examples
input_dim = mnist.train.images[0].shape[0]
w = h = 28

BATCH_SIZE = 100

partOneEpochs = 0
partTwoEpochs = 50

if(partOneEpochs):
  print('Training model with latent dimensions=5')
  # Train the model
  model = vae_util.trainer(input_dim, num_sample, mnist, learning_rate=1e-4,  batch_size=BATCH_SIZE, num_epoch=partOneEpochs, n_z=5)

  # Test the trained model: reconstruction
  batch = mnist.test.next_batch(100)
  x_reconstructed = model.reconstructor(batch[0])

  n = np.sqrt(model.batch_size).astype(np.int32)
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

tf.reset_default_graph()

if(partTwoEpochs):
  print('Training model with latent dimensions=2')
  # Train the model with 2d latent space
  model_2d = vae_util.trainer(input_dim, num_sample, mnist, learning_rate=1e-4,  batch_size=100, num_epoch=partTwoEpochs, n_z=2)

  # Test the trained model: transformation
  batch = mnist.test.next_batch(3000)
  z = model_2d.transformer(batch[0])
  fig = plt.figure()
  plt.scatter(z[:, 0], z[:, 1], c=np.argmax(batch[1], 1))
  plt.colorbar()
  plt.grid()
  plt.savefig('I_transformed.png')
  plt.close(fig)

  # Test the trained model: transformation
  n = 20
  x = np.linspace(-2, 2, n)
  y = np.linspace(-2, 2, n)

  I_latent = np.empty((h*n, w*n))
  for i, yi in enumerate(x):
    for j, xi in enumerate(y):
      z = np.array([[xi, yi]]*model_2d.batch_size)
      x_hat = model_2d.generator(z)
      I_latent[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = x_hat[0].reshape(28, 28)

  fig = plt.figure()
  plt.imshow(I_latent, cmap="gray")
  plt.savefig('I_latent.png')
  plt.close(fig)
