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

BATCH_SIZE = 500
NUM_EPOCHS = 100
LATENT_DIM = 2

# resets tensorflow graph, use if training multiple graphs in same session
#tf.reset_default_graph()

print('Training model with', LATENT_DIM, 'latent dimensions and', NUM_EPOCHS, 'epochs')
# Train the model with 2d latent space
model, trainLoss, valLoss, testLoss = vae_util.trainer(input_dim, num_sample, mnist, learning_rate=2e-4, batch_size=BATCH_SIZE, num_epoch=NUM_EPOCHS, n_z=LATENT_DIM)

#with plt.xkcd():
print('Best validation loss:', min(valLoss))
###### plot losses ######
fig, ax = plt.subplots()
t = np.arange(0, NUM_EPOCHS, 1)
ax.plot(t, trainLoss, label='Training Set (50000)')
ax.plot(t, valLoss, label='Validation Set (10000)')
# don't plot test set losses as it might influence thoughts to see results on test set.
#ax.plot(t, testLoss, label='Test Set (10000)')
ax.set(xlabel='epoch', ylabel='loss', title='Losses avg(sum(crossentropy of 784 pixels))')
ax.legend()
ax.grid()
fig.savefig("losses.png")

###### Test the trained model: transformation by sampling from Normal(mu, sigma) for each latent dimension ######
# I decided this isnt meaningful
batch = mnist.validation.next_batch(10000)
# z = model.transformerSample(batch[0])
# fig, ax = plt.subplots()
# #averageMu = z[:, 2]+z[:,3]
# ax.scatter(z[:, 0], z[:, 1], c=np.argmax(batch[1], 1), alpha=0.3)
# ax.set_xlabel(r'N($\mu_0$,$\sigma_0$)')
# ax.set_ylabel(r'N($\mu_1$,$\sigma_1$)')
# ax.set_title(r'Latent N($\mu_0$,$\sigma_0$) and N($\mu_1$,$\sigma_1$) for 10000 validation images')
# #fig.colorbar()
# ax.grid()
# fig.savefig('I_transformed.png')


###### Test the trained model: transformation plot only mu ######
#batch = mnist.test.next_batch(3000)
z = model.transformer2(batch[0])
fig, ax = plt.subplots()
colorValues = np.argmax(batch[1], 1)# * 9 / 8 - 0.5
s = ax.scatter(z[:, 0], z[:, 1], s=4, c=colorValues, alpha=0.3, cmap='tab10')
#s = ax.scatter(z[:, 0], z[:, 1], c=np.argmax(batch[1], 1), alpha=0.3)
ax.set_xlabel(r'$\mu_0$')
ax.set_ylabel(r'$\mu_1$')
ax.set_title(r'Latent $\mu_0$ and $\mu_1$ for 10000 validation images')
cbar = fig.colorbar(s, ax=ax)
ax.grid()
fig.savefig('transformed.png')

###### Test the trained model: continuous latent space ######
n = 40
minRange = -4
maxRange = 4
x = np.linspace(minRange, maxRange, n)
y = np.linspace(minRange, maxRange, n)

I_latent = np.empty((h*n, w*n))
num_extra_latent = LATENT_DIM - 2
extra_dim = []
if num_extra_latent > 0:
  extra_dim = [0] * num_extra_latent
for i, yi in enumerate(x):
  for j, xi in enumerate(y):
    #dim = [xi, yi] * int(LATENT_DIM/2)
    dim = np.concatenate(([xi, yi], extra_dim))
    z = np.array([dim] * model.batch_size)
    x_hat = model.generator(z)
    I_latent[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = x_hat[0].reshape(28, 28)

fig = plt.figure(num=None, figsize=(6.4, 4.8), dpi=int(n*28/2))

plt.imshow(I_latent, cmap="gray")
plt.xticks((0, n*28), (minRange, maxRange))
plt.yticks((0, n*28), (maxRange, minRange))
plt.xlabel(r'$\mu_0$')
plt.ylabel(r'$\mu_1$')
plt.title('Sampling latent space across first two variables (others set to 0)')
plt.savefig('latent.png')
plt.close(fig)


fig = plt.figure(num=None, figsize=(6.4, 4.8), dpi=int(n*28/2))
plt.imshow(-I_latent, cmap="gray")
plt.xticks((0, n*28), (minRange, maxRange))
plt.yticks((0, n*28), (maxRange, minRange))
plt.xlabel(r'$\mu_0$')
plt.ylabel(r'$\mu_1$')
plt.title('Sampling latent space across first two variables (others set to 0)')
plt.savefig('latent_inverted.png')
plt.close(fig)

###### Test the trained model: generation ######
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
plt.xticks([]);
plt.yticks([]);
plt.title(r'Generate images by sampling all $\mu$ and $\sigma$ from N(0,1) as input')
plt.savefig('generated.png')
plt.close(fig)

# Test the trained model: reconstruction
n = 10
batch = mnist.validation.next_batch(n*n)
x_reconstructed = model.reconstructor(batch[0])
I_reconstructed = np.empty((h*n, 2*w*n))
for i in range(n):
  for j in range(n):
    x = np.concatenate(
      (
        batch[0][i*n+j, :].reshape(h, w),
        x_reconstructed[i*n+j, :].reshape(h, w)
      ) , axis=1
    )
    I_reconstructed[i*h:(i+1)*h, j*2*w:(j+1)*2*w] = x

fig = plt.figure()
plt.imshow(I_reconstructed, cmap='gray')
plt.yticks([]);
plt.xticks((np.arange(n*2) + 0.5)*28, ('orig', 'recon')*n, rotation=-90);
plt.title('Original vs Reconstructed validation images')
plt.savefig('reconstructed.png')
plt.close(fig)

print('finished')
