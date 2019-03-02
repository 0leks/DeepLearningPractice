import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import time
import os


print('Tensorflow version:', tf.__version__)
assert tf.__version__ >= "2.0" # TensorFlow ≥ 2.0 required

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
print('images shape: ', train_images.shape, ', min,max: ', np.amin(train_images), ',', np.amax(train_images))

BUFFER_SIZE = 60000
BATCH_SIZE = 256
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print('data_set: ', train_dataset)


def make_generator_model():
  model = tf.keras.Sequential()
  model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  assert model.output_shape == (None, 7*7*256)  # Note: None is the batch size

  model.add(layers.Dense(28 * 28, use_bias=False, activation='tanh'))
  model.add(layers.Reshape((28, 28, 1)))
  assert model.output_shape == (None, 28, 28, 1)  # Note: None is the batch size

  return model


# def make_generator_model():
#   model = tf.keras.Sequential()
#   model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
#   model.add(layers.BatchNormalization())
#   model.add(layers.LeakyReLU())
#
#   model.add(layers.Reshape((7, 7, 256)))
#   assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size
#
#   model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
#   assert model.output_shape == (None, 7, 7, 128)
#   model.add(layers.BatchNormalization())
#   model.add(layers.LeakyReLU())
#
#   model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#   assert model.output_shape == (None, 14, 14, 64)
#   model.add(layers.BatchNormalization())
#   model.add(layers.LeakyReLU())
#
#   model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
#   assert model.output_shape == (None, 28, 28, 1)
#
#   return model

generator = make_generator_model()

# noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)
#
# plt.axis("off")
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()


def make_discriminator_model():
  model = tf.keras.Sequential()
  model.add(layers.Flatten(input_shape=[28, 28, 1]))
  assert model.output_shape == (None, 28 * 28)
  model.add(layers.Dense(1))
  assert model.output_shape == (None, 1)

  return model

#
# def make_discriminator_model():
#   model = tf.keras.Sequential()
#   model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
#                           input_shape=[28, 28, 1]))
#   model.add(layers.LeakyReLU())
#   model.add(layers.Dropout(0.3))
#
#   model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
#   model.add(layers.LeakyReLU())
#   model.add(layers.Dropout(0.3))
#
#   model.add(layers.Flatten())
#   model.add(layers.Dense(1))
#
#   return model

discriminator = make_discriminator_model()


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

#%%time
print('begin training')
train(train_dataset, EPOCHS)

#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.axis("off")
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()

decision = discriminator(generated_image)
print (decision)