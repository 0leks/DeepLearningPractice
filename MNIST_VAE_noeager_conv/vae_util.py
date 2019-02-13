
import numpy as np
import tensorflow as tf
import time
from tensorflow.contrib import slim


def trainer(input_dim, num_samples, dataset, learning_rate=1e-3, batch_size=100, num_epoch=75, n_z=10):
  #input_dim = (28,28)
  model = VariantionalAutoencoder(input_dim, learning_rate=learning_rate,
                                  batch_size=batch_size, n_z=n_z)
  totalLoss = []
  testLoss = []
  validateLoss = []
  counter = 0
  merge = tf.summary.merge_all()
  for epoch in range(num_epoch):
    start_time = time.time()
    epochLoss = []
    for iter in range(num_samples // batch_size):
      counter += 1
      # Obtain a batch
      batch = dataset.train.next_batch(batch_size)
      #batch = batch[0].reshape([batch_size, 28, 28, 1])
      batch = batch[0]
      #print(batch.shape)
      # Execute the forward and the backward pass and report computed losses
      loss, recon_loss, latent_loss = model.run_single_step(batch)
      epochLoss.append(loss)
    totalLoss.append(sum(epochLoss) / len(epochLoss))
    delta_time = time.time() - start_time

    batch = dataset.validation.next_batch(10000)[0]
    x_hat = model.reconstructor(batch)
    vloss, _, _ = model.compute_loss(batch)
    validateLoss.append(vloss)

    summary = model.sess.run(merge, feed_dict={model.x: dataset.validation.next_batch(10000)[0], model.is_train: False})
    model.writer.add_summary(summary, epoch)

    # batch = dataset.test.next_batch(10000)[0]
    # x_hat = model.reconstructor(batch)
    # tloss, _, _ = model.compute_loss(batch)
    # testLoss.append(tloss)

    if epoch % 1 == 0:
      print('[Epoch {} Time {}s] Loss: {}, Recon loss: {}, Latent loss: {}'.format(epoch, delta_time, totalLoss[-1], recon_loss, latent_loss))

  print('training losses: ', totalLoss)
  print('validation losses: ', validateLoss)
  print('testing losses: ', testLoss)
  #saver = tf.train.Saver()
  #save_path = saver.save(model.sess, "tmp/model.ckpt")
  #print('Model saved in path:', save_path)
  print('Done Training!')
  return model, totalLoss, validateLoss, testLoss


class VariantionalAutoencoder(object):

  def __init__(self, input_dim, learning_rate=1e-3, batch_size=100, n_z=10):
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.n_z = n_z
    self.input_dim = input_dim

    self.build()

    #self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    self.sess = tf.InteractiveSession()
    self.sess.run(tf.global_variables_initializer())
    self.writer = tf.summary.FileWriter('graphs', self.sess.graph)

  def add_summaries(self, var, name='summary'):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


  # Build the netowrk and the loss functions
  def build(self):
    self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 784])
    self.is_train = tf.placeholder(tf.bool, name="is_train");
    net = self.x
    self.add_summaries(net, 'x')
    print(net.get_shape()) # (?, 784)
    outerLayerSize = 200
    innerLayerSize = 100

    # Encode
    # x -> z_mean, z_sigma -> z
    net = slim.fully_connected(net, outerLayerSize, scope='enc_fc1', activation_fn=tf.nn.elu)
    net = tf.layers.batch_normalization(net, training=self.is_train)
    print(net.get_shape()) # (?, 200)
    self.add_summaries(net)

    net = slim.fully_connected(net, innerLayerSize, scope='enc_fc2', activation_fn=tf.nn.elu)
    net = tf.layers.batch_normalization(net, training=self.is_train)
    print(net.get_shape()) # (?, 200)
    self.add_summaries(net)

    self.z_mu = slim.fully_connected(net, self.n_z, scope='enc_fc3_mu', activation_fn=None)
    print('z_mu shape: ', self.z_mu.get_shape()) # (?, 2)
    self.z_log_sigma_sq = slim.fully_connected(net, self.n_z, scope='enc_fc3_sigma', activation_fn=None)
    print('z_log_sigma_sq shape: ', self.z_log_sigma_sq.get_shape()) # (?, 2)
    with tf.name_scope('reparam_trick'):
      eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
      self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps
      print('z shape: ', self.z.get_shape()) # (?, 2)
      net = self.z
    self.add_summaries(net, 'z')


    # Decode
    # z -> x_hat
    net = slim.fully_connected(net, innerLayerSize, scope='dec_fc1', activation_fn=tf.nn.elu)
    net = tf.layers.batch_normalization(net, training=self.is_train)
    print(net.get_shape())
    self.add_summaries(net)


    net = slim.fully_connected(net, outerLayerSize, scope='dec_fc2', activation_fn=tf.nn.elu)
    net = tf.layers.batch_normalization(net, training=self.is_train)
    print(net.get_shape())
    self.add_summaries(net)

    self.x_hat = slim.fully_connected(net, 784, scope='dec_fc3', activation_fn=tf.sigmoid) #tf.nn.elu
    print(self.x_hat.get_shape())
    self.add_summaries(net, 'x_hat')

    # Loss
    # Reconstruction loss
    with tf.name_scope('recon_loss'):
      # Minimize the cross-entropy loss
      # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
      epsilon = 1e-10
      recon_loss = -tf.reduce_sum(
        self.x * tf.log( epsilon +self.x_hat) + ( 1 -self.x) * tf.log( epsilon + 1 -self.x_hat),
        axis=1
      )
      self.recon_loss = tf.reduce_mean(recon_loss)
    self.summary_recon_loss = tf.summary.scalar('reconstruction loss', self.recon_loss)

    # Latent loss
    with tf.name_scope('latent_loss'):
      # Kullback Leibler divergence: measure the difference between two distributions
      # Here we measure the divergence between the latent distribution and N(0, 1)
      latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)
      self.latent_loss = tf.reduce_mean(latent_loss)
    self.summary_latent_loss = tf.summary.scalar('latent loss', self.latent_loss)

    with tf.name_scope('total_loss'):
      self.total_loss = tf.reduce_mean(recon_loss + latent_loss)
    self.summary_total_loss = tf.summary.scalar('total loss', self.total_loss)

    #with tf.name_scope('Trainer'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
    return


  # Execute the forward and the backward pass
  def run_single_step(self, x):
    _, loss, recon_loss, latent_loss = self.sess.run(
      [self.train_op, self.total_loss, self.recon_loss, self.latent_loss],
      feed_dict={self.x: x, self.is_train: True}
    )
    return loss, recon_loss, latent_loss

  def compute_loss(self, x):
    loss, recon_loss, latent_loss = self.sess.run(
      [self.total_loss, self.recon_loss, self.latent_loss],
      feed_dict={self.x: x, self.is_train: False}
    )
    return loss, recon_loss, latent_loss


  # x -> x_hat
  def reconstructor(self, x):
    x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x, self.is_train: False})
    return x_hat

  # z -> x
  def generator(self, z):
    x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z, self.is_train: False})
    return x_hat

  # x -> z
  def transformerSample(self, x):
    z = self.sess.run(self.z, feed_dict={self.x: x, self.is_train: False})
    return z

  # x -> z_mu
  def transformer2(self, x):
    z_mu = self.sess.run(self.z_mu, feed_dict={self.x: x, self.is_train: False})
    return z_mu
