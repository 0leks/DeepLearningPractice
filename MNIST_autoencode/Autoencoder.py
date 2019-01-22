import tensorflow as tf

# Taken from https://rubikscode.net/2018/11/26/3-ways-to-implement-autoencoders-with-tensorflow-and-python/
# https://gist.github.com/NMZivkovic/356322e47819ac58be88a73c02dac56e#file-autoencoder_tensorflow-py

class Autoencoder(object):
  def __init__(self, inout_dim, encoded_dim):
    learning_rate = 0.1

    # Weights and biases
    hiddel_layer_weights = tf.Variable(tf.random_normal([inout_dim, encoded_dim]))
    hiddel_layer_biases = tf.Variable(tf.random_normal([encoded_dim]))
    output_layer_weights = tf.Variable(tf.random_normal([encoded_dim, inout_dim]))
    output_layer_biases = tf.Variable(tf.random_normal([inout_dim]))

    # Neural network
    self._input_layer = tf.placeholder('float', [None, inout_dim])
    self._hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(self._input_layer, hiddel_layer_weights), hiddel_layer_biases))
    self._output_layer = tf.matmul(self._hidden_layer, output_layer_weights) + output_layer_biases
    self._real_output = tf.placeholder('float', [None, inout_dim])

    self._meansq = tf.reduce_mean(tf.square(self._output_layer - self._real_output))
    self._optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self._meansq)
    self._training = tf.global_variables_initializer()
    self._session = tf.Session()

  def train(self, input_train, input_test, batch_size, epochs):
    self._session.run(self._training)

    for epoch in range(epochs):
      epoch_loss = 0
      for i in range(int(input_train.shape[0] / batch_size)):
        epoch_input = input_train[i * batch_size: (i + 1) * batch_size]
        _, c = self._session.run([self._optimizer, self._meansq],
                                 feed_dict={self._input_layer: epoch_input, self._real_output: epoch_input})
        epoch_loss += c
        print('Epoch', epoch, '/', epochs, 'loss:', epoch_loss)

  def getEncodedImage(self, image):
    encoded_image = self._session.run(self._hidden_layer, feed_dict={self._input_layer: [image]})
    return encoded_image

  def getDecodedImage(self, image):
    decoded_image = self._session.run(self._output_layer, feed_dict={self._input_layer: [image]})
    return decoded_image