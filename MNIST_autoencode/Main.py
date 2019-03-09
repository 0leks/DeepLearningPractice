import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print('tensorflow version:', tf.__version__)

# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
# from tensorflow.keras.models import Model

# class Autoencoder2(object):
#   def __init__(self):
#     # Encoding
#     input_layer = Input(shape=(28, 28, 1))
#     encoding_conv_layer_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
#     encoding_pooling_layer_1 = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_1)
#     encoding_conv_layer_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoding_pooling_layer_1)
#     encoding_pooling_layer_2 = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_2)
#     encoding_conv_layer_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoding_pooling_layer_2)
#     code_layer = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_3)
#
#     # Decoding
#     decodging_conv_layer_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(code_layer)
#     decodging_upsampling_layer_1 = UpSampling2D((2, 2))(decodging_conv_layer_1)
#     decodging_conv_layer_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(decodging_upsampling_layer_1)
#     decodging_upsampling_layer_2 = UpSampling2D((2, 2))(decodging_conv_layer_2)
#     decodging_conv_layer_3 = Conv2D(16, (3, 3), activation='relu')(decodging_upsampling_layer_2)
#     decodging_upsampling_layer_3 = UpSampling2D((2, 2))(decodging_conv_layer_3)
#     output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decodging_upsampling_layer_3)
#
#     self._model = Model(input_layer, output_layer)
#     self._model.compile(optimizer='adadelta', loss='binary_crossentropy')
#
#   def train(self, input_train, input_test, batch_size, epochs):
#     self._model.fit(input_train,
#                     input_train,
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     shuffle=True,
#                     validation_data=(
#                       input_test,
#                       input_test))
#
#   def getDecodedImage(self, encoded_imgs):
#     decoded_image = self._model.predict(encoded_imgs)
#     return decoded_image

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

(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

print(x_train.shape)
print(x_test.shape)

# Prepare input
x_train = (x_train.astype('float32')-127.5)/127.5
x_test = (x_test.astype('float32')-127.5)/127.5
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)

# Tensorflow implementation
autoencodertf = Autoencoder(x_train.shape[1], 728)
autoencodertf.train(x_train, x_test, 100, epochs=30)
encoded_img = autoencodertf.getEncodedImage(x_test[1])
decoded_img = autoencodertf.getDecodedImage(x_test[1])

# Tensorflow implementation results
plt.figure(figsize=(20, 4))
subplot = plt.subplot(2, 10, 1)
plt.imshow(x_test[1].reshape(28, 28))
plt.gray()
subplot.get_xaxis().set_visible(False)
subplot.get_yaxis().set_visible(False)

subplot = plt.subplot(2, 10, 2)
plt.imshow(decoded_img.reshape(28, 28))
plt.gray()
subplot.get_xaxis().set_visible(False)
subplot.get_yaxis().set_visible(False)
plt.show()