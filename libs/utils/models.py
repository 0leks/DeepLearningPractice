import tensorflow as tf

def makeDenseClassifierModel(input_shape, layer_sizes):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=input_shape))

  for layer in layer_sizes[:-1]:
    print('layer:', layer)
    model.add(tf.keras.layers.Dense(layer, activation=tf.nn.relu))

  model.add(tf.keras.layers.Dense(layer_sizes[-1], activation=tf.nn.softmax))
  return model