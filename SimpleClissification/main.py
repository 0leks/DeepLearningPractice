import tensorflow as tf
print('tensorflow version:', tf.__version__)
import matplotlib.pyplot as plt
import data_utils

(xtrain, ytrain), (xvalidate, yvalidate) = data_utils.getFashionMNISTdatasetClassification(validation_ratio=1/6)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

numEpochs = 30
trainLoss = [0] * numEpochs
validationLoss = [0] * numEpochs

for i in range(numEpochs):
  # no batches here, just doing entire dataset
  model.fit(xtrain, ytrain, epochs=1)

  train_loss, train_acc = model.evaluate(xtrain, ytrain)
  trainLoss[i] = train_loss

  validation_loss, validation_acc = model.evaluate(xvalidate, yvalidate)
  validationLoss[i] = validation_loss

# TODO add title and save figure as png instead of displaying plot
# TODO create local library for saving images, loading data
plt.figure()
plt.plot(trainLoss, label='Train Set')
plt.plot(validationLoss, label='Validation Set')
plt.xlabel('Epoch')
plt.ylabel('Loss (sparse_categorical_crossentropy')
plt.legend()
plt.show()