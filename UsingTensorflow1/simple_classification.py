import tensorflow as tf

import utils.general
import utils.data
import utils.models

print('tensorflow version:', tf.__version__)

(xtrain, ytrain), (xvalidate, yvalidate) = utils.data.getFashionMNISTdatasetClassification(validation_ratio=1/6)
#(xtrain, ytrain), (xvalidate, yvalidate) = utils.data.getMNISTdatasetClassification(validation_ratio=1/6)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#modelShape = [512, 256, 128, 10]
modelShape = [512, 256, 128, 64, 32, 16, 10]
model = utils.models.makeDenseClassifierModel((28, 28, 1), modelShape)

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

numEpochs = 10
trainLoss = [0] * numEpochs
validationLoss = [0] * numEpochs

for i in range(numEpochs):
    # no batches here, just doing entire dataset
    model.fit(xtrain, ytrain, epochs=1)

    train_loss, train_acc = model.evaluate(xtrain, ytrain)
    trainLoss[i] = train_loss

    validation_loss, validation_acc = model.evaluate(xvalidate, yvalidate)
    validationLoss[i] = validation_loss

utils.general.plotLosses([trainLoss, validationLoss], ['Train Set', 'Validation Set'], 'Dense classifier(' + str(modelShape) + ')', 'losses.png')
