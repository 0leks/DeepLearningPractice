import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def dispInfo(name, x, y=None):
    string = name + ': ' + str(x.shape) + ' ' + str(np.amin(x)) + '->' + str(np.amax(x))
    if( y is not None):
        string += ',    ' + str(y.shape) + ' ' + str(np.amin(y)) + '->' + str(np.amax(y))
    print(string)

def getNormalizedData(dataset, validation_ratio):
    (train_images, train_labels), (_, _) = dataset.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]=
    dispInfo('Loaded', train_images, train_labels)
    
    XTraining, XValidation, YTraining, YValidation = train_test_split(train_images,train_labels,stratify=train_labels,test_size=validation_ratio)
    dispInfo('Training', XTraining, YTraining)
    dispInfo('Validation', XValidation, YValidation)
    return (XTraining, YTraining), (XValidation, YValidation)

def getMNISTdatasetClassification(validation_ratio = 1/6):
    return getNormalizedData(tf.keras.datasets.mnist, validation_ratio)
    
def getFashionMNISTdatasetClassification(validation_ratio = 1/6):
    return getNormalizedData(tf.keras.datasets.fashion_mnist, validation_ratio)

def makeDatasetFromData(data, batch_size = 256):
    # Batch and shuffle the data
    return tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0]).batch(batch_size)

def getMNISTdatasets(validation_ratio = 1/6, batch_size = 256):
    (xtrain, ytrain), (xvalidate, yvalidate) = getMNISTdatasetClassification(validation_ratio)
    train_dataset = makeDatasetFromData(xtrain, batch_size)
    validate_dataset = makeDatasetFromData(xvalidate, batch_size)
    return train_dataset, validate_dataset

def getFashionMNISTdatasets(validation_ratio = 1/6, batch_size = 256):
    (xtrain, ytrain), (xvalidate, yvalidate) = getFashionMNISTdatasetClassification(validation_ratio)
    train_dataset = makeDatasetFromData(xtrain, batch_size)
    validate_dataset = makeDatasetFromData(xvalidate, batch_size)
    return train_dataset, validate_dataset
