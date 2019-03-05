import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def dispInfo(name, x, y=None):
    string = name + ': ' + str(x.shape) + ' ' + str(np.amin(x)) + '->' + str(np.amax(x))
    if( y is not None):
        string += ',    ' + str(y.shape) + ' ' + str(np.amin(y)) + '->' + str(np.amax(y))
    print(string)

def getNormalizedData(all_images, all_labels, validation_ratio):
    all_images = all_images.reshape(all_images.shape[0], 28, 28, 1).astype('float32')
    #all_images = (all_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    all_images = all_images / 255.  # Normalize the images to [0, 1]
    dispInfo('Loaded', all_images, all_labels)
    
    XTraining, XValidation, YTraining, YValidation = train_test_split(all_images,all_labels,stratify=all_labels,test_size=validation_ratio)
    dispInfo('Training', XTraining, YTraining)
    dispInfo('Validation', XValidation, YValidation)
    return (XTraining, YTraining), (XValidation, YValidation)


def getMNISTdatasetClassification(validation_ratio = 1/6):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    return getNormalizedData(train_images, train_labels, validation_ratio)
    
def getFashionMNISTdatasetClassification(validation_ratio = 1/6):
    (train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    return getNormalizedData(train_images, train_labels, validation_ratio)

def makeDatasetFromData(data, batch_size = 256):
    # Batch and shuffle the data
    #return tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0]).batch(batch_size)
    return tf.data.Dataset.from_tensor_slices(data).batch(batch_size)

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

def loadCSVImageDataset(fileName):
    all_data = np.loadtxt(fileName,delimiter=',', skiprows=1)
    images = all_data[:, 1:]
    labels = all_data[:, :1]
    reshaped = np.reshape(images, (images.shape[0], 28, 28, 1))
    return getNormalizedData(reshaped, labels, 1/6)

