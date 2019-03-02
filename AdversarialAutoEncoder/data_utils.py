import tensorflow as tf
import numpy as np


def getMNISTdatasets(validation_ratio = 1/6, batch_size = 256):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    print('images shape: ', train_images.shape, ', min,max: ', np.amin(train_images), ',', np.amax(train_images))

    NUM_IMAGES = train_images.shape[0]

    NUM_VALIDATION = int(NUM_IMAGES * validation_ratio)
    NUM_TRAINING = NUM_IMAGES - NUM_VALIDATION
    print('Using', NUM_TRAINING, 'training images and', NUM_VALIDATION, 'validation images')

    train = train_images[:NUM_TRAINING, :, :, :]
    validate = train_images[NUM_TRAINING:, :, :, :]

    print('train shape:', train.shape)
    print('validate shape:', validate.shape)

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train).shuffle(train.shape[0]).batch(batch_size)
    validate_dataset = tf.data.Dataset.from_tensor_slices(validate).shuffle(validate.shape[0]).batch(batch_size)
    print('train_dataset: ', train_dataset)
    print('validate_dataset: ', validate_dataset)

    return train_dataset, validate_dataset
