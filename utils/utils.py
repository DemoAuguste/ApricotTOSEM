import os
import settings
import numpy as np
from keras.datasets import cifar10, cifar100, mnist
import keras
from datetime import datetime

def load_dataset(dataset='cifar10', preprocessing=True, shuffle=True):
    """
    return: x_train, x_test, y_train, y_test
    """
    if dataset == 'cifar10':
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        if preprocessing:
            x_train, x_test = color_preprocessing(x_train, x_test)
        else:
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255
    elif dataset == 'cifar100':
        num_classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
    elif dataset == 'imagenet':  # imagenet64 dataset
        num_classes = 1000
        x_train = None
        y_train = None
        x_test = None
        y_test = None

        # load training dataset
        for i in range(10):
            temp_file_path = os.path.join(IMAGENET_DATASET_DIR, 'train_data_batch_{}'.format(i+1))
            temp_data = np.load(temp_file_path)
            temp = temp_data['data'].reshape(-1, 3, 64, 64)
            temp_x = np.rollaxis(temp, 1, 4)

            temp_y = np.array(temp_data['labels']) - 1
            temp_y = keras.utils.to_categorical(temp_y, num_classes)

            if x_train is None:
                x_train = temp_x
                y_train = temp_y
            else:
                x_train = np.concatenate(x_train, temp_x)
                y_train = np.concatenate(y_train, temp_y)

        # load test dataset
        test_file_path = os.path.join(IMAGENET_DATASET_DIR, 'val_data')
        test_data = np.load(test_file_path)

        x_test = test_data['data'].reshape(-1, 3, 64, 64)
        x_test = np.rollaxis(x_test, 1, 4)

        y_test = np.array(test_data['labels']) -1
        y_test = keras.utils.to_categorical(y_test, num_classes)

    # shuffle training dataset
    if shuffle:
        np.random.seed(settings.RANDOM_SEED)
        np.random.shuffle(x_train)
        np.random.seed(settings.RANDOM_SEED)
        np.random.shuffle(y_train)

    return x_train, x_test, y_train, y_test