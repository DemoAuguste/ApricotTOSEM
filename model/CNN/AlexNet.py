"""
Reference: https://engmrk.com/alexnet-implementation-using-keras/
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
from keras.optimizers import RMSprop, SGD
from keras.utils.np_utils import to_categorical
import numpy as np

import random


def alexnet():
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(3,3), strides = 4, padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2,2),  strides = 2, padding='same'))
    model.add(Conv2D(256, kernel_size=(5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),  strides = 2))
    model.add(Conv2D(384, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(384, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides = 2))
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))
    sgd = SGD(lr=.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = alexnet()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255 
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=20, verbose=1)

    index_list = [i for i in range(len(x_train))]
    batch_num = int(len(index_list) / 128)

    for i in range(50):
        random.shuffle(index_list)
        for j in range(batch_num): 
            xs = x_train[index_list[i*128:(i+1)*128]]
            ys = y_train[index_list[i*128:(i+1)*128]]
            for k in range(10):
                model.train_on_batch(xs, ys)
        print(model.evaluate(x_test, y_test))
    