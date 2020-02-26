"""
lenet5.
"""

import keras
import numpy as np
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2


# def lenet5():
#     this one is not lenet.
#     img_rows, img_cols = 32, 32
#     nb_classes = 10
#     kernel_size = (3, 3)
#     input_shape = (img_rows, img_cols, 3)

#     input_tensor = Input(shape=input_shape)
#     x = Convolution2D(64, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
#     x = Convolution2D(64, kernel_size, activation='relu', padding='same', name='block1_conv2')(x)
#     x = MaxPooling2D(pool_size=(2, 2), strides=2, name='block1_pool1')(x)

#     # block2
#     x = Convolution2D(128, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
#     x = Convolution2D(128, kernel_size, activation='relu', padding='same', name='block2_conv2')(x)
#     x = MaxPooling2D(pool_size=(2, 2), strides=2, name='block2_pool1')(x)

#     x = Flatten(name='flatten')(x)
#     x = Dense(256, activation='relu', name='dense1')(x)
#     x = Dense(256, activation='relu', name='dense2')(x)
#     x = Dense(nb_classes, name='before_softmax')(x)
#     x = Activation('softmax', name='predictions')(x)

#     model = Model(input_tensor, x)
#     sgd = keras.optimizers.SGD(lr=1, momentum=0.0, decay=0.0, nesterov=False)
#     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
#     return model

def lenet5():
    """
    lenet 5 (cifar 10)
    """
    batch_size    = 128
    epochs        = 200
    iterations    = 391
    num_classes   = 10
    weight_decay  = 0.0001
    
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), input_shape=(32,32,3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
    model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
    model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
    # sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    from keras.utils import to_categorical

    # img_rows, img_cols = 32, 32
    # nb_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)


    m = lenet5()
    m.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test, y_test))
