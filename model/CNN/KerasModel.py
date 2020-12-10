# from keras.applications import mobilenet, mobilenet_v2, inception_v3
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.metrics import top_k_categorical_accuracy
from .densenet_mnist import DenseNet
import functools

def top_k_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


# def Inception_v3(input_tensor, num_classses, weights=None):  # weights='imagenet'
#     """
#     Notes: the minimal size of input is (150, 150, 3)
#     """
#     model = inception_v3.InceptionV3(input_tensor=input_tensor, weights=weights, include_top=True, classes=num_classses)
#     sgd = optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
#     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#     return model


def build_mobilenet(input_tensor, num_classses, k=1, weights=None, gpus=None):
    model = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights=weights, input_tensor=input_tensor, pooling=None, classes=num_classses)
    sgd = optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    if gpus:
        parallel_model = multi_gpu_model(model, gpus=gpus)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return parallel_model

    else:
        if k == 1:
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', top_k_acc])
        return model


def build_mobilenet_v2(input_tensor, num_classses, k=1, weights=None, gpus=None):
    model = MobileNetV2(input_shape=None, alpha=1.0, include_top=True, weights=weights, input_tensor=input_tensor, pooling=None, classes=num_classses)
    sgd = optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    if gpus:
        parallel_model = multi_gpu_model(model,gpus=gpus)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return parallel_model
    else:
        if k == 1:
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', top_k_acc])
        return model


def build_densenet(input_tensor, num_classses, k=1, weights=None, gpus=None):
    model = DenseNet121(include_top=True, weights=weights, input_tensor=input_tensor, input_shape=None, pooling=None, classes=num_classses)
    sgd = optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    if gpus:
        parallel_model = multi_gpu_model(model,gpus=gpus)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return parallel_model
    else:
        if k == 1:
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', top_k_acc])
        return model

def build_densenet_mnist(input_tensor, num_classes):
    # print(input_tensor, num_classes)
    densenet = DenseNet((28, 28, 1), nb_classes=num_classes, depth=25)
    model = densenet.build_model()
    model_optimizer = optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=model_optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    pass

    # from keras.layers import Input

    # input_tensor = Input(shape=(32, 32, 3))
    # mobile = build_mobilenet(input_tensor, 100)
    # x_train, x_test, y_train, y_test = load_dataset('cifar100')