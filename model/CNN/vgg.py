"""
Keras VGG models.
Reference: https://keras.io/zh/applications/#vgg16
"""
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

def get_vgg16(pretrained=False):
    weights = None
    if pretrained:
        weights = 'imagenet'
    base_model = VGG16(weights='imagenet') # None for random initialization or imagenet (pre-training ImageNet)
    return base_model

def get_vgg19(pretrained=False):
    weights = None
    if pretrained:
        weights = 'imagenet'
    base_model = VGG19(weights='imagenet') # None for random initialization or imagenet (pre-training ImageNet)
    return base_model
