"""
main function of apricot+ and apricot+ lite revision.
"""

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.layers import Input
from datetime import datetime
import os
from utils.utils import logger

def apricot(model, model_weights_dir):
    submodel_dir = os.path.join(model_weights_dir, 'submodels')
    fixed_weights_path = os.path.join(model_weights_dir, 'fixed.h5')
    log_path = os.path.join(model_weights_dir, 'log.txt')

    print('----------original model----------')
    
    


def apricot_lite():
    pass