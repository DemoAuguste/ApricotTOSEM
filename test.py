from model import *
from utils import *
import gensim
from settings import *

if __name__ == '__main__':
    model = build_networks('resnet20', num_classes=100)
    model.summary()
    model = build_networks('resnet20', num_classes=100)
