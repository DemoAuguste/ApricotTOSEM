from model import *
from utils import *
import gensim
from settings import *

if __name__ == '__main__':
    input_size = (32, 32, 3)
    model = build_networks('mobilenet', num_classes=10, input_size=input_size) 
    model.summary()
    model = build_networks('mobilenet', num_classes=100, input_size=input_size)
    model.summary()
