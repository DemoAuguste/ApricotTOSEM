from model import *
from utils import *
# import gensim
from settings import *
from Apricot import *

if __name__ == '__main__':
    input_size = (32, 32, 3)
    model = build_networks('resnet20', num_classes=10, input_size=input_size) 
    # model.summary()
    # model = build_networks('densenet', num_classes=100, input_size=input_size)
    # model.summary()
    

    x_train, x_test, y_train, y_test = load_dataset('cifar10')

    model.fit(x_train, y_train, batch_size=128, epochs=1)

    print(model.predict(x_train[:10]))
    
    # get_class_acc(model, x_test, y_test)
