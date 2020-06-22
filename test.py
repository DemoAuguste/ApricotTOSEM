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

    model.fit(x_train, y_train, batch_size=128, epochs=2)

    # print(model.predict(x_train[:10]))
    
    # get_class_acc(model, x_test, y_test)

    # calculate the predicted probability of each class
    pred_prob_mat = model.predict(x_train)
    max_ind_mat = list(map(lambda x: x==max(x), pred_prob_mat)) * np.ones(shape=pred_prob_mat.shape)
    max_prob_mat = pred_prob_mat * max_ind_mat
    sum_prob_mat = np.sum(max_prob_mat, axis=0)
    class_sum_mat = np.sum(max_ind_mat, axis=0)
    class_prob_mat = sum_prob_mat / class_sum_mat
    
    print(class_prob_mat)