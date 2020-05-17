from model import *
from utils import *
import gensim
from settings import *

if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, X_test, y_test, max_len = load_dataset('treebank')
    print(len(X_train))
    print(len(X_valid))
    print(len(X_test))
    # model = build_networks('lstm', rnn_num_classes, (max_len, word2vec_len))
