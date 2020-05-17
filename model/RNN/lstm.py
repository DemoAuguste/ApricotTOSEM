from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Bidirectional
from keras.layers import AveragePooling2D
from keras.layers import SimpleRNN, LSTM
from keras.models import model_from_json

def lstm(max_features):
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    return model


def bilistm(max_features):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=100))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    return model