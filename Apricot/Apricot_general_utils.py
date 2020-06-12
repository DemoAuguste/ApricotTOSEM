from sklearn.metrics import classification_report
import numpy as np

def get_class_acc(model, xs, ys):
    y_pred = model.predict(xs)
    print(y_pred.shape)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred.shape)
    print(y_pred[0])
    print(classification_report(ys, y_pred))
    
