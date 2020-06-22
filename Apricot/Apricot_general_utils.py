from sklearn.metrics import classification_report
import numpy as np

def get_class_acc(model, xs, ys):
    y_pred = model.predict(xs)
    print(y_pred.shape)

    for i in range(len(y_pred)):
        max_value=max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value==y_pred[i][j]:
                y_pred[i][j]=1
            else:
                y_pred[i][j]=0

    print(classification_report(ys, y_pred, output_dict=True))
    return classification_report(ys, y_pred, output_dict=True)  
