from sklearn.metrics import classification_report

def get_class_acc(model, xs, ys):
    y_pred = model.predict(xs)
    print(classification_report(ys, y_pred))
    
