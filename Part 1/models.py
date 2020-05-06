from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
import sklearn

def get_DecisionTree_roc(X_train, X_test, Y_train, Y_test):
    print("Building Decision Tree")
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    fpr_model, tpr_model, _ = roc_curve(Y_test, y_pred)
    print("Done")
    return fpr_model, tpr_model