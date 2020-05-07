from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sklearn

from metrics import run_metrics

class models:

    @staticmethod
    def DecisionTree(X_train, X_test, Y_train, Y_test):
        print("Building Decision Tree")
        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)

        result = run_metrics(Y_test, y_pred)
        print("Done")
        return result
    
    @staticmethod
    def SVM(X_train, X_test, Y_train, Y_test):
        print("Building SVM")
        model = SVC()
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)

        result = run_metrics(Y_test, y_pred)
        print("Done")
        return result