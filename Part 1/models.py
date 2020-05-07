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
    def NaiveBayes(X_train, X_test, Y_train, Y_test):
        from sklearn.naive_bayes import GaussianNB
        print("Building Gaussian Naive Bayes model")
        model = GaussianNB()
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)

        result = run_metrics(Y_test, y_pred)
        print("Done")
        return result

    @staticmethod
    def try_all(X_train, X_test, Y_train, Y_test):
        results = {}
        results['NaiveBayes'] = models.NaiveBayes(X_train, X_test, Y_train, Y_test)
        results['DecisionTree'] = models.NaiveBayes(X_train, X_test, Y_train, Y_test)
        return results