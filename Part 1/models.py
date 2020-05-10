from sklearn.tree import DecisionTreeClassifier
from metrics import run_metrics


class models:
    @staticmethod
    def DecisionTree(X_train, X_test, Y_train, Y_test, params, quiet=True):
        if not quiet:
            print("Building Decision Tree")
        model = DecisionTreeClassifier(**params, random_state=1337)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        result = run_metrics(Y_test, y_pred, y_proba)
        result["model"] = model
        if not quiet:
            print("Done")
        return result

    @staticmethod
    def NaiveBayes(X_train, X_test, Y_train, Y_test, params, quiet=True):
        from sklearn.naive_bayes import GaussianNB

        if not quiet:
            print("Building Gaussian Naive Bayes model")
        model = GaussianNB(**params)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        result = run_metrics(Y_test, y_pred, y_proba)
        result["model"] = model
        if not quiet:
            print("Done")
        return result

    @staticmethod
    def SVC(X_train, X_test, Y_train, Y_test, params, quiet=True):
        from sklearn.svm import SVC

        if not quiet:
            print("Building SVC model")
        model = SVC(**params, probability=True)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        result = run_metrics(Y_test, y_pred, y_proba)
        result["model"] = model
        if not quiet:
            print("Done")
        return result

    @staticmethod
    def try_all(X_train, X_test, Y_train, Y_test):
        results = {}
        results["NaiveBayes"] = models.NaiveBayes(X_train, X_test, Y_train, Y_test)
        results["DecisionTree"] = models.NaiveBayes(X_train, X_test, Y_train, Y_test)
        results["SVC"] = models.SVC(X_train, X_test, Y_train, Y_test)
        return results
