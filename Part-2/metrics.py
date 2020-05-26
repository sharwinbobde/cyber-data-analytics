from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    recall_score,
    precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    auc,
)


def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 1]


def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 0]


def tp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 1]


def run_metrics(Y_test, y_pred):
    result = {}
    fpr_model, tpr_model, _ = roc_curve(Y_test, y_pred)
    result["false_pos_rate"] = fpr_model
    result["true_pos_rate"] = tpr_model

    precision, recall, _ = precision_recall_curve(Y_test, y_pred)
    result["precision_curve"] = precision
    result["recall_curve"] = recall
    result["auc_pr"] = auc(precision, precision)

    result["accuracy"] = accuracy_score(Y_test, y_pred)
    result["precision"] = precision_score(Y_test, y_pred)
    result["recall"] = recall_score(Y_test, y_pred)
    result["f1"] = f1_score(Y_test, y_pred)
    result["auc"] = auc(fpr_model, tpr_model)
    result["confusion"] = confusion_matrix(Y_test, y_pred)
    result["confusion_norm"] = confusion_matrix(Y_test, y_pred, normalize='all')

    result["tn"] = tn(Y_test, y_pred)
    result["tp"] = tp(Y_test, y_pred)
    result["fn"] = fn(Y_test, y_pred)
    result["fp"] = fp(Y_test, y_pred)

    return result
