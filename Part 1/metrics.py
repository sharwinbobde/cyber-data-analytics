from sklearn.metrics import roc_curve, precision_recall_curve, recall_score, precision_score, accuracy_score, f1_score

def run_metrics(Y_test, y_pred):
    result = {}
    fpr_model, tpr_model, _ = roc_curve(Y_test, y_pred)
    result['false_pos_rate'] = fpr_model
    result['true_pos_rate'] = tpr_model
    
    precision, recall, _ = precision_recall_curve(Y_test, y_pred)
    result['precision_curve'] = precision
    result['recall_curve'] = recall

    result['accuracy'] = accuracy_score(Y_test, y_pred)
    result['precision'] = precision_score(Y_test, y_pred)
    result['recall'] = recall_score(Y_test, y_pred)
    result['f1'] = recall_score(Y_test, y_pred)

    return result
    