import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def model_selection_plot(models: list, X_train, y_train, X_test, y_test) -> 'plt.plot':
    """
    Return a plot of ROC curve
    """
    plt.figure(figsize=(10, 10))
    for model in models:
        model.fit(X_train, y_train)
        pred_scr = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, pred_scr)
        roc_auc = roc_auc_score(y_test, pred_scr)
        md = str(model)
        md = md[:md.find('(')]
        plt.plot(fpr, tpr, label='ROC fold %s (auc = %0.2f)' % (md, roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
