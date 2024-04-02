import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc,precision_recall_curve,confusion_matrix,accuracy_score
import matplotlib.pyplot as plt

def draw_curve(pre,tar):
    plt.figure(1)
    name=['dhtnn','dhtnn+xgb','morgan+xgb','dhtnn+morgan+xgb']
    for i in range(len(name)):
        fpr, tpr, threshold = roc_curve(tar, np.array(pre[i]))
        AUC = auc(fpr, tpr)
        pre_pro = [1 if i > 0.5 else 0 for i in pre[i]]
        tn, fp, fn, tp = confusion_matrix(tar, pre_pro).ravel()
        Sn = tp / (tp + fn)
        Sp = tn / (tn + fp)
        acc = accuracy_score(tar, pre_pro)
        plt.plot(fpr, tpr,
                 label=name[i]+'(area = %0.2f)' % AUC)


    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig('ROC_curve.png')



    # plt.figure(2)
    # for i in range(len(name)):
    #     precision, recall, threshold = precision_recall_curve(tar, pre[i])
    #     plt.plot(precision, recall,
    #              label=name[i])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend()
    # plt.title('PR curve')
    # plt.savefig('PR_curve')


