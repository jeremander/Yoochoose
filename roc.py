# Given a file listing true binary labels and another file listing predicted probabilities, constructs an ROC curve, plots it, and approximates the optimal threshold. Assumes data files are listed as rows of binary labels or probabilities.

import optparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, classification_report

def main():
    p = optparse.OptionParser()
    p.add_option('--labels', '-l', type = str, help = 'true binary label data filename')
    p.add_option('--probs', '-p', type = str, help = 'classifier prediction probability filename')
    opts, args = p.parse_args()

    y = np.asarray(pd.read_csv(opts.labels, header = None)[0], dtype = bool)
    num_true = sum(y)
    true_counts = np.array([num_true, len(y) - num_true])
    true_proportions = true_counts / len(y)
    probs = np.asarray(pd.read_csv(opts.probs, header = None)[0])

    def fscore(x, y):
        return 0.0 if (x * y == 0.0) else (2 * x * y) / (x + y)
    fpr, tpr, thresholds = roc_curve(y, probs)
    auc = roc_auc_score(y, probs)
    precision, recall, thresholds2 = precision_recall_curve(y, probs)
    thresholds3 = np.array(sorted(list(set(thresholds).intersection(set(thresholds2)))))
    df = pd.DataFrame(columns = ['threshold', 'tpr', 'fpr', 'precision', 'fscore'])
    df['threshold'] = thresholds3
    for i in range(len(df)):
        thresh = thresholds3[i]
        for j in range(len(thresholds)):
            if (thresholds[j] == thresh):
                tpr_val = tpr[j]
                df.set_value(i, 'tpr', tpr_val)
                df.set_value(i, 'fpr', fpr[j])
                break
        for j in range(len(thresholds2)):
            if (thresholds2[j] == thresh):
                df.set_value(i, 'precision', precision[j])
                df.set_value(i, 'fscore', fscore(tpr_val, precision[j]))
                break

    print("\nArea under ROC curve = %.3f\n" % auc)
    print("True counts:\nFalse  %d   %.3f\nTrue  %d   %.3f\n" % (true_counts[0], true_proportions[0], true_counts[1], true_proportions[1]))

    best_index = list(df['fscore']).index(max(df['fscore']))
    best_thresh = thresholds3[best_index]
    predictions = probs >= best_thresh
    conf_df = pd.crosstab(y, predictions, rownames = ['actual'], colnames = ['predicted'])
    class_report = classification_report(y, predictions)

    print("Best threshold = %.3f\n" % best_thresh)
    print("tpr = %.3f\nfpr = %.3f" % (df['tpr'][best_index], df['fpr'][best_index]))
    print("precision = %.3f" % df['precision'][best_index])
    print("f-score = %.3f\n" % df['fscore'][best_index])
    print(conf_df)
    print("")
    print(class_report)

    plt.plot(fpr, tpr)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC')
    plt.show()

if __name__ == "__main__":
    main()


