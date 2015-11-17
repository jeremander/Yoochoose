# Given a labeled data set (arbitrary features, binary response) and a filename with list of features to evaluate, constructs random forest model and compares the features

import optparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main():
    p = optparse.OptionParser()
    p.add_option('--model', '-m', default = 'model.pickle', type = str, help = 'model filename')
    p.add_option('--load', '-L', default = False, action = 'store_true', help = 'load model from file')
    p.add_option('--features', '-f', default = 'features.txt', type = str, help = 'feature filename')
    p.add_option('--data', '-d', default = 'data.csv', type = str, help = 'marked data filename')
    p.add_option('--verbose', '-v', default = False, action = 'store_true', help = 'verbosity flag')
    p.add_option('--thresh', '-T', default = 0.5, type = float, help = 'probability threshold to classify True')
    p.add_option('--n_estimators', '-n', default = 100, type = int, help = 'number of random forest estimators')
    p.add_option('--test_fraction', '-t', default = 0.25, type = float, help = 'fraction of data to use for testing')
    p.add_option('--seed', '-s', default = None, type = int, help = 'random seed')
    opts, args = p.parse_args()

    np.random.seed(opts.seed)

    if opts.verbose:
        print("\nReading marked data from %s..." % opts.data)

    # establish data frame
    df = pd.read_csv(opts.data)
    n_lines = len(df)

    # choose test set as random test_fraction of data, leaving the remainder for training
    n_test = int(opts.test_fraction * n_lines)
    if opts.verbose:
        print("Read %d lines of data -> %d lines (training), %d lines (test)" % (n_lines, n_lines - n_test, n_test))
    test_subset = np.random.permutation(range(n_lines))[:n_test]
    is_train = np.ones(n_lines, dtype = bool)
    for i in test_subset:
        is_train[i] = False

    # establish training and test sets
    train, test = df[is_train], df[~is_train]

    if opts.load:
        rfc = pickle.load(open(opts.model, 'rb'))
    else:
        # set the random forest instance
        rfc = RandomForestClassifier(n_estimators = opts.n_estimators, n_jobs = -1)
        # set list of features (all the uncommented features above dotted line in feature file; leading/trailing whitespace is stripped
        with open(opts.features, 'r') as f:
            lines = f.readlines()
        line_starts_with_dash = [(line[0] == '-') for line in lines]
        assert (line_starts_with_dash.count(True) == 1), "Feature file must have a single dashed line separating input/output features."
        dashed_line_index = line_starts_with_dash.index(True)
        rfc.input_features = []
        for i in range(dashed_line_index):
            feature = lines[i].partition('#')[0].strip()
            if (len(feature) > 0):
                rfc.input_features.append(feature)
        output_features = []
        for i in range(dashed_line_index + 1, len(lines)):
            feature = lines[i].partition('#')[0].strip()
            if (len(feature) > 0):
                output_features.append(feature)
        assert (len(output_features) == 1), "Feature file must have exactly one output feature."
        rfc.output_feature = output_features[0]

    num_features = len(rfc.input_features)
    assert (num_features > 0), "Feature file must have at least one input feature."
    X = train[rfc.input_features]
    y = train[rfc.output_feature]

    if (not opts.load):
        # train the forest
        if opts.verbose:
            print("\nTraining %d random forests..." % opts.n_estimators)
        rfc.fit(X, y)
        # save off the model
        pickle.dump(rfc, open(opts.model, 'wb'))
        if opts.verbose:
            print("\nSaved model to '%s'.\n" % opts.model)

    # make predictions on the test data
    probs = rfc.predict_proba(test[rfc.input_features])[:, 1]
    test_preds = (probs >= opts.thresh)
    conf_df = pd.crosstab(test[rfc.output_feature], test_preds, rownames = ['actual'], colnames = ['predicted'])
    conf_mat = np.asarray(conf_df)
    class_report = classification_report(test[rfc.output_feature], test_preds)
    print("\nConfusion Matrix")
    print(conf_df)
    print("\nClassification Report")
    print(class_report)
    accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / float(np.sum(conf_mat))
    print("Accuracy = %.3f%%" % (100. * accuracy))
    print("\nFeature Importances")
    triples = [(i, rfc.input_features[i], rfc.feature_importances_[i]) for i in range(num_features)]
    triples.sort(key = lambda pair : pair[2], reverse = True)
    indices, features, importances = zip(*triples)
    for i in range(num_features):
        print("%17s   %3d.%03d%%" % (features[i], int(100. * importances[i]), round(1000 * (100. * importances[i] - int(100. * importances[i])))))
    stds = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis = 0)
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances, color = 'r', yerr = [stds[i] for i in indices], align = 'center')
    plt.xticks(range(X.shape[1]), features, rotation = 'vertical')
    plt.xlim([-1, X.shape[1]])
    fig = plt.gcf()
    fig.subplots_adjust(bottom = 0.25)
    plt.show()


if __name__ == "__main__":
    main()