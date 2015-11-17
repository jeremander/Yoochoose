# Given a labeled data set (arbitrary features, binary response) and a filename with list of features to evaluate, constructs random forest model and compares the features

import optparse
import pandas as pandas
import numpy as numpy
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main():
    p = optparse.OptionParser()
    p.add_option('--output', '-o', default = 'model.pickle', type = str, help = 'output filename')
    p.add_option('--features', '-f', default = 'features.txt', type = str, help = 'feature filename')
    p.add_option('--data', '-d', default = 'data.csv', type = str, help = 'marked data filename')
    p.add_option('--verbose', '-v', default = False, action = 'store_true', help = 'verbosity flag')
    p.add_option('--thresh', '-T', default = 0.5, type = float, help = 'probability threshold to classify True')
    p.add_option('--n_estimators', '-n', default = 100, type = int, help = 'number of random forest estimators')
    p.add_option('--test_fraction', '-t', default = 0.25, type = float, help = 'fraction of data to use for testing')
    p.add_option('--seed', '-s', default = None, type = int, help = 'random seed')

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

    # set the random forest instance
    rfc = RandomForestClassifier(n_estimators = opts.n_estimators)

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
    num_features = len(rfc.input_features)
    assert (num_features > 0), "Feature file must have at least one input feature."
    output_features = []
    for i in range(dashed_line_index + 1, len(lines)):
        feature = lines[i].partition('#')[0].strip()
        if (len(feature) > 0):
            output_features.append(feature)
    assert (len(output_features) == 1), "Feature file must have exactly one output feature."
    rfc.output_feature = output_features[0]

    # train the forest
    if opts.verbose:
        print("\nTraining %d random forests..." % opts.n_estimators)
    X = train[rfc.input_features]
    y = train[rfc.output_feature]
    rfc.fit(X, y)

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
    pairs = [(rfc.input_features[i], rfc.feature_importances_[i]) for i in range(num_features)]
    pairs.sort(key = lambda pair : pair[1], reverse = True)
    for i in range(num_features):
        print("%17s   %3d.%03d%%" % (pairs[i][0], int(100. * pairs[i][1]), round(1000 * (100. * pairs[i][1] - int(100. * pairs[i][1])))))

    # save off the model
    pickle.dump(rfc, open(opts.output, 'wb'))
    if opts.verbose:
        print("\nSaved model to '%s'.\n" % opts.output)


if __name__ == "__main__":
    main()