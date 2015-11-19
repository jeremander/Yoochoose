# Given a labeled data set (arbitrary features, binary response) and a filename with list of features to evaluate, constructs random forest model and compares the features

# Trains and tests random forest on the full Yoochoose data set. Saves off the model, test set probabilities, and report of classification statistics

import optparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def main():
    p = optparse.OptionParser()
    p.add_option('--load', '-L', default = False, action = 'store_true', help = 'load model from file')
    p.add_option('--features', '-f', default = 'features.txt', type = str, help = 'feature filename')
    p.add_option('--verbose', '-v', default = False, action = 'store_true', help = 'verbosity flag')
    p.add_option('--thresh', '-T', default = 0.5, type = float, help = 'probability threshold to classify True')
    p.add_option('--n_estimators', '-n', default = 100, type = int, help = 'number of random forest estimators')
    p.add_option('--seed', '-s', default = None, type = int, help = 'random seed')
    p.add_option('--jobs', '-j', default = -1, type = int, help = 'number of jobs (-1 if maximum)')
    opts, args = p.parse_args()

    model_filename = 'model%s.pickle' % ('' if opts.seed is None else str(opts.seed))

    np.random.seed(opts.seed)

    if opts.verbose:
        print("\nReading data set...")

    train = pd.read_csv('data/yoochoose-training_session_features.csv').append(pd.read_csv('data/yoochoose-dev_session_features.csv'))
    test = pd.read_csv('data/yoochoose-test_session_features.csv')

    if opts.load:
        rfc = pickle.load(open(model_filename, 'rb'))
        if opts.verbose:
            print("\nLoaded model from '%s'.\n" % model_filename)
    else:
        # set the random forest instance
        rfc = RandomForestClassifier(n_estimators = opts.n_estimators, n_jobs = opts.jobs)
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
        pickle.dump(rfc, open(model_filename, 'wb'))
        if opts.verbose:
            print("\nSaved model to '%s'.\n" % model_filename)

    # make predictions on the test data
    probs = rfc.predict_proba(test[rfc.input_features])[:, 1]
    probs_series = pd.Series(probs)
    probs_series.to_csv('test_probs%s' % ('' if opts.seed is None else str(opts.seed)), index = False)
    test_preds = (probs >= opts.thresh)
    conf_df = pd.crosstab(test[rfc.output_feature], test_preds, rownames = ['actual'], colnames = ['predicted'])
    conf_mat = np.asarray(conf_df)
    class_report = classification_report(test[rfc.output_feature], test_preds)
    s = "\nConfusion Matrix\n"
    s += str(conf_df) + '\n'
    s += "\nClassification Report\n"
    s += class_report + '\n'
    accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / float(np.sum(conf_mat))
    s += "Accuracy = %.3f%%\n" % (100. * accuracy)
    s += "\nFeature Importances\n"
    triples = [(i, rfc.input_features[i], rfc.feature_importances_[i]) for i in range(num_features)]
    triples.sort(key = lambda pair : pair[2], reverse = True)
    indices, features, importances = zip(*triples)
    for i in range(num_features):
        s += "%17s   %3d.%03d%%\n" % (features[i], int(100. * importances[i]), round(1000 * (100. * importances[i] - int(100. * importances[i]))))
    with open('test_report%s' % ('' if opts.seed is None else str(opts.seed)), 'w') as f:
        f.write(s)

if __name__ == "__main__":
    main() 