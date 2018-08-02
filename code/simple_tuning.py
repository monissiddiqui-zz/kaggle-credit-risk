import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, make_scorer, roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint


def read_data(filename, sample_size: int = None):
    start = time.time()
    print('Reading training data...')
    df = pd.read_csv(filename)
    end = time.time()
    print(f'Finished reading training data in %.3f seconds' % (end - start))
    if sample_size is not None:
        df = df.sample(sample_size)
    return df


def get_target_distribution(df):
    rows = df.shape[0]
    target_dist_df = df[['SK_ID_CURR', 'TARGET']].groupby('TARGET').count()
    target_dist_df['PERCENT'] = target_dist_df['SK_ID_CURR'] * 100 / rows
    return target_dist_df


def transform_data(df):
    y = []  # collect targets
    data = []  # data (all columns except the target)

    target_col = 'TARGET'
    features = list([x for x in df.columns if x != target_col])

    for row in df.to_dict('records'):
        y.append(row[target_col])
        data.append({k: row[k] for k in features})

    return data, np.array(y)


def process_train_data(data_train):
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(data_train)

    # fill in nan values
    imputer = Imputer()
    X_train = imputer.fit_transform(X_train)

    # scaling data by columns so different features have roughly the same magnitude
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train, (vectorizer, imputer, scaler)  # need to reuse these preprocessors on test data


def process_test_data(data_test, processors):
    X_test = None
    for processor in processors:
        X_test = processor.transform(X_test if X_test is not None else data_test)

    return X_test


def process_data(data_train, data_val, y_train, y_val):
    print('Processing training data...')
    X_train, processors = process_train_data(data_train)

    print('Processing test data...')
    X_test = process_test_data(data_val, processors=processors)

    return X_train, X_test, y_train, y_val


def evaluate(y_test, y_score):
    roc_auc = roc_auc_score(y_test, y_score)
    print(f'auc: {roc_auc}')

    return roc_auc


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def get_sample_weight(y):
    sample_weights = []
    for yi in y:
        weight = 10.0 if yi else 1.0
        sample_weights.append(weight)

    return np.array(sample_weights)


def custom_auc(ground_truth, predictions):
    # I need only one column of predictions["0" and "1"]. You can get an error here
    # while trying to return both columns at once
    fpr, tpr, _ = roc_curve(ground_truth, predictions[:, 1], pos_label=1)
    return auc(fpr, tpr)


def parameter_tuning(X, y):
    sample_weights = get_sample_weight(y)

    model = GradientBoostingClassifier()

    # specify parameters and distributions to sample from
    param_dist = {
        # "n_estimators": [10 * i for i in range(5, 11)],
        "max_depth": [3, None],
        "max_features": sp_randint(1, 11),
        "criterion": ["friedman_mse", "mse", "mae"]}

    n_iter_search = 5
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       cv=5,
                                       scoring=make_scorer(custom_auc, greater_is_better=True,
                                                           needs_proba=True))

    start = time.time()
    random_search.fit(X, y, sample_weight=sample_weights)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    report(random_search.cv_results_)


if __name__ == '__main__':
    filename = '../data/application_train.csv'
    train_df = read_data(filename=filename, sample_size=10000)
    data, y = transform_data(train_df)

    print('Processing training data...')
    X, _ = process_train_data(data)

    print('Start parameter tuning')
    parameter_tuning(X, y)
