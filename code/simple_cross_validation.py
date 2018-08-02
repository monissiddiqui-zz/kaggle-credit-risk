import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MaxAbsScaler


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


def split_data(data, y, n_folds: int = 5):
    # train-test split
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds)
    folds_data = []
    for (j, (train_index, test_index)) in enumerate(skf.split(data, y)):
        data_train = [data[i] for i in train_index]  # data is a list, so we need to loop over it
        data_test = [data[i] for i in test_index]
        y_train = y[train_index]  # y is a numpy array, so we can slice it by a list of index
        y_test = y[test_index]

        folds_data.append((data_train, data_test, y_train, y_test))

    return folds_data


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


def train_predict(X_train, X_test, y_train, y_test):
    # fit model
    model = LogisticRegression(class_weight='balanced')

    start = time.time()
    print(f'Fitting model on {X_train.shape[0]} samples...')
    model.fit(X_train, y_train)

    end = time.time()
    print('Finished model training in %.3f seconds.' % (end - start))

    # compute area under ROC
    # we need probabilities to do this
    y_score = model.decision_function(X_test)
    return y_score


def evaluate(y_test, y_score):
    roc_auc = roc_auc_score(y_test, y_score)

    return roc_auc


if __name__ == '__main__':
    filename = '../data/application_train.csv'
    train_df = read_data(filename=filename, sample_size=10000)
    data, y = transform_data(train_df)

    folds_data = split_data(data, y, n_folds=5)
    fold_aucs = []
    for (i, (data_train, data_val, y_train, y_val)) in enumerate(folds_data):
        print(f'Training and evaluating on fold {i}...')
        print(f'{len(data_train)} for training and {len(data_val)} for validation')

        X_train, y_train, X_val, y_val = process_data(data_train, data_val, y_train, y_val)
        y_score = train_predict(X_train, y_train, X_val, y_val)

        roc_auc = evaluate(y_val, y_score)
        print(f'Area under roc: {roc_auc}')
        print('=' * 50)
        print()

        fold_aucs.append(roc_auc)

    print(f'average roc_auc over {len(fold_aucs)} folds: {np.mean(fold_aucs)}')
