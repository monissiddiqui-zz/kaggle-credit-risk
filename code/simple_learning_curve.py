import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MaxAbsScaler


def read_data(filename):
    start = time.time()
    print('Reading training data...')
    df = pd.read_csv(filename)
    end = time.time()
    print(f'Finished reading training data in %.3f seconds' % (end - start))
    return df


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
    for (j, (train_index, val_index)) in enumerate(skf.split(data, y)):
        data_train = [data[i] for i in train_index]  # data is a list, so we need to loop over it
        data_val = [data[i] for i in val_index]
        y_train = y[train_index]  # y is a numpy array, so we can slice it by a list of index
        y_val = y[val_index]

        folds_data.append((data_train, data_val, y_train, y_val))

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
    X_train, processors = process_train_data(data_train)
    X_val = process_test_data(data_val, processors=processors)

    return X_train, X_val, y_train, y_val


def train_predict(X_train, X_val, y_train, y_val):
    # fit model
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    # compute area under ROC
    # we need probabilities to do this
    y_score = model.decision_function(X_val)
    return y_score


def evaluate(y_test, y_score):
    roc_auc = roc_auc_score(y_test, y_score)

    return roc_auc


def get_roc_auc_on_sample_size(folds_data, sample_size: int):
    fold_aucs = []
    for (i, (data_train, data_val, y_train, y_val)) in enumerate(folds_data):
        X_train, X_val, y_train, y_val = process_data(data_train, data_val, y_train, y_val)
        # resample the data
        X_train = X_train[: sample_size]
        y_train = y_train[: sample_size]
        y_score = train_predict(X_train, X_val, y_train, y_val)

        roc_auc = evaluate(y_val, y_score)
        print(f'\tFold {i}\t{roc_auc}')
        fold_aucs.append(roc_auc)

    return np.mean(fold_aucs)


if __name__ == '__main__':
    filename = '../data/application_train.csv'
    train_df = read_data(filename=filename)

    print('Transforming data...')
    data, y = transform_data(train_df)

    print('Splitting data...')
    folds_data = split_data(data, y, n_folds=5)

    roc_aucs = []
    sample_sizes = []
    for i in range(1, 6):
        sample_size = int(i * 10000)
        print(f'Learning with {sample_size} samples...')
        roc_auc = get_roc_auc_on_sample_size(folds_data, sample_size=sample_size)

        print(f'AUROC on {sample_size} samples: {roc_auc}')
        sample_sizes.append(sample_size)
        roc_aucs.append(roc_auc)
