import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MaxAbsScaler


def read_data(filename, sample_size: int = None):
    start = time.time()
    print(f'Reading data from {filename}...')
    df = pd.read_csv(filename)
    end = time.time()
    print(f'Finished reading data from {filename} in %.3f seconds' % (end - start))
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


def split_data(data, y):
    # train-test split
    from sklearn.model_selection import train_test_split

    data_train, data_test, y_train, y_test = train_test_split(data, y, train_size=0.8, stratify=y)
    print(f'data_train: {len(data_train)}')
    print(f'data_test: {len(data_test)}')

    return data_train, data_test, y_train, y_test


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
    print(f'auc: {roc_auc}')

    return roc_auc


def join_datasets():
    filename = '../data/application_train.csv'
    train_df = read_data(filename=filename)

    prev_app_filename = '../data/previous_application.csv'
    prev_app_df = read_data(filename=prev_app_filename)

    prev_agg = prev_app_df.groupby('SK_ID_CURR')
    prev_df = prev_agg.agg({'SK_ID_PREV': 'count', 'AMT_ANNUITY': 'sum'}).rename(columns={
        'SK_ID_PREV': 'PREV_APPS', 'AMT_ANNUITY': 'PREV_AMT_ANNUITY'})
    df = train_df.join(prev_df, on='SK_ID_CURR')
    df[['PREV_APPS', 'PREV_AMT_ANNUITY']] = df[['PREV_APPS', 'PREV_AMT_ANNUITY']].fillna(value=0)

    bureau_filename = '../data/bureau.csv'
    bureau_df = read_data(filename=bureau_filename)
    active_bureau_df = bureau_df[bureau_df['CREDIT_ACTIVE'] == 'Active']
    active_bureau_agg = active_bureau_df.groupby('SK_ID_CURR')
    active_bureau_agg_df = active_bureau_agg.agg(
        {'AMT_CREDIT_SUM_DEBT': 'sum', 'CNT_CREDIT_PROLONG': 'sum'})
    df = df.join(active_bureau_agg_df, on='SK_ID_CURR')
    df[['AMT_CREDIT_SUM_DEBT', 'CNT_CREDIT_PROLONG']] = df[['AMT_CREDIT_SUM_DEBT', 'CNT_CREDIT_PROLONG']].fillna(value=0)

    return df


if __name__ == '__main__':
    train_df = join_datasets()

    data, y = transform_data(train_df)
    data_train, data_val, y_train, y_val = split_data(data, y)

    X_train, X_val, y_train, y_val = process_data(data_train, data_val, y_train, y_val)

    y_score = train_predict(X_train, X_val, y_train, y_val)

    evaluate(y_val, y_score)
