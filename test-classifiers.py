#!/usr/bin/env python
import datetime
import json
import os
import pickle
from pathlib import Path
import csv
from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from tqdm import tqdm

from notebooks.virus.XGBoost import XGBoost
from notebooks.virus.RandomForest import RandomForest

user = os.environ['USER']
rf = RandomForest(f'/home/{user}/workspace/wirusy')
xgb = XGBoost(f'/home/{user}/workspace/wirusy')
test_data = Path(f'{rf.PROJECT_DIR}/test/data')
output = 'classifiers-test-%Y-%m-%d-T-%H-%M-%S'
timestamp_format = '%Y-%m-%d-T-%H-%M-%S'
now = datetime.datetime.now()
output_name = now.strftime(output)
timestamp = now.strftime(timestamp_format)
output_path = Path(f'{rf.PROJECT_DIR}/test')
os.makedirs(output_path, exist_ok=True)
exclude_features = ['crisprdetect_2mismatch_score', 'piler_2mismatch_score']
with open(test_data / 'random-undersampling.pickle', 'rb') as fp:
    random_undersampling = pickle.load(fp)
with open(test_data / 'tax-family-undersampling.pickle', 'rb') as fp:
    tax_family_undersampling = pickle.load(fp)

csv_data = []
print('Data loaded.')

for test_family, train_df, test_df in tqdm(random_undersampling, 
                                            desc='Random Undersampling', 
                                            nrows=len(random_undersampling)):
    y_train = train_df.reset_index().loc[:, 'interaction']
    y_test = test_df.reset_index().loc[:, 'interaction']
    X_train = train_df.reset_index().iloc[:, 2:-1]
    X_test = test_df.reset_index().iloc[:, 2:-1]
    if exclude_features:
        X_train = X_train.loc[:, ~X_train.columns.isin(exclude_features)]
        X_test = X_test.loc[:, ~X_test.columns.isin(exclude_features)]
    xboost = XGBClassifier(n_estimators=20, use_label_encoder=False, n_jobs=8)
    rforest = RandomForestClassifier(n_jobs=-1)
    xboost.fit(X_train, y_train, 
                    eval_metric='logloss',
                    verbose=True)
    rforest.fit(X_train, y_train)
    data_to_predict = test_df.reset_index().iloc[:, 2:-1]
    if exclude_features:
        data_to_predict = data_to_predict.loc[:, ~data_to_predict.columns.isin(exclude_features)]
    interaction_xboost = xboost.predict_proba(data_to_predict)
    interaction_rforest = rforest.predict_proba(data_to_predict)
    csv_data.extend([
                            {'virus': virus,
                             'host': host,
                             'prediction': prediction,
                             'classificator': 'XGBoost',
                             'undersampling': 'Random'
                             }
                            for virus, host, prediction
                            in zip(
                                [pair[0] for pair in test_df.index],
                                [pair[1] for pair in test_df.index],
                                [str(prediction[1]) for prediction in interaction_xboost])
                            ])
    csv_data.extend([
                            {'virus': virus,
                             'host': host,
                             'prediction': prediction,
                             'classificator': 'Random Forest',
                             'undersampling': 'Random'
                             }
                            for virus, host, prediction
                            in zip(
                                [pair[0] for pair in test_df.index],
                                [pair[1] for pair in test_df.index],
                                [str(prediction[1]) for prediction in interaction_rforest])
                            ])
print("Random undersampling done.")

for test_family, train_df, test_df in tqdm(tax_family_undersampling, 
                                            desc='Family Undersampling', 
                                            nrows=len(tax_family_undersampling)):
    y_train = train_df.reset_index().loc[:, 'interaction']
    y_test = test_df.reset_index().loc[:, 'interaction']
    X_train = train_df.reset_index().iloc[:, 2:-1]
    X_test = test_df.reset_index().iloc[:, 2:-1]
    if exclude_features:
        X_train = X_train.loc[:, ~X_train.columns.isin(exclude_features)]
        X_test = X_test.loc[:, ~X_test.columns.isin(exclude_features)]
    xboost = XGBClassifier(n_estimators=20, use_label_encoder=False, n_jobs=8)
    rforest = RandomForestClassifier(n_jobs=-1)
    xboost.fit(X_train, y_train, 
                    eval_metric='logloss',
                    verbose=True)
    rforest.fit(X_train, y_train)
    data_to_predict = test_df.reset_index().iloc[:, 2:-1]
    if exclude_features:
        data_to_predict = data_to_predict.loc[:, ~data_to_predict.columns.isin(exclude_features)]
    interaction_xboost = xboost.predict_proba(data_to_predict)
    interaction_rforest = rforest.predict_proba(data_to_predict)
    csv_data.extend([
                            {'virus': virus,
                             'host': host,
                             'prediction': prediction,
                             'classificator': 'XGBoost',
                             'undersampling': 'Family'
                             }
                            for virus, host, prediction
                            in zip(
                                [pair[0] for pair in test_df.index],
                                [pair[1] for pair in test_df.index],
                                [str(prediction[1]) for prediction in interaction_xboost])
                            ])
    csv_data.extend([
                            {'virus': virus,
                             'host': host,
                             'prediction': prediction,
                             'classificator': 'Random Forest',
                             'undersampling': 'Family'
                             }
                            for virus, host, prediction
                            in zip(
                                [pair[0] for pair in test_df.index],
                                [pair[1] for pair in test_df.index],
                                [str(prediction[1]) for prediction in interaction_rforest])
                            ])
print("Family undersampling done.")

with open(output_path / f'{output_name}.csv', mode='w') as fp:
            writer = csv.DictWriter(
                fp, fieldnames=['virus', 'host', 'prediction', 'classificator', 'undersampling'])
            writer.writeheader()
            writer.writerows(csv_data)
print("Done.")