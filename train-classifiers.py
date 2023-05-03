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
worklist = [
            ('random-undersampling.pickle', 'Random Undersampling'),
            ('tax-species-undersampling.pickle', 'Species Undersampling'),
            ('tax-family-undersampling.pickle', 'Family Undersampling'),
            ('tax-order-undersampling.pickle', 'Order Undersampling'),
            ('smote-random-1.pickle', 'SMOTE k=1 + Random Undersampling'),
            ('smote-random-2.pickle', 'SMOTE k=2 + Random Undersampling'),
            ('smote-random-3.pickle', 'SMOTE k=3 + Random Undersampling'),
            ('smote-random-4.pickle', 'SMOTE k=4 + Random Undersampling'),
            ('smote-random-5.pickle', 'SMOTE k=5 + Random Undersampling'),
           ]

csv_data = []
print('Data loaded.')

for filename, sampling in worklist:
    with open(test_data / filename, 'rb') as fp:
        train_test_data = pickle.load(fp)
    for test_family, train_df, test_df in tqdm(train_test_data, 
                                                desc=sampling, 
                                                total=len(train_test_data)):
        X_train = train_df.iloc[:, :-1]
        y_train = train_df.loc[:, 'interaction']
        X_test = test_df.reset_index().iloc[:, 2:-1]
        y_test = test_df.reset_index().loc[:, 'interaction']
        if exclude_features:
            X_train = X_train.loc[:, ~X_train.columns.isin(exclude_features)]
            X_test = X_test.loc[:, ~X_test.columns.isin(exclude_features)]
        xboost = XGBClassifier(n_estimators=20, use_label_encoder=False, n_jobs=8)
        rforest = RandomForestClassifier(n_jobs=-1)
        xboost.fit(X_train, y_train, 
                        eval_metric='logloss',
                        verbose=True)
        rforest.fit(X_train, y_train)
        interaction_xboost = xboost.predict_proba(X_test)
        interaction_rforest = rforest.predict_proba(X_test)
        csv_data.extend([
                                {'virus': virus,
                                'host': host,
                                'prediction': prediction,
                                'classificator': 'XGBoost',
                                'sampling': sampling
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
                                'sampling': sampling
                                }
                                for virus, host, prediction
                                in zip(
                                    [pair[0] for pair in test_df.index],
                                    [pair[1] for pair in test_df.index],
                                    [str(prediction[1]) for prediction in interaction_rforest])
                                ])

with open(output_path / f'{output_name}.csv', mode='w') as fp:
            writer = csv.DictWriter(
                fp, fieldnames=['virus', 'host', 'prediction', 'classificator', 'sampling'])
            writer.writeheader()
            writer.writerows(csv_data)
print("Done.")