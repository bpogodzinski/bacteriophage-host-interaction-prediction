#!/usr/bin/env python
import datetime
import json
import os
import pickle
from pathlib import Path

import pandas as pd

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
output_path = f'{rf.PROJECT_DIR}/test'
os.makedirs(output_path, exist_ok=True)
exclude_features = ['crisprdetect_2mismatch_score', 'piler_2mismatch_score']
with open(test_data / 'random-undersampling.pickle', 'rb') as fp:
    random_undersampling = pickle.load(fp)
with open(test_data / 'tax-family-undersampling.pickle', 'rb') as fp:
    tax_family_undersampling = pickle.load(fp)

dict_dataframe = {
    'virus' : [],
    'host' : [],
    'prediction' : [],
    'classificator' : [],
    'undersampling' : []
}

print('Data loaded.')

# for test_family, train_df, test_df in 