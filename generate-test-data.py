#!/usr/bin/env python

import json
import os
import pickle
from pathlib import Path

import pandas as pd

from notebooks.virus.cross_validation.undersampling import (
    leave_one_out_train_test_generator_random,
    leave_one_out_train_test_generator_tax_group)


def main():
    user = os.environ['USER']
    project_dir = Path(f'/home/{user}/workspace/wirusy')
    dataframe_path = project_dir / 'dataframe.csv'
    host_json = json.loads((project_dir / 'host.json').read_text())
    virus_groups = json.loads((project_dir / 'virus-groups.json').read_text())
    output_path = Path(f'{project_dir}/test/data')
    os.makedirs(output_path, exist_ok=True)
    df = pd.read_csv(dataframe_path, header=0, index_col=[0, 1])
    print('Data loaded.')

    random = []
    tax_group = []
    for test_family, train_df, test_df in leave_one_out_train_test_generator_random(df, virus_groups, host_json):
        random.append((test_family, train_df, test_df))
    print('Random undersampling done.')
    for test_family, train_df, test_df in leave_one_out_train_test_generator_tax_group(df, virus_groups, host_json):
        tax_group.append((test_family, train_df, test_df))
    print('Family undersampling done.')

    with open(output_path / 'random-undersampling.pickle', 'wb') as fp:
        pickle.dump(random, fp)
    with open(output_path / 'tax-family-undersampling.pickle', 'wb') as fp:
        pickle.dump(tax_group, fp)
    print('Done')


if __name__ == '__main__':
    main()
