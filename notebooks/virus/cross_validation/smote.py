import random
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline


def leave_one_out_train_test_generator_random_smote(df, virus_groups, host_json, tax_group, k_neighbors=5):
    groups = virus_groups.items()
    over = SMOTE(sampling_strategy=1, k_neighbors=k_neighbors)
    under = RandomUnderSampler(sampling_strategy=0.4)
    steps = [('u', under), ('o', over)]
    pipeline = Pipeline(steps=steps)
    for test_group, test_viruses in groups:
        train_viruses = []
        for train_group, train_viruses_in_group in groups:
            if test_group == train_group: 
                continue
            train_viruses.extend(train_viruses_in_group)
        train_df = df.loc[train_viruses]
        train_df_y = train_df.loc[:, 'interaction']
        train_df_X = train_df.iloc[:, :-1]
        train_df_X_resampled, train_df_y_resampled = pipeline.fit_resample(train_df_X, train_df_y)
        train_df_X_resampled['interaction'] = train_df_y_resampled
        train_df_resampled = train_df_X_resampled
        test_df = df.loc[test_viruses]
        yield test_group, train_df_resampled, test_df

def leave_one_out_train_test_generator_ENN_smote(df, virus_groups, host_json, tax_group, k_neighbors=5):
    groups = virus_groups.items()
    over = SMOTE(sampling_strategy=1, k_neighbors=k_neighbors)
    under = EditedNearestNeighbours()
    steps = [('u', under), ('o', over)]
    pipeline = Pipeline(steps=steps)
    for test_group, test_viruses in groups:
        train_viruses = []
        for train_group, train_viruses_in_group in groups:
            if test_group == train_group: 
                continue
            train_viruses.extend(train_viruses_in_group)
        train_df = df.loc[train_viruses]
        train_df_y = train_df.loc[:, 'interaction']
        train_df_X = train_df.iloc[:, :-1]
        train_df_X_resampled, train_df_y_resampled = pipeline.fit_resample(train_df_X, train_df_y)
        train_df_X_resampled['interaction'] = train_df_y_resampled
        train_df_resampled = train_df_X_resampled
        test_df = df.loc[test_viruses]
        yield test_group, train_df_resampled, test_df

def leave_one_out_train_test_generator_SMOTEENN(df, virus_groups, host_json, tax_group):
    groups = virus_groups.items()
    sme = SMOTEENN(sampling_strategy=0.4, n_jobs=-1)
    for test_group, test_viruses in groups:
        train_viruses = []
        for train_group, train_viruses_in_group in groups:
            if test_group == train_group: 
                continue
            train_viruses.extend(train_viruses_in_group)
        train_df = df.loc[train_viruses]
        train_df_y = train_df.loc[:, 'interaction']
        train_df_X = train_df.iloc[:, :-1]
        train_df_X_resampled, train_df_y_resampled = sme.fit_resample(train_df_X, train_df_y)
        train_df_X_resampled['interaction'] = train_df_y_resampled
        train_df_resampled = train_df_X_resampled
        test_df = df.loc[test_viruses]
        yield test_group, train_df_resampled, test_df