#!/usr/bin/env python

import json
import os
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from notebooks.virus.cross_validation.undersampling import (
    TaxGroup,
    leave_one_out_train_test_generator_random,
    leave_one_out_train_test_generator_tax_group)

from notebooks.virus.cross_validation.smote import (
    leave_one_out_train_test_generator_random_smote,
    leave_one_out_train_test_generator_ENN_smote,
    leave_one_out_train_test_generator_SMOTEENN)

VIRUS_GROUPS = 48

def main():
    user = os.environ['USER']
    project_dir = Path(f'/home/{user}/workspace/wirusy')
    dataframe_path = project_dir / 'dataframe.csv'
    host_json = json.loads((project_dir / 'host.json').read_text())
    virus_groups = json.loads((project_dir / 'virus-groups.json').read_text())
    output_path = Path(f'{project_dir}/test/data')
    os.makedirs(output_path, exist_ok=True)
    df = pd.read_csv(dataframe_path, header=0, index_col=[0, 1])
    random = []
    species_group = []
    genus_group = []
    family_group = []
    order_group = []
    smote_random_group_1 = []
    smote_random_group_2 = []
    smote_random_group_3 = []
    smote_random_group_4 = []
    smote_random_group_5 = []
    smote_enn_group_1 = []
    smote_enn_group_2 = []
    smote_enn_group_3 = []
    smote_enn_group_4 = []
    smote_enn_group_5 = []

    SMOTEENN_list = []
    
    print('Data loaded.')

    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_random(df, virus_groups, host_json),
    #                                             desc="Random undersampling",
    #                                             total=VIRUS_GROUPS,):
    #     random.append((test_family, train_df, test_df))
    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_tax_group(df, virus_groups, host_json, TaxGroup.SPECIES),
    #                                             desc="Species undersampling",
    #                                             total=VIRUS_GROUPS):
    #     species_group.append((test_family, train_df, test_df))
    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_tax_group(df, virus_groups, host_json, TaxGroup.GENUS),
    #                                             desc="Genus undersampling",
    #                                             total=VIRUS_GROUPS):
    #     genus_group.append((test_family, train_df, test_df))
    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_tax_group(df, virus_groups, host_json, TaxGroup.FAMILY),
    #                                             desc="Family undersampling",
    #                                             total=VIRUS_GROUPS):
    #     family_group.append((test_family, train_df, test_df))
    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_tax_group(df, virus_groups, host_json, TaxGroup.ORDER),
    #                                             desc="Order undersampling",
    #                                             total=VIRUS_GROUPS):
    #     order_group.append((test_family, train_df, test_df))
    
    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_random_smote(df, virus_groups, host_json, TaxGroup.ORDER),
    #                                             desc="Random Undersampling + SMOTE k=5",
    #                                             total=VIRUS_GROUPS):
    #     smote_random_group_5.append((test_family, train_df, test_df))
    # with open(output_path / 'smote-random-5.pickle', 'wb') as fp:
    #     pickle.dump(smote_random_group_5, fp)
    # del smote_random_group_5


    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_random_smote(df, virus_groups, host_json, TaxGroup.ORDER, k_neighbors=4),
    #                                             desc="Random Undersampling + SMOTE k=4",
    #                                             total=VIRUS_GROUPS):
    #     smote_random_group_4.append((test_family, train_df, test_df))
    # with open(output_path / 'smote-random-4.pickle', 'wb') as fp:
    #     pickle.dump(smote_random_group_4, fp)
    # del smote_random_group_4

    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_random_smote(df, virus_groups, host_json, TaxGroup.ORDER, k_neighbors=3),
    #                                             desc="Random Undersampling + SMOTE k=3",
    #                                             total=VIRUS_GROUPS):
    #     smote_random_group_3.append((test_family, train_df, test_df))
    # with open(output_path / 'smote-random-3.pickle', 'wb') as fp:
    #     pickle.dump(smote_random_group_3, fp)
    # del smote_random_group_3

    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_random_smote(df, virus_groups, host_json, TaxGroup.ORDER, k_neighbors=2),
    #                                             desc="Random Undersampling + SMOTE k=2",
    #                                             total=VIRUS_GROUPS):
    #     smote_random_group_2.append((test_family, train_df, test_df))
    # with open(output_path / 'smote-random-2.pickle', 'wb') as fp:
    #     pickle.dump(smote_random_group_2, fp)
    # del smote_random_group_2

    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_random_smote(df, virus_groups, host_json, TaxGroup.ORDER, k_neighbors=1),
    #                                             desc="Random Undersampling + SMOTE k=1",
    #                                             total=VIRUS_GROUPS):
    #     smote_random_group_1.append((test_family, train_df, test_df))
    # with open(output_path / 'smote-random-1.pickle', 'wb') as fp:
    #     pickle.dump(smote_random_group_1, fp)
    # del smote_random_group_1

    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_ENN_smote(df, virus_groups, host_json, TaxGroup.ORDER),
    #                                             desc="ENN Undersampling + SMOTE k=5",
    #                                             total=VIRUS_GROUPS):
    #     smote_enn_group_5.append((test_family, train_df, test_df))
    # with open(output_path / 'smote-enn-5.pickle', 'wb') as fp:
    #     pickle.dump(smote_enn_group_5, fp)
    # del smote_enn_group_5

    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_ENN_smote(df, virus_groups, host_json, TaxGroup.ORDER, k_neighbors=4),
    #                                             desc="ENN Undersampling + SMOTE k=4",
    #                                             total=VIRUS_GROUPS):
    #     smote_enn_group_4.append((test_family, train_df, test_df))
    # with open(output_path / 'smote-enn-4.pickle', 'wb') as fp:
    #     pickle.dump(smote_enn_group_4, fp)
    # del smote_enn_group_4

    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_ENN_smote(df, virus_groups, host_json, TaxGroup.ORDER, k_neighbors=3),
    #                                             desc="ENN Undersampling + SMOTE k=3",
    #                                             total=VIRUS_GROUPS):
    #     smote_enn_group_3.append((test_family, train_df, test_df))
    # with open(output_path / 'smote-enn-3.pickle', 'wb') as fp:
    #     pickle.dump(smote_enn_group_3, fp)
    # del smote_enn_group_3

    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_ENN_smote(df, virus_groups, host_json, TaxGroup.ORDER, k_neighbors=2),
    #                                             desc="ENN Undersampling + SMOTE k=2",
    #                                             total=VIRUS_GROUPS):
    #     smote_enn_group_2.append((test_family, train_df, test_df))
    # with open(output_path / 'smote-enn-2.pickle', 'wb') as fp:
    #     pickle.dump(smote_enn_group_2, fp)
    # del smote_enn_group_2

    # for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_ENN_smote(df, virus_groups, host_json, TaxGroup.ORDER, k_neighbors=1),
    #                                             desc="ENN Undersampling + SMOTE k=1",
    #                                             total=VIRUS_GROUPS):
    #     smote_enn_group_1.append((test_family, train_df, test_df))
    # with open(output_path / 'smote-enn-1.pickle', 'wb') as fp:
    #     pickle.dump(smote_enn_group_1, fp)
    # del smote_enn_group_1

    for test_family, train_df, test_df in tqdm(leave_one_out_train_test_generator_SMOTEENN(df, virus_groups, host_json, TaxGroup.ORDER),
                                                desc="SMOTEENN",
                                                total=VIRUS_GROUPS):
        SMOTEENN_list.append((test_family, train_df, test_df))
    with open(output_path / 'smote-enn.pickle', 'wb') as fp:
        pickle.dump(SMOTEENN_list, fp)
    del SMOTEENN_list

    # with open(output_path / 'random-undersampling.pickle', 'wb') as fp:
    #     pickle.dump(random, fp)
    # with open(output_path / 'tax-species-undersampling.pickle', 'wb') as fp:
    #     pickle.dump(species_group, fp)
    # with open(output_path / 'tax-genus-undersampling.pickle', 'wb') as fp:
    #     pickle.dump(genus_group, fp)
    
    # with open(output_path / 'tax-family-undersampling.pickle', 'wb') as fp:
    #     pickle.dump(family_group, fp)
    # with open(output_path / 'tax-order-undersampling.pickle', 'wb') as fp:
    #     pickle.dump(order_group, fp)
    print('Done')


if __name__ == '__main__':
    main()
