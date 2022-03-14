import random
import pandas as pd

def leave_one_out_train_test_generator(df, virus_groups, host_json):
    groups = virus_groups.items()
    for test_group, test_viruses in groups:
        train_viruses = []
        for train_group, train_viruses_in_group in groups:
            if test_group == train_group: 
                continue
            train_viruses.extend(train_viruses_in_group)
        train_df = df.loc[train_viruses]
        test_df = df.loc[test_viruses]
        yield test_group, train_df, test_df

def leave_one_out_train_test_generator_random(df, virus_groups, host_json):
    groups = virus_groups.items()
    for test_group, test_viruses in groups:
        train_viruses = []
        for train_group, train_viruses_in_group in groups:
            if test_group == train_group: 
                continue
            train_viruses.extend(train_viruses_in_group)
        train_df = df.loc[train_viruses]
        train_df_positive = train_df[train_df['interaction'] == 1]
        train_df_negative = train_df[train_df['interaction'] == 0]
        negative_sample = train_df_negative.sample(n = len(train_df_positive))
        train_sample = pd.concat([train_df_positive, negative_sample])
        test_df = df.loc[test_viruses]
        yield test_group, train_sample, test_df

#   - [ ] Manipuluj negatywami na różnych poziomach taksonomiczych, wybieraj negatywy np daleko/blisko spokrewnione z pozytywem
#     - negatyw należy do innej rodziny, niż prawdziwy gospodarz - sprawdź. Dlaczego, bo nie widzi się w biologii żeby wirus atakował powyżej rodziny (family)
#     - sprawdź powyżej rodzaju (genus) jako pierwsze


def leave_one_out_train_test_generator_tax_group(df, virus_groups, host_json):
    groups = virus_groups.items()
    
    for test_group, test_viruses in groups:
        train_viruses = []
        for train_group, train_viruses_in_group in groups:
            if test_group == train_group: 
                continue
            train_viruses.extend(train_viruses_in_group)
        train_df = df.loc[train_viruses]
        train_df_positive = train_df[train_df['interaction'] == 1]
        negative_sample_indexes = []
        for virus, host in train_df_positive.index:
            true_host_family = host_json[host]['lineage_names'][-3]
            virus_hosts = train_df.loc[virus]
            virus_hosts_no_interaction = virus_hosts[virus_hosts['interaction'] == 0]
            virus_hosts_different_family = [host for host in virus_hosts_no_interaction.index
                                        if host_json[host]['lineage_names'][-3] != true_host_family]
            negative_sample_indexes.append([virus, random.choice(virus_hosts_different_family)])

        train_df_negative = train_df.loc[negative_sample_indexes]
        train_sample = pd.concat([train_df_positive, train_df_negative])
        test_df = df.loc[test_viruses]
        yield test_group, train_sample, test_df