import csv
import json
import pickle
from pathlib import Path
from random import choice
from typing import Any, Callable, Dict, Generator, List, Tuple

import numpy as np
import pandas as pd
import plotnine as p9
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

class XGBoost:

    def __init__(self, project_dir: str) -> None:

        self.PROJECT_DIR = Path(project_dir)
        self.DATAFRAME = self.PROJECT_DIR / 'dataframe.csv'
        self.HOST_JSON = self.PROJECT_DIR / 'host.json'
        self.VIRUS_JSON = self.PROJECT_DIR / 'virus.json'
        self.VIRUS_GROUP_JSON = self.PROJECT_DIR / 'virus-groups.json'
        self.FEATURES_DIR = self.PROJECT_DIR / 'features'
        self.HOST_DIR = self.PROJECT_DIR / 'data' / 'host'
        self.VIRUS_DIR = self.PROJECT_DIR / 'data' / 'virus'

        assert self.PROJECT_DIR.exists() \
            and self.HOST_JSON.exists() \
            and self.VIRUS_JSON.exists() \
            and self.VIRUS_GROUP_JSON.exists() \
            and self.FEATURES_DIR.exists() \
            and self.HOST_DIR.exists() \
            and self.VIRUS_DIR.exists()

        with open(self.VIRUS_GROUP_JSON, mode='r') as fp:
            self.virus_groups = json.load(fp)
        self.df = pd.read_csv(self.DATAFRAME, header=0, index_col=[0, 1])

    def train(self,
              df: pd.DataFrame,
              virus_groups: Dict[str, List[str]], \
              # Callable input
              sampling_function: Callable[[pd.DataFrame, Dict[str, List[str]]], \
                                          # Callable output
                                          Generator[
                  # Yielding values
                  Tuple[str, pd.DataFrame, pd.DataFrame], \
                  None, None]], \
              exclude_features=None, \
              **kwargs) -> List[Tuple[str, RandomForestClassifier]]:
        """Train classifiers

        Args:
            df (pd.DataFrame): full dataframe of host and viruses.
            virus_groups (Dict[str, List[str]]): Viruses grouped by family rank.
            sampling_function (Callable): undersampling function
            **kwargs forwarded to RandomForestClassifier
        """
        results_models = []
        host_json = json.loads(self.HOST_JSON.read_text())
        for test_family, train_df, test_df in sampling_function(df, virus_groups, host_json):
            y_train = train_df.reset_index().loc[:, 'interaction']
            y_test = test_df.reset_index().loc[:, 'interaction']
            X_train = train_df.reset_index().iloc[:, 2:-1]
            X_test = test_df.reset_index().iloc[:, 2:-1]
            if exclude_features:
                X_train = X_train.loc[:, ~X_train.columns.isin(exclude_features)]
                X_test = X_test.loc[:, ~X_test.columns.isin(exclude_features)]
            
            clf = xgb.XGBClassifier(**kwargs)
            clf.fit(X_train, y_train, 
                    eval_metric='logloss',
                    verbose=True)
            results_models.append((test_family, clf))

        return results_models

    def save_data_to_pickle(self,
                            data: Any,
                            filename: str) -> None:
        """Save pickable data to file.

        Args:
            data (Any): pickable data
            filename (str):
        """
        with open(self.PROJECT_DIR / filename, mode='bw') as fp:
            pickle.dump(data, fp)

    def save_results_to_csv(self,
                            models_data: List[Tuple[str, RandomForestClassifier]],
                            virus_groups: Dict[str, List[str]],
                            df: pd.DataFrame,
                            filename: str,
                            exclude_features=None) -> None:
        """Save virus, host and prediction of interaction.

        Args:
            models_data (List[List[str, RandomForestClassifier]]): Name of the test group, classifier
            virus_groups (Dict[str, List[str]]): Viruses grouped by family rank.
            df (pd.DataFrame): full dataframe of host and viruses.
            filename (str):
        """
        csv_data = []
        for group_name, model in models_data:
            virus_to_test = virus_groups[group_name]
            host_to_test_df = df.loc[virus_to_test]
            data_to_predict = host_to_test_df.reset_index().iloc[:, 2:-1]
            if exclude_features:
                data_to_predict = data_to_predict.loc[:, ~data_to_predict.columns.isin(exclude_features)]
            results_interaction = model.predict_proba(data_to_predict)
            csv_data.extend([
                            {'virus': virus,
                             'host': host,
                             'prediction': prediction
                             }
                            for virus, host, prediction
                            in zip(
                                [pair[0] for pair in host_to_test_df.index],
                                [pair[1] for pair in host_to_test_df.index],
                                [f'{prediction[1]:.5f}' for prediction in results_interaction])
                            ])
        with open(self.PROJECT_DIR / filename, mode='w') as fp:
            writer = csv.DictWriter(
                fp, fieldnames=['virus', 'host', 'prediction'])
            writer.writeheader()
            writer.writerows(csv_data)

    def load_data_from_pickle(self, filename: str) -> Any:
        """Load pickable data from file.

        Args:
            filename (str):

        Returns:
            Any: pickable data
        """
        with open(self.PROJECT_DIR / filename, mode='br') as fp:
            data = pickle.load(fp)
        return data

    def load_results_from_csv(self, filename: str) -> pd.DataFrame:
        """Load virus, host and prediction of interaction.

        Args:
            filename (str):

        Returns:
            pd.DataFrame: virus, host & prediction
        """
        return pd.read_csv(self.PROJECT_DIR / filename, header=0, index_col=[0, 1]).sort_values(by='prediction', ascending=False)

    def get_feature_importances(self, classifiers: List[RandomForestClassifier], exclude_features=None) -> pd.DataFrame:
        """Get feature importances data.

        Args:
            classifiers (List[RandomForestClassifier]): List of classifiers.

        Returns:
            pd.DataFrame: features vs importance, percent, standard deviation
        """
        columns = self.df.columns[:-1]
        if exclude_features:
            columns = columns[~columns.isin(exclude_features)]
        importances_list = []
        for model in classifiers:
            importances = model.feature_importances_
            importances_list.append(pd.Series(importances, index=columns))

        importances_result = pd.concat(importances_list)
        # Add every feature to extract stderr and mean
        mean_dict = dict()
        stderr_dict = dict()
        for feature in columns:
            mean_value = importances_result.loc[feature].mean()
            stderr_value = importances_result.loc[feature].std()
            mean_dict.update({feature: mean_value})
            stderr_dict.update({feature: stderr_value})
        importances_result = pd.Series(mean_dict)
        importances_percent_from_mean = ((importances_result - importances_result.mean())*100).round(2)
        yerr_result = pd.Series(stderr_dict)
        return pd.concat([importances_result, importances_percent_from_mean, yerr_result], axis=1).rename(columns={0: 'importance', 1: 'percent', 2: 'yerr'})

    def plot_feature_importances(self,
                                 classifiers: List[RandomForestClassifier],
                                 size: Tuple[int, int] = (25, 10),
                                 exclude_features=None,
                                 **kwargs) -> p9.ggplot:
        """Plot feature importances from list of clasifiers having
        `classifier.feature_importances_` field.

        Args:
            classifiers (List[RandomForestClassifier]): List of classifiers.
            size (Tuple[int,int], optional): Size of the plot. Defaults to (25,10).
            **kwargs forwarded to overwrite ggplot's labs() function.

        Returns:
            ggplot: plot object
        """
        df = self.get_feature_importances(classifiers, exclude_features)
        mean = df['importance'].mean()
        return p9.ggplot(data=df, mapping=p9.aes(x=df.index, y='importance', fill='percent')) \
            + p9.geom_bar(stat='identity') \
            + p9.geom_errorbar(p9.aes(y="yerr", ymin="importance-yerr", ymax="importance+yerr")) \
            + p9.geom_hline(yintercept=mean, linetype='dashed', color='red', size=0.9, alpha=0.4) \
            + p9.labs(title='Feature importance', y='Mean importance (the higher, the better)', fill='Deviation from mean (%)') \
            + p9.labs(**kwargs) \
            + p9.scale_fill_gradient2(low='red', mid='darkgrey', high='darkgreen') \
            + p9.scale_y_continuous(breaks=np.arange(0, 0.9, 0.1), limits=[0, 0.8]) \
            + p9.theme_seaborn() \
            + p9.theme(figure_size=size,
                       panel_grid=p9.element_line(color='darkgrey'),
                       panel_grid_major=p9.element_line(size=1.4, alpha=1),
                       panel_grid_major_x=p9.element_line(
                           linetype='dashed'),
                       panel_grid_major_y=p9.element_line(
                           linetype='dashdot'),
                       panel_grid_minor=p9.element_line(alpha=.25),
                       panel_grid_minor_x=p9.element_line(color='grey'),
                       panel_grid_minor_y=p9.element_line(color='grey'))

    def plot_host_taxonomy_probability(self, predictions: pd.DataFrame, size:Tuple[int,int]=(12,10), **kwargs) -> p9.ggplot:
        """Plot host correct predictions between taxonomic groups from list of predictions.

        Args:
            predictions (pd.DataFrame): final predictions results
            size (Tuple[int,int], optional): Size of the plot. Defaults to (12,10).
            **kwargs forwarded to overwrite ggplot's labs() function.

        Returns:
            ggplot: plot object
        """
        correct_predictions = {'superkingdom': 0,
                       'phylum': 0,
                       'class': 0,
                       'order': 0,
                       'family': 0,
                       'genus': 0,
                       'species': 0
                       }
        virus_json = json.loads(self.VIRUS_JSON.read_text())
        host_json = json.loads(self.HOST_JSON.read_text())

        for virus, new_df in predictions.groupby(level='virus'):
            max_scores = new_df[new_df == new_df.max()].dropna()
            hosts = max_scores.index.get_level_values('host')
            random_host = choice(hosts)
            results = [x == y for x,y in zip(virus_json[virus]['host']['lineage_names'], host_json[random_host]['lineage_names'])]
            if results[0]:
                correct_predictions['superkingdom'] += 1
            if results[1]:
                correct_predictions['phylum'] += 1
            if results[2]:
                correct_predictions['class'] += 1
            if results[3]:
                correct_predictions['order'] += 1
            if results[4]:
                correct_predictions['family'] += 1
            if results[5]:
                correct_predictions['genus'] += 1
            if results[6]:
                correct_predictions['species'] += 1
        
        max_value = max(list(correct_predictions.values()))
        normalized_predictions = {key:value / max_value for key, value in correct_predictions.items()}
        pred_res = pd.Series(normalized_predictions)
        df = pred_res.to_frame().rename(columns={0:'prediction'}).sort_values(by='prediction', ascending=False)
        df = df.reset_index().rename(columns={'index':'taxonomy'})
        df['taxonomy'] = pd.Categorical(df.taxonomy, categories=pd.unique(df.taxonomy))
        df['percent'] = (df.prediction * 100).round(2)
        mean = df.prediction.mean()
        return p9.ggplot(data=df, mapping=p9.aes(x='taxonomy', y='prediction', label='percent')) \
            + p9.geom_bar(stat='identity') \
            + p9.geom_hline(yintercept=mean, linetype='dashed', color='red', size=0.9, alpha=0.4, show_legend=True) \
            + p9.scale_y_continuous(breaks=np.arange(0, 1.1, 0.1), limits=[0, 1]) \
            + p9.labs(title='Correct host predictions to taxonomic groups', y='Frequency', x='Taxonomic group') \
            + p9.labs(**kwargs) \
            + p9.geom_text(position=p9.position_stack(vjust=0.9)) \
            + p9.theme_seaborn() \
            + p9.theme(figure_size=size,
                       panel_grid=p9.element_line(color='darkgrey'),
                       panel_grid_major=p9.element_line(size=1.4, alpha=1),
                       panel_grid_major_x=p9.element_line(
                           linetype='dashed'),
                       panel_grid_major_y=p9.element_line(
                           linetype='dashdot'),
                       panel_grid_minor=p9.element_line(alpha=.25),
                       panel_grid_minor_x=p9.element_line(color='grey'),
                       panel_grid_minor_y=p9.element_line(color='grey')) 
    
    def get_y_true_and_y_pred(self, predictions_filename: str) -> Tuple[List[float], List[float]]:
        data = []
        prediction = []
        data_dict = {}
        prediction_dict = {}
        pairs_probs = {
            'positive':[],
            'negative':[]
        }


        with open(self.PROJECT_DIR / predictions_filename, mode='r') as fp_results:
            reader_results = csv.DictReader(fp_results)
            for row in reader_results:
                prediction_dict[(row['virus'], row['host'])] = row['prediction']

        with open(self.DATAFRAME, mode='r') as fp_data:
            reader_data = csv.DictReader(fp_data)
            for row in reader_data:
                data_dict[(row['virus'], row['host'])] = row['interaction']

        for virus, host in prediction_dict.keys():
            data.append(float(data_dict[(virus, host)]))
            prediction.append(float(prediction_dict[(virus, host)]))
            if data_dict[(virus, host)] == '0':
                pairs_probs['negative'].append(float(prediction_dict[(virus, host)]))
            else:
                pairs_probs['positive'].append(float(prediction_dict[(virus, host)]))

        return data, prediction, pairs_probs
        
    def plot_precision_recall_curve(self, data, predicitons, size, **kwargs):
        precision, recall, thresholds = metrics.precision_recall_curve(data, predicitons)
        f_score = (2 * precision * recall) / (precision + recall)
        df = pd.DataFrame(data={'precision': precision, 'recall': recall})

        index = np.argmax(f_score)
        best_f_score = round(f_score[index], ndigits=4)
        best_threshold = round(thresholds[index], ndigits=4)
        best_recall = round(recall[index], ndigits=4)
        best_precision = round(precision[index], ndigits=4)
        auc = round(metrics.auc(recall, precision), ndigits=4)
        
        # dodaj geom step
        return p9.ggplot(data=df) \
                + p9.geom_step(p9.aes(x = 'recall', y = 'precision'), direction='vh') \
                + p9.geom_text(p9.aes(x = best_recall, y = best_precision),
                                label = f'Optimal threshold {best_threshold}',
                                nudge_x = 0.18,
                                nudge_y = 0,
                                size = 8,
                                fontstyle = 'italic') \
                + p9.labs(title=f'Precision Recall Curve\nBest f-score: {best_f_score} Best threshold: {best_threshold}\nAUC: {auc}',
                            x='Recall', y='Precision') \
                + p9.labs(**kwargs) \
                + p9.scale_x_continuous(breaks=np.arange(0, 1.1, 0.1), limits=[0, 1]) \
                + p9.scale_y_continuous(breaks=np.arange(0, 1.1, 0.1), limits=[0, 1]) \
                + p9.theme_seaborn() \
                + p9.theme(figure_size=size,
                            panel_grid=p9.element_line(color='darkgrey'),
                            panel_grid_major=p9.element_line(size=1.4, alpha=1),
                            panel_grid_major_x=p9.element_line(
                                linetype='dashed'),
                            panel_grid_major_y=p9.element_line(
                                linetype='dashdot'),
                            panel_grid_minor=p9.element_line(alpha=.25),
                            panel_grid_minor_x=p9.element_line(color='grey'),
                            panel_grid_minor_y=p9.element_line(color='grey')) \
    
    def plot_roc_curve(self, y_data, y_predictions, size=(12,10), **kwargs):
        FPR, TPR, _ = metrics.roc_curve(y_data, y_predictions)
        df = pd.DataFrame(data={'FPR': FPR, 'TPR': TPR})
        df['red_line'] = np.linspace(0,1,len(FPR))
        auc = round(metrics.auc(FPR, TPR), ndigits=4)
        return p9.ggplot(data=df) \
            + p9.geom_line(p9.aes(y='TPR', x='FPR'), size=0.8) \
            + p9.geom_line(p9.aes(x='red_line', y='red_line'), color='red', linetype='dashed', size=0.7, alpha=0.4) \
            + p9.scale_x_continuous(breaks=np.arange(0,1.1,0.1)) \
            + p9.scale_y_continuous(breaks=np.arange(0,1.1,0.1)) \
            + p9.labs(title=f'ROC Curve\nAUC={auc}', x='False Positive Rate', y='True Positive Rate') \
            + p9.labs(**kwargs) \
            + p9.theme_seaborn() \
            + p9.theme(figure_size=size,
                            panel_grid=p9.element_line(color='darkgrey'),
                            panel_grid_major=p9.element_line(size=1.4, alpha=1),
                            panel_grid_major_x=p9.element_line(
                                linetype='dashed'),
                            panel_grid_major_y=p9.element_line(
                                linetype='dashdot'),
                            panel_grid_minor=p9.element_line(alpha=.25),
                            panel_grid_minor_x=p9.element_line(color='grey'),
                            panel_grid_minor_y=p9.element_line(color='grey')) 

    def plot_boxplot_proba(self, pairs_proba, size=(12,10), **kwargs):
        df_positive = pd.Series(pairs_proba['positive']).to_frame().reset_index(drop=True).rename(columns={0:'positive'})
        df_negative = pd.Series(pairs_proba['negative']).to_frame().reset_index(drop=True).rename(columns={0:'negative'})
        
        return p9.ggplot() + p9.geom_boxplot(data=df_positive, mapping=p9.aes(x=df_positive.columns, y='positive'), outlier_shape='', fill='green') \
            + p9.geom_boxplot(data=df_negative, mapping=p9.aes(x=df_negative.columns, y='negative'), outlier_shape='', fill='red') \
            + p9.scale_y_continuous(breaks=np.arange(0, 1.1, 0.1), limits=[0, 1]) \
            + p9.labs(title='Distribution of predictions by label', y='Probability of interaction', x='Label') \
            + p9.labs(**kwargs) \
            + p9.theme_seaborn() \
            + p9.theme(figure_size=(10,12),
                       panel_grid_major=p9.element_line(size=1.4, alpha=1),
                       panel_grid_major_y=p9.element_line(
                           linetype='dashdot'),
                       panel_grid_minor=p9.element_line(alpha=.25),
                       panel_grid_minor_y=p9.element_line(color='grey')) 
                

