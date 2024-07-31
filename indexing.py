
import pandas as pd
import numpy as np
# the user can update these values to indicate the lower-level metrics to be used in computing brand equity
# ********** CHOOSING 8 week rolling mean ***************************


def adjust_negative_values(a):
    if any(n < 0 for n in a):
        return [abs(min(a))+x for x in a]
    else:
        return a


class IndexClass:

    __slots__ = ["smoothening_parameters",
                 "awareness_metrics",
                 "saliency_metrics",
                 "affinity_metrics",
                 "index_brand"]

    def __init__(self, smoothening_parameters: dict, awareness_metrics: list, saliency_metrics: list, affinity_metrics: list, index_brand: dict):
        """_summary_

        Args:
            smoothening_parameters (dict): _description_
            awareness_metrics (list): _description_
            saliency_metrics (list): _description_
            affinity_metrics (list): _description_
            index_brand (dict): _description_
        """

        self.smoothening_parameters = smoothening_parameters
        self.awareness_metrics = awareness_metrics
        self.saliency_metrics = saliency_metrics
        self.affinity_metrics = affinity_metrics
        self.index_brand = index_brand

    def indexing_normalisation(self, equity_file: pd.DataFrame, category: str, method: str) -> pd.DataFrame:
        """_summary_

        Args:
            equity_file (pd.DataFrame): _description_
            category (str): _description_
            method (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        data = equity_file.copy()
        if '_smoothened' in method:
            awareness_metrics_sm = [
                i+'_' + str(self.smoothening_parameters['window_size'][0]) + '_week_rolling_mean' for i in self.awareness_metrics]
            saliency_metrics_sm = [
                i+'_' + str(self.smoothening_parameters['window_size'][0]) + '_week_rolling_mean' for i in self.saliency_metrics]
            affinity_metrics_sm = [
                i+'_' + str(self.smoothening_parameters['window_size'][0]) + '_week_rolling_mean' for i in self.affinity_metrics]

            # compute weekly normalised indexed performance for lower-level metrics
            all_equity_metrics = awareness_metrics_sm + \
                saliency_metrics_sm + affinity_metrics_sm

        else:
            # compute weekly normalised indexed performance for lower-level metrics
            all_equity_metrics = self.awareness_metrics + \
                self.saliency_metrics + self.affinity_metrics

        # compute the proportioned weekly performance for the lower-level metrics
        weekly_metrics_df = pd.DataFrame()
        for metric in all_equity_metrics:

            # determine the baseline value as the first week with data > 0 for index brand
            # try:
            #     index = data[(data['brand'] == self.index_brand[category]) & (~data[metric].isnull()) & (data[metric] > 0)][metric].reset_index(drop=True)[0]
            # except Exception as e:
            #     print(e)
            index = 1
            # index everything to this baseline value
            if metric not in data.columns:raise ValueError(f"{metric} does not exist in data columns.")
            indexed_data = data[['Week Commencing', 'brand', metric]].copy()
            variable_name = str(metric) + "_indexed"
            indexed_data[variable_name] = indexed_data[metric]/index
    #        if 'Value' in metric:
            indexed_data[variable_name] = indexed_data.groupby(['Week Commencing'])[variable_name].transform(adjust_negative_values)

            # proportion the performance of the baselined valuesmalised
            proportioned_index = (indexed_data.groupby(['Week Commencing', 'brand'])[variable_name].sum()/indexed_data.groupby(['Week Commencing'])[variable_name].sum()).reset_index()
            combined_index = pd.merge(indexed_data[['Week Commencing', 'brand', metric]], proportioned_index, on=['Week Commencing', 'brand'], how='inner')
            if len(weekly_metrics_df) == 0:
                weekly_metrics_df = combined_index.copy('deep')
            else:
                weekly_metrics_df = pd.merge(weekly_metrics_df, combined_index, on=['Week Commencing', 'brand'], how='left')
                

        return weekly_metrics_df

    def indexing_normalisation_north_star(self, equity_file: pd.DataFrame, category: str, method: str) -> pd.DataFrame:
        """_summary_

        Args:
            equity_file (pd.DataFrame): _description_
            category (str): _description_
            method (str): _description_

        Raises: 
            ValueError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        data = equity_file.copy()
        if '_smoothened' in method:
            awareness_metrics_sm = [
                i+'_' + str(self.smoothening_parameters['window_size'][0]) + '_week_rolling_mean' for i in self.awareness_metrics]
            saliency_metrics_sm = [
                i+'_' + str(self.smoothening_parameters['window_size'][0]) + '_week_rolling_mean' for i in self.saliency_metrics]
            affinity_metrics_sm = [
                i+'_' + str(self.smoothening_parameters['window_size'][0]) + '_week_rolling_mean' for i in self.affinity_metrics]

            # compute weekly normalised indexed performance for lower-level metrics
            all_equity_metrics = awareness_metrics_sm + \
                saliency_metrics_sm + affinity_metrics_sm

        else:
            # compute weekly normalised indexed performance for lower-level metrics
            all_equity_metrics = self.awareness_metrics + \
                self.saliency_metrics + self.affinity_metrics

        # compute the proportioned weekly performance for the lower-level metrics
        weekly_metrics_df = pd.DataFrame()
        for metric in all_equity_metrics:
            try:
                # determine the baseline value as the first week with data > 0 for index brand
                index = data[(data['brand'] == self.index_brand[category]) & (~data[metric].isnull()) & (data[metric] > 0)][metric].reset_index(drop=True)[0]
            except Exception as e:
                print(e)
                index = 1
            # index everything to this baseline value
            if metric not in data.columns:
                raise ValueError(f"{metric} does not exist in data columns.")
            indexed_data = data[['Week Commencing', 'brand', metric]].copy()
            variable_name = str(metric) + "_indexed"
            indexed_data[variable_name] = indexed_data[metric]/index
            # if 'Value' in metric:
            indexed_data[variable_name] = indexed_data.groupby(['Week Commencing'])[variable_name].transform(adjust_negative_values)
            if len(weekly_metrics_df) == 0:
                weekly_metrics_df = indexed_data.copy('deep')
            else:
                weekly_metrics_df = pd.merge(weekly_metrics_df, indexed_data, on=[
                    'Week Commencing', 'brand'], how='left')

        return weekly_metrics_df
