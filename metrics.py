import pandas as pd


class AggregateMetrics:

    def __init__(self, smoothening_parameters: dict, awareness_metrics: list, saliency_metrics: list, affinity_metrics: list, weigths: dict):

        self.smoothening_parameters = smoothening_parameters
        self.awareness_metrics = awareness_metrics
        self.saliency_metrics = saliency_metrics
        self.affinity_metrics = affinity_metrics
        self.weigths = weigths

    def prepare_final_weekly_metrics(self, weekly_metrics_df: pd.DataFrame, method: str,smoothening_parameters,awareness_metrics,saliency_metrics,affinity_metrics,weights) -> pd.DataFrame:
        """ Function that prepares the main weekly dataframe for all the calculations. All other views and cuts are made using this weekly dataframe as reference. 

        Args:
            weekly_metrics_df_original (pd.DataFrame): Original weekly dataframe before processing
            method (str): Method from the "metrics_calc_method" settings from the config file

        Returns:
            pd.DataFrame: Processed dataframe 
        """
        weekly_metrics_df.fillna(0, inplace=True)
        if '_smoothened' in method:
            aw_mt = [i+'_' + str(smoothening_parameters['window_size'][0]) + '_week_rolling_mean_indexed' for i in awareness_metrics]
            sa_mt = [i+'_' + str(smoothening_parameters['window_size'][0]) + '_week_rolling_mean_indexed' for i in saliency_metrics]
            af_mt = [i+'_' + str(smoothening_parameters['window_size'][0]) + '_week_rolling_mean_indexed' for i in affinity_metrics]
                
        elif '_unsmoothened' in method:
            aw_mt = [i + '_indexed' for i in awareness_metrics]
            sa_mt = [i + '_indexed' for i in saliency_metrics]
            af_mt = [i + '_indexed' for i in affinity_metrics]
                
    
        weekly_metrics_df['Framework_Awareness'] = weekly_metrics_df[aw_mt].mul(weights['awareness']).sum(axis=1)
        weekly_metrics_df['Framework_Saliency'] = weekly_metrics_df[sa_mt].mul(weights['saliency']).sum(axis=1)
        weekly_metrics_df['Framework_Affinity'] = weekly_metrics_df[af_mt].mul(weights['affinity']).sum(axis=1)
        
        if 'average' in method:
            final_metrics = [i for i in list(weekly_metrics_df.columns) if 'indexed' in i]
            weekly_metrics_df = weekly_metrics_df[['Week Commencing', 'brand'] + final_metrics + 
                                                ['Framework_Awareness','Framework_Saliency','Framework_Affinity']]
            weekly_metrics_df.columns = ['Week Commencing','brand', 'AA_eSoV', 'AA_Reach', 'AA_Brand_Breadth',
                'AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov",'AS_Search_Index',
                'AS_Brand_Centrality', 'AF_Brand_Love', 'AF_Motivation_for_Change', 'AF_Consumption_Experience',
                'AF_Supporting_Experience','AF_Value_for_Money','Framework_Awareness',
                'Framework_Saliency', 'Framework_Affinity']
            weekly_metrics_df.rename({'Week Commencing': 'time'}, axis=1, inplace=True)
            weekly_metrics_df['time_period'] = 'weeks'
            weekly_metrics_df = weekly_metrics_df[['time', 'time_period','brand', 'AA_eSoV', 'AA_Reach', 'AA_Brand_Breadth',
                'AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov",'AS_Search_Index',
                'AS_Brand_Centrality', 'AF_Brand_Love', 'AF_Motivation_for_Change', 'AF_Consumption_Experience',
                'AF_Supporting_Experience','AF_Value_for_Money','Framework_Awareness',
                'Framework_Saliency', 'Framework_Affinity']]
    
        elif 'total' in method:
            final_metrics = [i for i in list(weekly_metrics_df.columns) if 'indexed' not in i]
            weekly_metrics_df = weekly_metrics_df[final_metrics]
            weekly_metrics_df.columns = ['Week Commencing','brand', 'AA_eSoV', 'AA_Reach', 'AA_Brand_Breadth',
                'AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov",'AS_Search_Index',
                'AS_Brand_Centrality', 'AF_Brand_Love', 'AF_Motivation_for_Change', 'AF_Consumption_Experience',
                'AF_Supporting_Experience','AF_Value_for_Money','Framework_Awareness',
                'Framework_Saliency', 'Framework_Affinity']

            weekly_metrics_df.rename({'Week Commencing': 'time'}, axis=1, inplace=True)
            weekly_metrics_df['time_period'] = 'weeks'
            weekly_metrics_df = weekly_metrics_df[['time', 'time_period','brand', 'AA_eSoV', 'AA_Reach', 'AA_Brand_Breadth',
                'AS_Average_Engagement', 'AS_Usage_SoV', 'AS_Trial_Sov',"Quitting_Sov","Consideration_Sov",'AS_Search_Index',
                'AS_Brand_Centrality', 'AF_Brand_Love', 'AF_Motivation_for_Change', 'AF_Consumption_Experience',
                'AF_Supporting_Experience','AF_Value_for_Money', 'Framework_Awareness',
                'Framework_Saliency', 'Framework_Affinity']]
        
        weekly_metrics_df['Framework_Awareness'] = weekly_metrics_df['Framework_Awareness']*100
        weekly_metrics_df['Framework_Saliency'] = weekly_metrics_df['Framework_Saliency']*100
        weekly_metrics_df['Framework_Affinity'] = weekly_metrics_df['Framework_Affinity']*100
            
        return weekly_metrics_df
    

    

    def prepare_final_weekly_metrics_refresh(self, weekly_metrics_df: pd.DataFrame, method: str,smoothening_parameters,awareness_metrics,saliency_metrics,affinity_metrics,weights) -> pd.DataFrame:
        """ Function that prepares the main weekly dataframe for all the calculations. All other views and cuts are made using this weekly dataframe as reference. 

        Args:
            weekly_metrics_df_original (pd.DataFrame): Original weekly dataframe before processing
            method (str): Method from the "metrics_calc_method" settings from the config file

        Returns:
            pd.DataFrame: Processed dataframe 
        """
        weekly_metrics_df.fillna(0, inplace=True)
        if '_smoothened' in method:
            aw_mt = [i+'_' + str(52) + '_week_rolling_mean_indexed' for i in awareness_metrics]
            sa_mt = [i+'_' + str(52) + '_week_rolling_mean_indexed' for i in saliency_metrics]
            af_mt = [i+'_' + str(52) + '_week_rolling_mean_indexed' for i in affinity_metrics]
                
        elif '_unsmoothened' in method:
            aw_mt = [i + '_indexed' for i in awareness_metrics]
            sa_mt = [i + '_indexed' for i in saliency_metrics]
            af_mt = [i + '_indexed' for i in affinity_metrics]
                
    
        weekly_metrics_df['Framework_Awareness'] = weekly_metrics_df[aw_mt].mul(weights['awareness']).sum(axis=1)
        weekly_metrics_df['Framework_Saliency'] = weekly_metrics_df[sa_mt].mul(weights['saliency']).sum(axis=1)
        weekly_metrics_df['Framework_Affinity'] = weekly_metrics_df[af_mt].mul(weights['affinity']).sum(axis=1)
        
        if 'average' in method:
            final_metrics = [i for i in list(weekly_metrics_df.columns) if 'indexed' in i]
            weekly_metrics_df = weekly_metrics_df[['Week Commencing', 'brand'] + final_metrics + 
                                                ['Framework_Awareness','Framework_Saliency','Framework_Affinity']]
            weekly_metrics_df.columns = ['Week Commencing','brand', 'AA_eSoV', 'AA_Reach', 'AA_Brand_Breadth',
                'AS_Average_Engagement', 'AS_Usage_SoV', 'AS_Trial_Sov',"Quitting_Sov","Consideration_Sov",'AS_Search_Index',
                'AS_Brand_Centrality', 'AF_Brand_Love', 'AF_Motivation_for_Change', 'AF_Consumption_Experience',
                'AF_Supporting_Experience','AF_Value_for_Money','Framework_Awareness',
                'Framework_Saliency', 'Framework_Affinity']
            weekly_metrics_df.rename({'Week Commencing': 'time'}, axis=1, inplace=True)
            weekly_metrics_df['time_period'] = 'weeks'
            weekly_metrics_df = weekly_metrics_df[['time', 'time_period','brand', 'AA_eSoV', 'AA_Reach', 'AA_Brand_Breadth',
                'AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov", 'AS_Search_Index',
                'AS_Brand_Centrality', 'AF_Brand_Love', 'AF_Motivation_for_Change', 'AF_Consumption_Experience',
                'AF_Supporting_Experience','AF_Value_for_Money','Framework_Awareness',
                'Framework_Saliency', 'Framework_Affinity']]
    
        elif 'total' in method:
            final_metrics = [i for i in list(weekly_metrics_df.columns) if 'indexed' not in i]
            weekly_metrics_df = weekly_metrics_df[final_metrics]
            weekly_metrics_df.columns = ['Week Commencing','brand', 'AA_eSoV', 'AA_Reach', 'AA_Brand_Breadth',
                'AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov", 'AS_Search_Index',
                'AS_Brand_Centrality', 'AF_Brand_Love', 'AF_Motivation_for_Change', 'AF_Consumption_Experience',
                'AF_Supporting_Experience','AF_Value_for_Money','Framework_Awareness',
                'Framework_Saliency', 'Framework_Affinity']

            weekly_metrics_df.rename({'Week Commencing': 'time'}, axis=1, inplace=True)
            weekly_metrics_df['time_period'] = 'weeks'
            weekly_metrics_df = weekly_metrics_df[['time', 'time_period','brand', 'AA_eSoV', 'AA_Reach', 'AA_Brand_Breadth',
                'AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov", 'AS_Search_Index',
                'AS_Brand_Centrality', 'AF_Brand_Love', 'AF_Motivation_for_Change', 'AF_Consumption_Experience',
                'AF_Supporting_Experience','AF_Value_for_Money', 'Framework_Awareness',
                'Framework_Saliency', 'Framework_Affinity']]
        
        weekly_metrics_df['Framework_Awareness'] = weekly_metrics_df['Framework_Awareness']*100
        weekly_metrics_df['Framework_Saliency'] = weekly_metrics_df['Framework_Saliency']*100
        weekly_metrics_df['Framework_Affinity'] = weekly_metrics_df['Framework_Affinity']*100
            
        return weekly_metrics_df


    def calculate_monthly_metrics(self, weekly_metrics_df: pd.DataFrame, method: str) -> pd.DataFrame:
        """ Function that takes the weekly metric dataframe and gets the aggregated value. 

        Args:
            weekly_metrics_df_original (pd.DataFrame): Original weekly dataframe before processing
            method (str): Method from the "metrics_calc_method" settings from the config file

        Returns:
            pd.DataFrame: Processed dataframe 
        """
        weekly_metrics_df['Month Commencing'] = weekly_metrics_df['time'].apply(lambda x: (x.to_period('M')))
        metrics = ['eSoV', 'Reach',
       'Brand Breadth', 'Average Engagement',
       'Usage SoV', 'Search Index',
       'Brand Centrality','Entry points & Key Moments','Brand Prestige & Love','Baby Milk','Adverts and Promotions','Value For Money',
        'Buying Experience','Preparing Milk','Baby Experience',
        'Awareness',
       'Saliency','Affinity']
    
        if 'average' in method:
            monthly_metrics_df = pd.DataFrame()
            for mt in metrics:

                # calculate quarterly metric
                monthly_df = weekly_metrics_df.groupby(['Month Commencing', 'brand'])[mt].mean().reset_index()

                # proportion metric so it adds upto a 100
                proportioned_index = (monthly_df.groupby(['Month Commencing', 'brand'])[mt].sum()/monthly_df.groupby(['Month Commencing'])[mt].sum()).reset_index()

                if len(monthly_metrics_df) == 0:
                    monthly_metrics_df = proportioned_index.copy('deep')
                else:
                    monthly_metrics_df = pd.merge(monthly_metrics_df, proportioned_index, on = ['Month Commencing', 'brand'], how='left')
            
        elif 'total' in method:
            monthly_metrics_df = pd.DataFrame()
            for mt in metrics:
                if mt in ['AA_eSoV', 'AA_Reach','AS_Average_Engagement', 'AS_Usage_SoV',"AS_Trial_Sov","Quitting_Sov","Consideration_Sov",'AF_Brand_Love', 'AF_Motivation_for_Change', 'AF_Consumption_Experience',
                'AF_Supporting_Experience','AF_Value_for_Money']:

                    monthly_df = weekly_metrics_df.groupby(['Month Commencing', 'brand'])[mt].sum().reset_index()
                
                else:
                    monthly_df = weekly_metrics_df.groupby(['Month Commencing', 'brand'])[mt].mean().reset_index()
                
                if mt in ['Framework_Awareness', 'Framework_Saliency','Framework_Affinity']:
                    # proportion metric so it adds upto a 100
                    proportioned_index = (monthly_df.groupby(['Month Commencing', 'brand'])[mt].sum()/monthly_df.groupby(['Month Commencing'])[mt].sum()).reset_index()
                else:
                    proportioned_index = monthly_df[['Month Commencing', 'brand', mt]]
                    
                if len( monthly_metrics_df) == 0:
                    monthly_metrics_df = proportioned_index.copy('deep')
                else:
                    monthly_metrics_df = pd.merge(monthly_metrics_df, proportioned_index, on = ['Month Commencing', 'brand'], how='left')

        
        monthly_metrics_df['Awareness'] =  monthly_metrics_df['Awareness']*100
        monthly_metrics_df['Saliency'] =  monthly_metrics_df['Saliency']*100
        monthly_metrics_df['Affinity'] =  monthly_metrics_df['Affinity']*100

        monthly_metrics_df['time'] = pd.PeriodIndex(monthly_metrics_df['Month Commencing'], freq='M').to_timestamp()
        monthly_metrics_df.drop('Month Commencing', axis=1, inplace=True)
        weekly_metrics_df.drop('Month Commencing', axis=1, inplace=True)
        monthly_metrics_df['time_period'] = 'months'
        monthly_metrics_df = monthly_metrics_df[['time', 'time_period', 'brand'] + metrics]
        monthly_metrics_df.fillna(0, inplace=True)
        return monthly_metrics_df
    
    def calculate_quarterly_metrics(self, weekly_metrics_df: pd.DataFrame, method: str) -> pd.DataFrame:
        """ Function that takes the weekly metric dataframe and gets the aggregated value. 

        Args:
            weekly_metrics_df_original (pd.DataFrame): Original weekly dataframe before processing
            method (str): Method from the "metrics_calc_method" settings from the config file

        Returns:
            pd.DataFrame: Processed dataframe 
        """
        weekly_metrics_df['Quarter Commencing'] = weekly_metrics_df['time'].apply(
            lambda x: (x.to_period('Q')))
        metrics = ['eSoV', 'Reach',
       'Brand Breadth', 'Average Engagement',
       'Usage SoV', 'Search Index',
       'Brand Centrality','Entry points & Key Moments','Brand Prestige & Love','Baby Milk','Adverts and Promotions','Value For Money',
        'Buying Experience','Preparing Milk','Baby Experience',
        'Awareness',
       'Saliency','Affinity']

        if 'average' in method:
            quarterly_metrics_df = pd.DataFrame()
            for mt in metrics:

                # calculate quarterly metric
                quarterly_df = weekly_metrics_df.groupby(['Quarter Commencing', 'brand'])[
                    mt].mean().reset_index()

                # proportion metric so it adds upto a 100
                proportioned_index = (quarterly_df.groupby(['Quarter Commencing', 'brand'])[
                                      mt].sum()/quarterly_df.groupby(['Quarter Commencing'])[mt].sum()).reset_index()

                if len(quarterly_metrics_df) == 0:
                    quarterly_metrics_df = proportioned_index.copy('deep')
                else:
                    quarterly_metrics_df = pd.merge(quarterly_metrics_df, proportioned_index, on=[
                                                    'Quarter Commencing', 'brand'], how='left')

        elif 'total' in method:
            quarterly_metrics_df = pd.DataFrame()
            for mt in metrics:
                if mt in ['AA_eSoV', 'AA_Reach', 'AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov", 'AF_Motivation for change',
                          'AF_Journey into NGP and brands', 'AF_Brand Prestige and love', 'AF_Value for money', 'AF_Smoking experience', 'AF_Suporting experience', 'AF_Adverts_and_promotions']:

                    quarterly_df = weekly_metrics_df.groupby(['Quarter Commencing', 'brand'])[
                        mt].sum().reset_index()
                else:
                    quarterly_df = weekly_metrics_df.groupby(['Quarter Commencing', 'brand'])[
                        mt].mean().reset_index()

                if mt in ['Framework_Awareness', 'Framework_Saliency', 'Framework_Affinity']:
                    # proportion metric so it adds upto a 100
                    proportioned_index = (quarterly_df.groupby(['Quarter Commencing', 'brand'])[
                                          mt].sum()/quarterly_df.groupby(['Quarter Commencing'])[mt].sum()).reset_index()
                else:
                    proportioned_index = quarterly_df[[
                        'Quarter Commencing', 'brand', mt]]

                if len(quarterly_metrics_df) == 0:
                    quarterly_metrics_df = proportioned_index.copy('deep')
                else:
                    quarterly_metrics_df = pd.merge(quarterly_metrics_df, proportioned_index, on=[
                                                    'Quarter Commencing', 'brand'], how='left')

        quarterly_metrics_df['Awareness'] = quarterly_metrics_df['Awareness']*100
        quarterly_metrics_df['Saliency'] = quarterly_metrics_df['Saliency']*100
        quarterly_metrics_df['Affinity'] = quarterly_metrics_df['Affinity']*100

        quarterly_metrics_df['time'] = pd.PeriodIndex(
            quarterly_metrics_df['Quarter Commencing'], freq='Q').to_timestamp()
        quarterly_metrics_df.drop('Quarter Commencing', axis=1, inplace=True)
        weekly_metrics_df.drop('Quarter Commencing', axis=1, inplace=True)
        quarterly_metrics_df['time_period'] = 'quarters'
        quarterly_metrics_df = quarterly_metrics_df[[
            'time', 'time_period', 'brand'] + metrics]
        quarterly_metrics_df.fillna(0, inplace=True)
        return quarterly_metrics_df

    


    def calculate_quarterly_metrics(self,weekly_metrics_df, method):
        weekly_metrics_df['Quarter Commencing'] = weekly_metrics_df['time'].apply(lambda x: (x.to_period('Q')))
        metrics =  ['eSoV', 'Reach',
       'Brand Breadth', 'Average Engagement',
       'Usage SoV', 'Search Index',
       'Brand Centrality','Entry points & Key Moments','Brand Prestige & Love','Baby Milk','Adverts and Promotions','Value For Money',
        'Buying Experience','Preparing Milk','Baby Experience',
        'Awareness',
       'Saliency','Affinity']
    
        if 'average' in method:
            quarterly_metrics_df = pd.DataFrame()
            for mt in metrics:

                # calculate quarterly metric
                quarterly_df = weekly_metrics_df.groupby(['Quarter Commencing', 'brand'])[mt].mean().reset_index()

                # proportion metric so it adds upto a 100
                proportioned_index = (quarterly_df.groupby(['Quarter Commencing', 'brand'])[mt].sum()/quarterly_df.groupby(['Quarter Commencing'])[mt].sum()).reset_index()

                if len( quarterly_metrics_df) == 0:
                    quarterly_metrics_df = proportioned_index.copy('deep')
                else:
                    quarterly_metrics_df = pd.merge(quarterly_metrics_df, proportioned_index, on = ['Quarter Commencing', 'brand'], how='left')
        
        elif 'total' in method:
            quarterly_metrics_df = pd.DataFrame()
            for mt in metrics:
                if mt in ['AA_eSoV', 'AA_Reach','AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov",'AF_Brand_Love', 'AF_Motivation_for_Change', 'AF_Consumption_Experience',
                'AF_Supporting_Experience','AF_Value_for_Money']:
                    
                    quarterly_df = weekly_metrics_df.groupby(['Quarter Commencing', 'brand'])[mt].sum().reset_index()
                else:
                    quarterly_df = weekly_metrics_df.groupby(['Quarter Commencing', 'brand'])[mt].mean().reset_index()
                
                if mt in ['Framework_Awareness', 'Framework_Saliency','Framework_Affinity']:
                    # proportion metric so it adds upto a 100
                    proportioned_index = (quarterly_df.groupby(['Quarter Commencing', 'brand'])[mt].sum()/quarterly_df.groupby(['Quarter Commencing'])[mt].sum()).reset_index()
                else:
                    proportioned_index = quarterly_df[['Quarter Commencing', 'brand', mt]]
                    
                if len( quarterly_metrics_df) == 0:
                    quarterly_metrics_df = proportioned_index.copy('deep')
                else:
                    quarterly_metrics_df = pd.merge(quarterly_metrics_df, proportioned_index, on = ['Quarter Commencing', 'brand'], how='left')
    
        quarterly_metrics_df['Awareness'] =  quarterly_metrics_df['Awareness']*100
        quarterly_metrics_df['Saliency'] =  quarterly_metrics_df['Saliency']*100
        quarterly_metrics_df['Affinity'] =  quarterly_metrics_df['Affinity']*100

        quarterly_metrics_df['time'] = pd.PeriodIndex(quarterly_metrics_df['Quarter Commencing'], freq='Q').to_timestamp()
        quarterly_metrics_df.drop('Quarter Commencing', axis=1, inplace=True)
        weekly_metrics_df.drop('Quarter Commencing', axis=1, inplace=True)
        quarterly_metrics_df['time_period'] = 'quarters'
        quarterly_metrics_df = quarterly_metrics_df[['time', 'time_period', 'brand'] + metrics]
        quarterly_metrics_df.fillna(0, inplace=True)
        return quarterly_metrics_df




    def convert_to_semiannual(self,date):
        year = date.year
        month = date.month
        if month <= 6:
            return f'H1-{year}'
        else:
            return f'H2-{year}'

    def convert_semiannual_to_timestamp(self,period):
        # Extract year and period (H1 or H2)
        year = int(period.split('-')[1])
        period_indicator = period.split('-')[0]

        # Set the month and day based on the period
        if period_indicator == 'H1':
            month = 1
            day = 1
        else:
            period_indicator == 'H2'
            month = 7
            day = 1
    
        return pd.Timestamp(year=year, month=month, day=day)


    def calculate_halfyearly_metrics(self,weekly_metrics_df, method):
        weekly_metrics_df['Half-year Commencing'] = weekly_metrics_df['time'].apply(self.convert_to_semiannual)
        metrics = ['eSoV', 'Reach',
       'Brand Breadth', 'Average Engagement',
       'Usage SoV', 'Search Index',
       'Brand Centrality','Entry points & Key Moments','Brand Prestige & Love','Baby Milk','Adverts and Promotions','Value For Money',
        'Buying Experience','Preparing Milk','Baby Experience',
        'Awareness',
       'Saliency','Affinity']
        
        if 'average' in method:
            halfyearly_metrics_df = pd.DataFrame()
            for mt in metrics:

                # calculate halfyearly metric
                halfyearly_df = weekly_metrics_df.groupby(['Half-year Commencing', 'brand'])[mt].mean().reset_index()

                if len(halfyearly_metrics_df) == 0:
                    halfyearly_metrics_df = halfyearly_df.copy('deep')
                else:
                    halfyearly_metrics_df = pd.merge(halfyearly_metrics_df, halfyearly_df, on = ['Half-year Commencing', 'brand'], how='left')
        elif 'total' in method:
                halfyearly_metrics_df = pd.DataFrame()
                for mt in metrics:
                    if mt in ['AA_eSoV', 'AA_Reach','AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov",'AF_Brand_Love', 'AF_Motivation_for_Change', 'AF_Consumption_Experience',
                'AF_Supporting_Experience','AF_Value_for_Money']:

                        halfyearly_df = weekly_metrics_df.groupby(['Half-year Commencing', 'brand'])[mt].sum().reset_index()
                    else:
                        halfyearly_df = weekly_metrics_df.groupby(['Half-year Commencing', 'brand'])[mt].mean().reset_index()

                    if len(halfyearly_metrics_df) == 0:
                        halfyearly_metrics_df = halfyearly_df.copy('deep')
                    else:
                        halfyearly_metrics_df = pd.merge(halfyearly_metrics_df, halfyearly_df, on = ['Half-year Commencing', 'brand'], how='left')

        halfyearly_metrics_df['time'] = halfyearly_metrics_df['Half-year Commencing'].apply(self.convert_semiannual_to_timestamp)
        halfyearly_metrics_df.drop('Half-year Commencing', axis=1, inplace=True)
        weekly_metrics_df.drop('Half-year Commencing', axis=1, inplace=True)
        halfyearly_metrics_df['time_period'] = 'semiannual'
        halfyearly_metrics_df = halfyearly_metrics_df[['time', 'time_period', 'brand'] + metrics]
        halfyearly_metrics_df.fillna(0, inplace=True) 
        return halfyearly_metrics_df


    def calculate_yearly_metrics(self, weekly_metrics_df: pd.DataFrame, method: str) -> pd.DataFrame:
        """ Function that takes the weekly metric dataframe and gets the aggregated value. 

        Args:
            weekly_metrics_df_original (pd.DataFrame): Original weekly dataframe before processing
            method (str): Method from the "metrics_calc_method" settings from the config file

        Returns:
            pd.DataFrame: Processed dataframe 
        """
        weekly_metrics_df['Year Commencing'] = weekly_metrics_df['time'].apply(lambda x: (x.to_period('Y')))
    
        metrics =  ['eSoV', 'Reach',
       'Brand Breadth', 'Average Engagement',
       'Usage SoV', 'Search Index',
       'Brand Centrality','Entry points & Key Moments','Brand Prestige & Love','Baby Milk','Adverts and Promotions','Value For Money',
        'Buying Experience','Preparing Milk','Baby Experience',
        'Awareness',
       'Saliency','Affinity']
    
        if 'average' in method:
            yearly_metrics_df = pd.DataFrame()
            for mt in metrics:

                # calculate yearly metric
                yearly_df = weekly_metrics_df.groupby(['Year Commencing', 'brand'])[mt].mean().reset_index()

                # proportion metric so it adds upto a 100
                proportioned_index = (yearly_df.groupby(['Year Commencing', 'brand'])[mt].sum()/yearly_df.groupby(['Year Commencing'])[mt].sum()).reset_index()

                if len(yearly_metrics_df) == 0:
                    yearly_metrics_df = proportioned_index.copy('deep')
                else:
                    yearly_metrics_df = pd.merge(yearly_metrics_df, proportioned_index, on = ['Year Commencing', 'brand'], how='left')
    
        elif 'total' in method:
            yearly_metrics_df = pd.DataFrame()
            for mt in metrics:
                if mt in ['AA_eSoV', 'AA_Reach','AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov",'AF_Brand_Love', 'AF_Motivation_for_Change', 'AF_Consumption_Experience',
                'AF_Supporting_Experience','AF_Value_for_Money']:
                    
                    yearly_df = weekly_metrics_df.groupby(['Year Commencing', 'brand'])[mt].sum().reset_index()
                else:
                    yearly_df = weekly_metrics_df.groupby(['Year Commencing', 'brand'])[mt].mean().reset_index()
                
                if mt in ['Framework_Awareness', 'Framework_Saliency','Framework_Affinity']:
                    # proportion metric so it adds upto a 100
                    proportioned_index = (yearly_df.groupby(['Year Commencing', 'brand'])[mt].sum()/yearly_df.groupby(['Year Commencing'])[mt].sum()).reset_index()
                else:
                    proportioned_index = yearly_df[['Year Commencing', 'brand', mt]]
                
                if len( yearly_metrics_df) == 0:
                    yearly_metrics_df = proportioned_index.copy('deep')
                else:
                    yearly_metrics_df = pd.merge(yearly_metrics_df, proportioned_index, on = ['Year Commencing', 'brand'], how='left')
        
            
        yearly_metrics_df['Awareness'] =  yearly_metrics_df['Awareness']*100
        yearly_metrics_df['Saliency'] =  yearly_metrics_df['Saliency']*100
        yearly_metrics_df['Affinity'] =  yearly_metrics_df['Affinity']*100

        yearly_metrics_df['time'] = pd.PeriodIndex(yearly_metrics_df['Year Commencing'], freq='Y').to_timestamp()
        yearly_metrics_df.drop('Year Commencing', axis=1, inplace=True)
        weekly_metrics_df.drop('Year Commencing', axis=1, inplace=True)
        yearly_metrics_df['time_period'] = 'years'
        yearly_metrics_df = yearly_metrics_df[['time', 'time_period', 'brand'] + metrics]
        yearly_metrics_df.fillna(0, inplace=True)
        return yearly_metrics_df

    

    # new method
    def calculate_yearly_metrics_refresh(self, eqf) -> pd.DataFrame:
        """ Function that takes the weekly metric dataframe and gets the aggregated value. 

        Args:
            weekly_metrics_df_original (pd.DataFrame): Original weekly dataframe before processing
            method (str): Method from the "metrics_calc_method" settings from the config file

        Returns:
            pd.DataFrame: Processed dataframe 
        """
                #calculate the average of latest year
        latest_year_metrics_df = eqf[eqf.time_period == 'quarters']
        latest_year_metrics_df['Year Commencing'] = latest_year_metrics_df['time'].apply(lambda x: (x.to_period('Y')))
        metrics =  ['eSoV', 'Reach',
       'Brand Breadth', 'Average Engagement',
       'Usage SoV', 'Search Index',
       'Brand Centrality','Entry points & Key Moments','Brand Prestige & Love','Baby Milk','Adverts and Promotions','Value For Money',
        'Buying Experience','Preparing Milk','Baby Experience',
        'Awareness',
       'Saliency','Affinity']

        yearly_metrics_df = pd.DataFrame()
        for mt in metrics:

            # calculate yearly metric
            yearly_df = latest_year_metrics_df.groupby(['Year Commencing', 'brand'])[mt].mean().reset_index()

            # proportion metric so it adds upto a 100
        #    proportioned_index = (yearly_df.groupby(['Year Commencing', 'brand'])[mt].sum()/yearly_df.groupby(['Year Commencing'])[mt].sum()).reset_index()

            if len( yearly_metrics_df) == 0:
                yearly_metrics_df = yearly_df.copy('deep')
            else:
                yearly_metrics_df = pd.merge(yearly_metrics_df, yearly_df, on = ['Year Commencing', 'brand'], how='left')

        #    yearly_metrics_df[mt] = yearly_metrics_df[mt]*100
        #    yearly_metrics_df[mt] = yearly_metrics_df[mt].round(2)

        yearly_metrics_df = yearly_metrics_df[(yearly_metrics_df['Year Commencing'] == '2024-01-01')]
        yearly_metrics_df['Total_Equity'] = yearly_metrics_df[['Framework_Awareness', 'Framework_Saliency','Framework_Affinity']].mean(axis=1)                                  
        yearly_metrics_df['time'] = pd.PeriodIndex(yearly_metrics_df['Year Commencing'], freq='Y').to_timestamp()
        yearly_metrics_df.drop('Year Commencing', axis=1, inplace=True)
        yearly_metrics_df['time_period'] = 'years'
        yearly_metrics_df = yearly_metrics_df[['time', 'time_period', 'brand'] + metrics + ['Total_Equity']]
        yearly_metrics_df = yearly_metrics_df.round(2)
        yearly_metrics_df.fillna(0, inplace=True)
        yearly_metrics_df.reset_index(inplace=True, drop=True)

        return yearly_metrics_df




    def calculate_monthly_metrics_north_star(self, weekly_metrics_df_original: pd.DataFrame, method: str) -> pd.DataFrame:
        """ Function that takes the weekly metric dataframe and gets the aggregated value. 

        Args:
            weekly_metrics_df_original (pd.DataFrame): Original weekly dataframe before processing
            method (str): Method from the "metrics_calc_method" settings from the config file

        Returns:
            pd.DataFrame: Processed dataframe 
        """
        weekly_metrics_df = weekly_metrics_df_original.copy()

        weekly_metrics_df['Month Commencing'] = weekly_metrics_df['time'].apply(
            lambda x: (x.to_period('M')))
        metrics =  ['eSoV', 'Reach',
       'Brand_Breadth', 'Average_Engagement',
       'Usage_SoV', 'Search_Index',
       'Brand_Centrality','Entry_point','Brand_Love','Baby_Milk','Adverts_Promo','Value_for_Money','Buying_Exp','Prep_Milk','Baby_exp',
        'Awareness',
       'Saliency','Affinity']

        if 'average' in method:
            monthly_metrics_df = pd.DataFrame()
            for mt in metrics:

                # calculate quarterly metric
                monthly_df = weekly_metrics_df.groupby(['Month Commencing', 'brand'])[
                    mt].mean().reset_index()

                if len(monthly_metrics_df) == 0:
                    monthly_metrics_df = monthly_df.copy('deep')
                else:
                    monthly_metrics_df = pd.merge(monthly_metrics_df, monthly_df, on=[
                                                  'Month Commencing', 'brand'], how='left')

        elif 'total' in method:
            monthly_metrics_df = pd.DataFrame()
            for mt in metrics:
                if mt in ['AA_eSoV', 'AA_Reach', 'AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov", 'AF_Motivation for change',
                          'AF_Journey into NGP and brands', 'AF_Brand Prestige and love', 'AF_Value for money', 'AF_Smoking experience', 'AF_Suporting experience', 'AF_Adverts_and_promotions']:

                    monthly_df = weekly_metrics_df.groupby(['Month Commencing', 'brand'])[
                        mt].sum().reset_index()

                else:
                    monthly_df = weekly_metrics_df.groupby(['Month Commencing', 'brand'])[
                        mt].mean().reset_index()

                if len(monthly_metrics_df) == 0:
                    monthly_metrics_df = monthly_df.copy('deep')
                else:
                    monthly_metrics_df = pd.merge(monthly_metrics_df, monthly_df, on=[
                                                  'Month Commencing', 'brand'], how='left')

        monthly_metrics_df['time'] = pd.PeriodIndex(
            monthly_metrics_df['Month Commencing'], freq='M').to_timestamp()
        monthly_metrics_df.drop('Month Commencing', axis=1, inplace=True)
        weekly_metrics_df.drop('Month Commencing', axis=1, inplace=True)
        monthly_metrics_df['time_period'] = 'months'
        monthly_metrics_df = monthly_metrics_df[[
            'time', 'time_period', 'brand'] + metrics]
        monthly_metrics_df.fillna(0, inplace=True)
        return monthly_metrics_df

    def calculate_quarterly_metrics_north_star(self, weekly_metrics_df_original: pd.DataFrame, method: str) -> pd.DataFrame:
        """ Function that takes the weekly metric dataframe and gets the aggregated value. 

        Args:
            weekly_metrics_df_original (pd.DataFrame): Original weekly dataframe before processing
            method (str): Method from the "metrics_calc_method" settings from the config file

        Returns:
            pd.DataFrame: Processed dataframe 
        """
        weekly_metrics_df = weekly_metrics_df_original.copy()

        weekly_metrics_df['Quarter Commencing'] = weekly_metrics_df['time'].apply(
            lambda x: (x.to_period('Q')))
        metrics = ['AA_eSoV', 'AA_Reach',
       'AA_Brand_Breadth', 'AS_Average_Engagement',
       'AS_Usage_SoV', 'AS_Search_Index',
       'AS_Brand_Centrality','AF_Entry_point','AF_Brand_Love','AF_Baby_Milk','AF_Adverts_Promo','AF_Value_for_Money','AF_Buying_Exp','AF_Prep_Milk','AF_Baby_exp',
        'Framework_Awareness',
       'Framework_Saliency','Framework_Affinity']
        
        if 'average' in method:
            quarterly_metrics_df = pd.DataFrame()
            for mt in metrics:

                # calculate quarterly metric
                quarterly_df = weekly_metrics_df.groupby(['Quarter Commencing', 'brand'])[
                    mt].mean().reset_index()

                if len(quarterly_metrics_df) == 0:
                    quarterly_metrics_df = quarterly_df.copy('deep')
                else:
                    quarterly_metrics_df = pd.merge(quarterly_metrics_df, quarterly_df, on=[
                                                    'Quarter Commencing', 'brand'], how='left')
        elif 'total' in method:
            quarterly_metrics_df = pd.DataFrame()
            for mt in metrics:
                if mt in ['AA_eSoV', 'AA_Reach', 'AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov", 'AF_Motivation for change',
                          'AF_Journey into NGP and brands', 'AF_Brand Prestige and love', 'AF_Value for money', 'AF_Smoking experience', 'AF_Suporting experience', 'AF_Adverts_and_promotions']:

                    quarterly_df = weekly_metrics_df.groupby(['Quarter Commencing', 'brand'])[
                        mt].sum().reset_index()
                else:
                    quarterly_df = weekly_metrics_df.groupby(['Quarter Commencing', 'brand'])[
                        mt].mean().reset_index()

                if len(quarterly_metrics_df) == 0:
                    quarterly_metrics_df = quarterly_df.copy('deep')
                else:
                    quarterly_metrics_df = pd.merge(quarterly_metrics_df, quarterly_df, on=[
                                                    'Quarter Commencing', 'brand'], how='left')

        quarterly_metrics_df['time'] = pd.PeriodIndex(
            quarterly_metrics_df['Quarter Commencing'], freq='Q').to_timestamp()
        quarterly_metrics_df.drop('Quarter Commencing', axis=1, inplace=True)
        weekly_metrics_df.drop('Quarter Commencing', axis=1, inplace=True)
        quarterly_metrics_df['time_period'] = 'quarters'
        quarterly_metrics_df = quarterly_metrics_df[[
            'time', 'time_period', 'brand'] + metrics]
        quarterly_metrics_df.fillna(0, inplace=True)
        return quarterly_metrics_df


    def calculate_halfyearly_metrics_north_star(self,weekly_metrics_df, method):
        
        weekly_metrics_df['Half-year Commencing'] = weekly_metrics_df['time'].apply(self.convert_to_semiannual)
        metrics = ['AA_eSoV', 'AA_Reach',
       'AA_Brand_Breadth', 'AS_Average_Engagement',
       'AS_Usage_SoV', 'AS_Search_Index',
       'AS_Brand_Centrality','AF_Entry_point','AF_Brand_Love','AF_Baby_Milk','AF_Adverts_Promo','AF_Value_for_Money','AF_Buying_Exp','AF_Prep_Milk','AF_Baby_exp',
        'Framework_Awareness',
       'Framework_Saliency','Framework_Affinity']
        
        if 'average' in method:
            halfyearly_metrics_df = pd.DataFrame()
            for mt in metrics:

    

                # calculate halfyearly metric
                halfyearly_df = weekly_metrics_df.groupby(['Half-year Commencing', 'brand'])[mt].mean().reset_index()

    

                if len( halfyearly_metrics_df) == 0:
                    halfyearly_metrics_df = halfyearly_df.copy('deep')
                else:
                    halfyearly_metrics_df = pd.merge(halfyearly_metrics_df, halfyearly_df, on = ['Half-year Commencing', 'brand'], how='left')
        elif 'total' in method:
            halfyearly_metrics_df = pd.DataFrame()
            for mt in metrics:
                if mt in ['AA_eSoV', 'AA_Reach','AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov",'AF_Taste', 
                        'AF_Nutrition','AF_Sustainability', 'AF_Functionality', 'AF_Brand_Strength','AF_Value_for_Money']:

    

                    halfyearly_df = weekly_metrics_df.groupby(['Half-year Commencing', 'brand'])[mt].sum().reset_index()
                else:
                    halfyearly_df = weekly_metrics_df.groupby(['Half-year Commencing', 'brand'])[mt].mean().reset_index()

    

                if len( halfyearly_metrics_df) == 0:
                    halfyearly_metrics_df = halfyearly_df.copy('deep')
                else:
                    halfyearly_metrics_df = pd.merge(halfyearly_metrics_df, halfyearly_df, on = ['Half-year Commencing', 'brand'], how='left')

    

        halfyearly_metrics_df['time'] = halfyearly_metrics_df['Half-year Commencing'].apply(self.convert_semiannual_to_timestamp)
        halfyearly_metrics_df.drop('Half-year Commencing', axis=1, inplace=True)
        weekly_metrics_df.drop('Half-year Commencing', axis=1, inplace=True)
        halfyearly_metrics_df['time_period'] = 'semiannual'
        halfyearly_metrics_df = halfyearly_metrics_df[['time', 'time_period', 'brand'] + metrics]
        halfyearly_metrics_df.fillna(0, inplace=True)
        return halfyearly_metrics_df





    def calculate_yearly_metrics_north_star(self, weekly_metrics_df_original: pd.DataFrame, method: str) -> pd.DataFrame:
        """ Function that takes the weekly metric dataframe and gets the aggregated value. 

        Args:
            weekly_metrics_df_original (pd.DataFrame): Original weekly dataframe before processing
            method (str): Method from the "metrics_calc_method" settings from the config file

        Returns:
            pd.DataFrame: Processed dataframe 
        """
        weekly_metrics_df = weekly_metrics_df_original.copy()
        weekly_metrics_df['Year Commencing'] = weekly_metrics_df['time'].apply(
            lambda x: (x.to_period('Y')))
        metrics = ['AA_eSoV', 'AA_Reach',
       'AA_Brand_Breadth', 'AS_Average_Engagement',
       'AS_Usage_SoV', 'AS_Search_Index',
       'AS_Brand_Centrality','AF_Entry_point','AF_Brand_Love','AF_Baby_Milk','AF_Adverts_Promo','AF_Value_for_Money','AF_Buying_Exp','AF_Prep_Milk','AF_Baby_exp',
        'Framework_Awareness',
       'Framework_Saliency','Framework_Affinity']
        
        if 'average' in method:
            yearly_metrics_df = pd.DataFrame()
            for mt in metrics:

                # calculate yearly metric
                yearly_df = weekly_metrics_df.groupby(['Year Commencing', 'brand'])[
                    mt].mean().reset_index()

                if len(yearly_metrics_df) == 0:
                    yearly_metrics_df = yearly_df .copy('deep')
                else:
                    yearly_metrics_df = pd.merge(yearly_metrics_df, yearly_df, on=[
                                                 'Year Commencing', 'brand'], how='left')
        elif 'total' in method:
            yearly_metrics_df = pd.DataFrame()
            for mt in metrics:
                if mt in ['AA_eSoV', 'AA_Reach', 'AS_Average_Engagement', 'AS_Usage_SoV','AS_Trial_Sov',"Quitting_Sov","Consideration_Sov", 'AF_Motivation for change',
                          'AF_Journey into NGP and brands', 'AF_Brand Prestige and love', 'AF_Value for money', 'AF_Smoking experience', 'AF_Suporting experience', 'AF_Adverts_and_promotions']:

                    yearly_df = weekly_metrics_df.groupby(['Year Commencing', 'brand'])[
                        mt].sum().reset_index()
                else:
                    yearly_df = weekly_metrics_df.groupby(['Year Commencing', 'brand'])[
                        mt].mean().reset_index()

                if len(yearly_metrics_df) == 0:
                    yearly_metrics_df = yearly_df.copy('deep')
                else:
                    yearly_metrics_df = pd.merge(yearly_metrics_df, yearly_df, on=[
                                                 'Year Commencing', 'brand'], how='left')

        yearly_metrics_df['time'] = pd.PeriodIndex(
            yearly_metrics_df['Year Commencing'], freq='Y').to_timestamp()
        yearly_metrics_df.drop('Year Commencing', axis=1, inplace=True)
        weekly_metrics_df.drop('Year Commencing', axis=1, inplace=True)
        yearly_metrics_df['time_period'] = 'years'
        yearly_metrics_df = yearly_metrics_df[[
            'time', 'time_period', 'brand'] + metrics]
        yearly_metrics_df.fillna(0, inplace=True)
        return yearly_metrics_df
