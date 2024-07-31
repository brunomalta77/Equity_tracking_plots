import pandas as pd


def avg_individual_metrics_calculation(dfc: pd.DataFrame, dfs: pd.DataFrame, dfn: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        dfc (pd.DataFrame): _description_
        dfs (pd.DataFrame): _description_
        dfn (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
   # AWARENESS METRICS
    # eSoV = brand mentions/total mentions for a week
    eSoV = (dfc.groupby(['Week Commencing', 'brand'])['mentions'].sum()/dfc.groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'eSoV'})

    # reach = sum of all followers in a week for a brand
    reach = (dfc.groupby(['Week Commencing', 'brand'])['followers'].sum()).reset_index().rename(columns={'followers': 'Reach'})

    # SALIENCY METRICS
    
    # only for testing
    dfc = dfc.loc[dfc["message_type"] != "News"]
    
    # engagement = sum of all earned engagements / total mentions in a week for a brand
    engagement = (dfc[dfc['message_type'].isin(['Instagram', 'Twitter', 'Facebook'])].groupby(['Week Commencing', 'brand'])['earned_engagements'].sum()/dfc[dfc['message_type'].isin(['Instagram', 'Twitter', 'Facebook'])].groupby(['Week Commencing', 'brand'])['mentions'].sum()).reset_index().rename(columns={0: 'Average_Engagement'})

    # usage_SoV = brand user mentions/total user mentions for a week
    usage_SoV = (dfc[dfc['journey_predictions'].isin(["Usage"])].groupby(['Week Commencing', 'brand'])['mentions'].sum()/dfc[dfc['journey_predictions'].isin(["Usage"])].groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'Usage_SoV'})
    
    #trial_SoV
    trial_SoV =  (dfc[dfc['journey_predictions'].isin(["Trial or Experimentation"])].groupby(['Week Commencing', 'brand'])['mentions'].sum()/dfc[dfc['journey_predictions'].isin(["Trial or Experimentation"])].groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'Trial_SoV'})

    #quitting_SoV
    quitting_SoV =  (dfc[dfc['journey_predictions'].isin(["Quitting"])].groupby(['Week Commencing', 'brand'])['mentions'].sum()/dfc[dfc['journey_predictions'].isin(["Quitting"])].groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'Quitting_SoV'})

    #consideration_SoV
    consideration_SoV =  (dfc[dfc['journey_predictions'].isin(["Consideration of buying"])].groupby(['Week Commencing', 'brand'])['mentions'].sum()/dfc[dfc['journey_predictions'].isin(["Consideration of buying"])].groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'Consideration_SoV'})

    # search metric
    dfs['Search_Index'] = dfs['Search_Index'].replace('<1', 0.5)
    dfs['Search_Index'] = dfs['Search_Index'].astype(int)

    # AFFINITY METRICS
    # benefits scores = (positive + neutral - negative) mentions of benefits for brand for a given week

    #taking out the weakest brand
    dfc = dfc.loc[dfc["brand"] != "geekbar"]
    

    aff_benefits = ['brand_prest_lov_brand_love', 'brand_prest_lov_quality','brand_prest_lov_design_innovation', 'brand_prest_lov_customisation',
                    'motivation_change_less_harm_health','motivation_change_health_performance','motivation_change_smell_around_me','motivation_change_greater_freedom',
                    'cons_exp_taste_flavor', 'cons_exp_smell','cons_exp_nicotine_satisfaction', 'cons_exp_consumption_duration','cons_exp_relaxation', 'cons_exp_feeling_cool','cons_exp_shared_moments',
                    'supp_exp_availability','supp_exp_buying_experience', 'supp_exp_customer_service','supp_exp_battery_life', 'supp_exp_convenience']
    
    vfm_benefits = ['vfm_consumable_vfm','vfm_device_vfm', 'vfm_promo']
    
    all_benefits = aff_benefits + vfm_benefits
    # get the positive and negative count
    plus_benefits_scores_aff = dfc[dfc['sentiment'].isin(['Positive',"Neutral"])].groupby(['Week Commencing', 'brand'])[aff_benefits].apply(lambda x: x.astype(int).sum())
    plus_benefits_scores_vfm = dfc[(dfc['sentiment'].isin(['Positive','Neutral']))].groupby(['Week Commencing', 'brand'])[vfm_benefits].apply(lambda x: x.astype(int).sum()) 
    plus_benefits_scores = pd.concat([plus_benefits_scores_aff, plus_benefits_scores_vfm], axis=1).fillna(0) 

    for column in plus_benefits_scores:
        new_name = "Plus_"+str(column)
        plus_benefits_scores = plus_benefits_scores.rename(columns={column: new_name})

    # get the negative count
    minus_benefits_scores_aff = dfc[dfc['sentiment'].isin(['Negative'])].groupby(['Week Commencing', 'brand'])[aff_benefits].apply(lambda x: x.astype(int).sum())
    minus_benefits_scores_vfm = dfc[(dfc['sentiment'].isin(['Negative']))].groupby(['Week Commencing', 'brand'])[vfm_benefits].apply(lambda x: x.astype(int).sum())
    minus_benefits_scores = pd.concat([minus_benefits_scores_aff, minus_benefits_scores_vfm], axis=1).fillna(0)
    

    for column in minus_benefits_scores:
        new_name = "Minus_"+str(column)
        minus_benefits_scores = minus_benefits_scores.rename(columns={column: new_name})

    
    # get the absolute count
    absolute_benefits_scores_aff = dfc[(dfc['sentiment'].isin(['Negative','Positive','Neutral']))].groupby(['Week Commencing', 'brand'])[aff_benefits].apply(lambda x: x.astype(int).sum())
    absolute_benefits_scores_vfm = dfc[(dfc['sentiment'].isin(['Negative','Positive','Neutral']))].groupby(['Week Commencing', 'brand'])[vfm_benefits].apply(lambda x: x.astype(int).sum())
    absolute_benefit_scores = pd.concat([absolute_benefits_scores_aff, absolute_benefits_scores_vfm], axis=1).fillna(0)

    for column in absolute_benefit_scores:
        new_name = "Absolute_"+str(column)
        absolute_benefit_scores = absolute_benefit_scores.rename(columns={column: new_name})

    
    # combine the plus and negative benefits scores
    benefits_scores = pd.concat([absolute_benefit_scores, plus_benefits_scores, minus_benefits_scores], axis=1).fillna(0)

    # subtract one from the other
    for benefit in all_benefits:
        benefits_scores["Net_"+str(benefit)] = benefits_scores["Plus_" +str(benefit)] - benefits_scores["Minus_"+str(benefit)]

    # compute the benefit level scores
    for benefit in all_benefits:
        benefits_scores["Score - "+str(benefit)] = benefits_scores["Net_"+str(benefit)]/benefits_scores["Absolute_"+str(benefit)]

    
    # compute the STAR level metrics - absolute
    benefits_scores['Star - Brand (Absolute)'] = benefits_scores[['Absolute_brand_prest_lov_brand_love','Absolute_brand_prest_lov_quality','Absolute_brand_prest_lov_design_innovation','Absolute_brand_prest_lov_customisation']].sum(axis=1)
    benefits_scores['Star - Change (Absolute)'] = benefits_scores[['Absolute_motivation_change_less_harm_health', 'Absolute_motivation_change_health_performance','Absolute_motivation_change_smell_around_me','Absolute_motivation_change_greater_freedom']].sum(axis=1)
    benefits_scores['Star - Consumption (Absolute)'] = benefits_scores[['Absolute_cons_exp_taste_flavor','Absolute_cons_exp_smell','Absolute_cons_exp_nicotine_satisfaction','Absolute_cons_exp_consumption_duration','Absolute_cons_exp_relaxation','Absolute_cons_exp_feeling_cool','Absolute_cons_exp_shared_moments']].sum(axis=1)
    benefits_scores['Star - Supporting (Absolute)'] = benefits_scores[['Absolute_supp_exp_availability', 'Absolute_supp_exp_buying_experience','Absolute_supp_exp_customer_service','Absolute_supp_exp_battery_life','Absolute_supp_exp_convenience']].sum(axis=1)
    benefits_scores['Star - VFM (Absolute)'] = benefits_scores[['Absolute_vfm_consumable_vfm','Absolute_vfm_device_vfm', 'Absolute_vfm_promo']].sum(axis=1)

    # compute the STAR level metrics - net
    benefits_scores['Star - Brand (Net)'] = benefits_scores[['Net_brand_prest_lov_brand_love','Net_brand_prest_lov_quality','Net_brand_prest_lov_design_innovation','Net_brand_prest_lov_customisation']].sum(axis=1)
    benefits_scores['Star - Change (Net)'] = benefits_scores[['Net_motivation_change_less_harm_health', 'Net_motivation_change_health_performance','Net_motivation_change_smell_around_me','Net_motivation_change_greater_freedom']].sum(axis=1)
    benefits_scores['Star - Consumption (Net)'] = benefits_scores[['Net_cons_exp_taste_flavor','Net_cons_exp_smell','Net_cons_exp_nicotine_satisfaction','Net_cons_exp_consumption_duration','Net_cons_exp_relaxation','Net_cons_exp_feeling_cool','Net_cons_exp_shared_moments']].sum(axis=1)
    benefits_scores['Star - Supporting (Net)'] = benefits_scores[['Net_supp_exp_availability', 'Net_supp_exp_buying_experience','Net_supp_exp_customer_service','Net_supp_exp_battery_life','Net_supp_exp_convenience']].sum(axis=1)
    benefits_scores['Star - VFM (Net)'] = benefits_scores[['Net_vfm_consumable_vfm','Net_vfm_device_vfm', 'Net_vfm_promo']].sum(axis=1)

    # compute the STAR level metrics - score
    benefits_scores['Brand'] = benefits_scores['Star - Brand (Net)'] / benefits_scores['Star - Brand (Absolute)']
    benefits_scores['Change'] = benefits_scores['Star - Change (Net)'] / benefits_scores['Star - Change (Absolute)']
    benefits_scores['Consumption'] = benefits_scores['Star - Consumption (Net)'] / benefits_scores['Star - Consumption (Absolute)']
    benefits_scores['Supporting'] = benefits_scores['Star - Supporting (Net)'] / benefits_scores['Star - Supporting (Absolute)']
    benefits_scores['VFM'] = benefits_scores['Star - VFM (Net)'] / benefits_scores['Star - VFM (Absolute)']

    benefits_scores = benefits_scores.reset_index()
    star_scores = benefits_scores[['Week Commencing', 'brand', 'Brand', 'Change', 'Consumption', 'Supporting', 'VFM']]
    
    # COMBINE - AT A WEEKLY LEVEL
    merge_columns = ['Week Commencing', 'brand']

    weekly_equity = pd.merge(eSoV, reach, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, dfn, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, engagement, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, usage_SoV, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, trial_SoV, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, consideration_SoV, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, quitting_SoV, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, dfs, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, star_scores, on = merge_columns, how='left')
    weekly_equity = weekly_equity.fillna(0)
    
    return weekly_equity


def total_individual_metrics_calculation(dfc: pd.DataFrame, dfs: pd.DataFrame, dfn: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        dfc (pd.DataFrame): _description_
        dfs (pd.DataFrame): _description_
        dfn (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # AWARENESS METRICS
    # eSoV = brand mentions/total mentions for a week
    eSoV = (dfc.groupby(['Week Commencing', 'brand'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'eSoV'})

    # reach = sum of all followers in a week for a brand
    reach = (dfc.groupby(['Week Commencing', 'brand'])['followers'].sum()).reset_index().rename(columns={'followers': 'Reach'})

    # SALIENCY METRICS
    dfc = dfc.loc[dfc["message_type"] != "News"]
    # engagement = sum of all earned engagements / total mentions in a week for a brand
    engagement = (dfc[dfc['message_type'].isin(['Instagram', 'Twitter', 'Facebook'])].groupby(['Week Commencing', 'brand'])['earned_engagements'].sum()).reset_index().rename(columns={'earned_engagements': 'Average_Engagement'})

    # usage_SoV = brand user mentions/total user mentions for a week
    usage_SoV = (dfc[dfc['journey_predictions'].isin(["Usage"])].groupby(['Week Commencing', 'brand'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'Usage_SoV'})

    #trial_SoV
    trial_SoV =  (dfc[dfc['journey_predictions'].isin(["Trial or Experimentation"])].groupby(['Week Commencing', 'brand'])['mentions'].sum()/dfc[dfc['journey_predictions'].isin(["Trial or Experimentation"])].groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'Trial_SoV'})

    #quitting_SoV
    quitting_SoV =  (dfc[dfc['journey_predictions'].isin(["Quitting"])].groupby(['Week Commencing', 'brand'])['mentions'].sum()/dfc[dfc['journey_predictions'].isin(["Quitting"])].groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'Quitting_SoV'})

    #consideration_SoV
    consideration_SoV =  (dfc[dfc['journey_predictions'].isin(["Consideration of buying"])].groupby(['Week Commencing', 'brand'])['mentions'].sum()/dfc[dfc['journey_predictions'].isin(["Consideration of buying"])].groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'Consideration_SoV'})

    # search metric
    dfs['Search_Index'] = dfs['Search_Index'].replace('<1', 0.5)
    dfs['Search_Index'] = dfs['Search_Index'].astype(int)

    # AFFINITY METRICS
    # benefits scores = (positive + neutral - negative) mentions of benefits for brand for a given week
    
    #for testing
    dfc = dfc.loc[dfc["brand"] != "geekbar"]
    
    aff_benefits = ['brand_prest_lov_brand_love', 'brand_prest_lov_quality','brand_prest_lov_design_innovation', 'brand_prest_lov_customisation',
                    'motivation_change_less_harm_health','motivation_change_health_performance','motivation_change_smell_around_me','motivation_change_greater_freedom',
                    'cons_exp_taste_flavor', 'cons_exp_smell','cons_exp_nicotine_satisfaction', 'cons_exp_consumption_duration','cons_exp_relaxation', 'cons_exp_feeling_cool','cons_exp_shared_moments',
                    'supp_exp_availability','supp_exp_buying_experience', 'supp_exp_customer_service','supp_exp_battery_life', 'supp_exp_convenience']
    vfm_benefits = ['vfm_consumable_vfm','vfm_device_vfm', 'vfm_promo']
    all_benefits = aff_benefits + vfm_benefits


    # get the positive and neutral count
    plus_benefits_scores_aff = dfc[dfc['sentiment'].isin(['Positive','Neutral'])].groupby(['Week Commencing', 'brand'])[aff_benefits].apply(lambda x : x.astype(int).sum())
    plus_benefits_scores_vfm = dfc[dfc['sentiment'].isin(['Positive','Neutral'])].groupby(['Week Commencing', 'brand'])[vfm_benefits].apply(lambda x : x.astype(int).sum())
    plus_benefits_scores = pd.concat([plus_benefits_scores_aff, plus_benefits_scores_vfm], axis=1).fillna(0)
    
    for column in plus_benefits_scores:
        new_name = "Plus_"+str(column)
        plus_benefits_scores = plus_benefits_scores.rename(columns = {column:new_name})

    # subtract one from the other
    for benefit in all_benefits:
        plus_benefits_scores["Net_"+str(benefit)] = plus_benefits_scores["Plus_"+str(benefit)]

    # compute the benefit level scores
    for benefit in all_benefits:
        plus_benefits_scores["Score - "+str(benefit)] = plus_benefits_scores["Net_"+str(benefit)]

    # compute the STAR level metrics - net
    plus_benefits_scores['Brand'] = plus_benefits_scores[['Net_brand_prest_lov_brand_love','Net_brand_prest_lov_quality','Net_brand_prest_lov_design_innovation','Net_brand_prest_lov_customisation']].sum(axis=1)
    plus_benefits_scores['Change'] = plus_benefits_scores[['Net_motivation_change_less_harm_health', 'Net_motivation_change_health_performance','Net_motivation_change_smell_around_me','Net_motivation_change_greater_freedom']].sum(axis=1)
    plus_benefits_scores['Consumption'] = plus_benefits_scores[['Net_cons_exp_taste_flavor','Net_cons_exp_smell','Net_cons_exp_nicotine_satisfaction','Net_cons_exp_consumption_duration','Net_cons_exp_relaxation','Net_cons_exp_feeling_cool','Net_cons_exp_shared_moments']].sum(axis=1)
    plus_benefits_scores['Supporting'] = plus_benefits_scores[['Net_supp_exp_availability', 'Net_supp_exp_buying_experience','Net_supp_exp_customer_service','Net_supp_exp_battery_life','Net_supp_exp_convenience']].sum(axis=1)
    plus_benefits_scores['VFM'] = plus_benefits_scores[['Net_vfm_consumable_vfm','Net_vfm_device_vfm', 'Net_vfm_promo']].sum(axis=1)


    plus_benefits_scores = plus_benefits_scores.reset_index()
    star_scores = plus_benefits_scores[['Week Commencing', 'brand', 'Brand', 'Change', 'Consumption', 'Supporting', 'VFM']]
    
    # COMBINE - AT A WEEKLY LEVEL
    merge_columns = ['Week Commencing', 'brand']

    weekly_equity = pd.merge(eSoV, reach, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, dfn, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, engagement, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, usage_SoV, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, trial_SoV, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, consideration_SoV, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, quitting_SoV, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, dfs, on = merge_columns, how='left')
    weekly_equity = pd.merge(weekly_equity, star_scores, on = merge_columns, how='left')
    weekly_equity = weekly_equity.fillna(0)
    
    return weekly_equity
