import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import re
import time
import string
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import requests
import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os 
from PIL import Image
import msal
import io
import requests
from metrics import AggregateMetrics
#--------------------------------------------------------------------------------------// Aesthetic Global Variables // -------------------------------------------------------------------------

user_to_equity = {'Entry points & Key Moments':'AF_Entry_point','Brand Prestige & Love':'AF_Brand_Love','Baby Milk':'AF_Baby_Milk','Adverts and Promotions':'AF_Adverts_Promo','Value For Money':'AF_Value_for_Money',
                        'Buying Experience': 'AF_Buying_Exp','Preparing Milk':'AF_Prep_Milk','Baby Experience':'AF_Baby_exp','Total Equity':'Total_Equity',"Awareness":'Framework_Awareness','Saliency':'Framework_Saliency','Affinity':'Framework_Affinity','eSoV':'AA_eSoV', 'Reach':'AA_Reach',
       'Brand Breadth': 'AA_Brand_Breadth', 'Average Engagement':'AS_Average_Engagement', 'Usage SoV':'AS_Usage_SoV',
       'Search Index': 'AS_Search_Index', 'Brand Centrality':'AS_Brand_Centrality'}

affinity_labels = ['AF_Entry_point','AF_Brand_Love','AF_Baby_Milk','AF_Adverts_Promo','AF_Value_for_Money','AF_Buying_Exp','AF_Prep_Milk','AF_Baby_exp']



framework_to_user = {'Total_Equity':'Total Equity','Framework_Awareness':"Awareness",'Framework_Saliency':'Saliency','Framework_Affinity':'Affinity','AA_eSoV':'eSoV', 'AA_Reach':'Reach',
       'AA_Brand_Breadth':'Brand Breadth', 'AS_Average_Engagement':'Average Engagement', 'AS_Usage_SoV':'Usage SoV',
       'AS_Search_Index':'Search Index', 'AS_Brand_Centrality':'Brand Centrality'}


categories_changed = {"baby_milk":"Baby Milk"}


framework_options_ = ["Total Equity","Awareness","Saliency","Affinity",'Entry points & Key Moments','Brand Prestige & Love','Baby Milk','Adverts and Promotions','Value For Money',
                                'Buying Experience','Preparing Milk','Baby Experience']


affinity_to_user = {'AF_Entry_point':'Entry points & Key Moments','AF_Brand_Love':'Brand Prestige & Love','AF_Baby_Milk':'Baby Milk','AF_Adverts_Promo':'Adverts and Promotions','AF_Value_for_Money':'Value For Money',
                                'AF_Buying_Exp':'Buying Experience','AF_Prep_Milk':'Preparing Milk','AF_Baby_exp':'Baby Experience'}

general_equity_to_user = {'Total_Equity':'Total Equity','Framework_Awareness':'Awareness','Framework_Saliency':'Saliency','Framework_Affinity':'Affinity'}


value_columns_  = [ 'Total Equity','Awareness', 'Saliency', 'Affinity',
        'eSoV', 'Reach',
        'Brand Breadth', 'Average Engagement', 'Usage SoV',
        'Search Index', 'Brand Centrality','Entry points & Key Moments','Brand Prestige & Love','Baby Milk','Adverts and Promotions','Value For Money',
                                    'Buying Experience','Preparing Milk','Baby Experience']

join_data_average = ['time', 'time_period', 'brand', 'AA_eSoV_average', 'AA_Reach_average',
                'AA_Brand_Breadth_average', 'AS_Average_Engagement_average',
                'AS_Usage_SoV_average', 'AS_Search_Index_average',
                'AS_Brand_Centrality_average','AF_Entry_point_average', 'AF_Brand_Love_average', 'AF_Baby_Milk_average','AF_Adverts_Promo_average','AF_Value_for_Money_average','AF_Buying_Exp_average',
                'AF_Prep_Milk_average','AF_Baby_exp_average',
                'Framework_Awareness_average', 'Framework_Saliency_average',
                'Framework_Affinity_average', 'Total_Equity_average',
                'Category_average']


join_data_total = ['time', 'time_period', 'brand', 'AA_eSoV_total', 'AA_Reach_total',
                  'AA_Brand_Breadth_total', 'AS_Average_Engagement_total',
                  'AS_Usage_SoV_total', 'AS_Search_Index_total',
                  'AS_Brand_Centrality_total','AF_Entry_point_total', 'AF_Brand_Love_total', 'AF_Baby_Milk_total','AF_Adverts_Promo_total','AF_Value_for_Money_total','AF_Buying_Exp_total',
                  'AF_Prep_Milk_total','AF_Baby_exp_total',
                  'Framework_Awareness_total', 'Framework_Saliency_total',
                  'Framework_Affinity_total', 'Total_Equity_total', 'Category_total']



list_fix = ['time', 'time_period', 'brand', 'AA_eSoV_average', 'AA_Reach_average',
                  'AA_Brand_Breadth_average', 'AS_Average_Engagement_average',
                  'AS_Usage_SoV_average', 'AS_Search_Index_average',
                  'AS_Brand_Centrality_average','Framework_Awareness_average', 'Framework_Saliency_average','Total_Equity_average',
                  'Category_average']


order_list = ['time', 'time_period', 'brand', 'AA_eSoV_average', 'AA_Reach_average',
       'AA_Brand_Breadth_average', 'AS_Average_Engagement_average',
       'AS_Usage_SoV_average', 'AS_Search_Index_average',
       'AS_Brand_Centrality_average','weighted_AF_Entry_point','weighted_AF_Brand_Love','weighted_AF_Baby_Milk','weighted_AF_Adverts_Promo','weighted_AF_Value_for_Money','weighted_AF_Buying_Exp','weighted_AF_Prep_Milk','weighted_AF_Baby_exp',
        'Framework_Awareness_average',
       'Framework_Saliency_average','weighted_Framework_Affinity','Total_Equity',"Category_average"]

rename_all = {'AA_eSoV_average':'AA_eSoV', 'AA_Reach_average':'AA_Reach',
       'AA_Brand_Breadth_average':'AA_Brand_Breadth', 'AS_Average_Engagement_average':'AS_Average_Engagement',
       'AS_Usage_SoV_average':'AS_Usage_SoV', 'AS_Search_Index_average':'AS_Search_Index',
       'AS_Brand_Centrality_average':'AS_Brand_Centrality','weighted_AF_Entry_point':'AF_Entry_point','weighted_AF_Brand_Love':'AF_Brand_Love',
       'weighted_AF_Brand_Love':'AF_Brand_Love','weighted_AF_Baby_Milk':'AF_Baby_Milk','weighted_AF_Buying_Exp':'AF_Buying_Exp','weighted_AF_Prep_Milk':'AF_Prep_Milk'
       ,'weighted_AF_Baby_exp':'AF_Baby_exp',
       'weighted_AF_Adverts_Promo':'AF_Adverts_Promo',
       'weighted_AF_Value_for_Money':'AF_Value_for_Money','Framework_Awareness_average':'Framework_Awareness',
       'Framework_Saliency_average':'Framework_Saliency','weighted_Framework_Affinity':'Framework_Affinity','Category_average':'Category'}


smoothening_weeks_list = ['Total Equity','Awareness','Saliency','eSoV', 'Reach','Brand Breadth', 'Average Engagement',
       'Usage SoV', 'Search Index','Affinity','Entry points & Key Moments','Brand Prestige & Love','Baby Milk','Adverts and Promotions','Value For Money',
                                'Buying Experience','Preparing Milk','Baby Experience']


############ -------------------------------------------------------------------------Equity Analysis config ----------------------------------------------------------------------######
aff_metrics_analysis = ["life_entry_expecting_parents","life_entry_baby_lack_of_sleep",
                   "life_entry_feeding_frequency","life_entry_point_Pressure_to_provide_best",
                   "life_entry_allergies","life_entry_breasts_uncom","life_entry_going_back_to_work",
                   "brand_prest_love_reputation","brand_prest_love_trust",
                   "brand_prest_love_love","brand_prest_love_transparency",
                   "baby_milk_ingredients","baby_milk_premium_quality",
                   "baby_milk_organic","baby_milk_clinical_benefits",
                   "vfm_price_cuts_discounts","vmf_core_base_price",
                   "buying_exp_availability","buying_exp_delivery_exp",
                   "prep_milk_convenience","prep_milk_packaging",
                   "baby_exp_taste_aftertaste_smell",
                   "baby_exp_texture","baby_exp_comfort_hapiness",
                   "baby_exp_easy_to_digest","adverts_promo_product_launch"]


aff_metrics_pillars_analysis = ['life_entry','brand_prest_lov','baby_milk','vfm','buying_exp','prep_milk','baby_exp','adverts_promo']

aw_metrics = ["AF_Entry_point","AF_Brand_Love","AF_Baby_Milk","AF_Adverts_Promo","AF_Value_for_Money","AF_Buying_Exp",
                  "AF_Prep_Milk","AF_Baby_exp"]



#--------------------------------------------------------------------------------------// Aesthetic Global Variables // -------------------------------------------------------------------------
#page config
st.set_page_config(page_title="Equity Tracking plots app",page_icon="💼",layout="wide")
logo_path = r"data/brand_logo.png"
logo_microsoft_path =  r"https://www.shareicon.net/data/256x256/2015/09/15/101518_microsoft_512x512.png"
image = Image.open(logo_path)

#colors used for the plots
colors = ["blue", "green", "red", "purple", "orange","teal","black","paleturquoise","indigo","darkseagreen","gold","darkviolet","firebrick","navy","deeppink",
         "orangered"]


# creating a user database type for getting access to the app -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Microsoft Azure AD configurations
CLIENT_ID = "baa3d4a8-3aa7-45bd-9245-93e8610b2b84"
CLIENT_SECRET = "RBm8Q~fsy.iJ1_afdMITcNIgGe~n~mDmAx9cSaBO"
AUTHORITY = "https://login.microsoftonline.com/68421f43-a2e1-4c77-90f4-e12a5c7e0dbc"
SCOPE = ["User.Read"]
REDIRECT_URI = "https://equitytrackingplots-jqwdds7kl4pnw98dcmxfbz.streamlit.app/" # This should match your Azure AD app configuration

# Initialize MSAL application
app = msal.ConfidentialClientApplication(
    CLIENT_ID, authority=AUTHORITY,
    client_credential=CLIENT_SECRET)

def get_auth_url():
    return app.get_authorization_request_url(SCOPE, redirect_uri=REDIRECT_URI)

def get_token_from_code(code):
    try:
        result = app.acquire_token_by_authorization_code(code, SCOPE, redirect_uri=REDIRECT_URI)
        if "access_token" in result:
            return result["access_token"]
        else:
            st.error(f"Failed to acquire token. Error: {result.get('error')}")
            st.error(f"Error description: {result.get('error_description')}")
            return None
    except Exception as e:
        st.error(f"An exception occurred: {str(e)}")
        return None

#def get_user_info(access_token):
#    headers = {'Authorization': f'Bearer {access_token}'}
#    response = requests.get('https://graph.microsoft.com/v1.0/me', headers=headers)
#    return response.json()

def login():
         auth_url = get_auth_url()
         #st.markdown(f'[Login with Microsoft]({auth_url})')
         html_string = f"""
         <a href="{auth_url}">
             <img src="{logo_microsoft_path}" style="width: 20px; height: 20px; vertical-align: middle;">
                Log in with Microsoft
         </a>
         """

         # Use st.markdown to render the HTML
         st.markdown(html_string, unsafe_allow_html=True)


def get_user_info(access_token):
       headers = {'Authorization': f'Bearer {access_token}'}
       response = requests.get('https://graph.microsoft.com/v1.0/me', headers=headers)
       user_info = response.json()
       return user_info.get('mail') or user_info.get('userPrincipalName')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


col1, col2 = st.columns([4, 1])  # Adjust the width ratios as needed

# Logo on the left
#with col2:
    #st.image(image)  # Adjust the width as needed

# Title on the right
with col1:
    st.title("Danone - Equity Tracking Plots")


# getting the excel file first by user input
data = r"data"
media_data = r"data/Media_invest_all.xlsx"


# equity file
@st.cache_data() 
def reading_df(filepath,sheet_name):
    df = pd.read_excel(filepath,sheet_name=sheet_name)
    return


#Some info
awareness_metrics =  ["eSoV", "Reach", "Brand_Breadth"]
saliency_metrics = ["Average_Engagement","Usage_SoV","Trial_SoV","Quitting_SoV","Consideration_SoV","Search_Index","Brand_Centrality"]
affinity_metrics = ["Brand","Change","Consumption","Supporting","VFM"]
metrics_calc_method =  ["average_smoothened","total_smoothened","average_unsmoothened","total_unsmoothened","weighted_average"]
smoothening_parameters = {"window_size": [12] }
index_brand= {"vape": "elfbar"}
weights = {
"awareness": [0.5, 0.5, 0],
"saliency": [0.2, 0.2, 0.2, 0,0, 0.2, 0.2],
"affinity": [0.2, 0.2, 0.2, 0.2, 0.2],
"weighted_avg":0.75,
"weighted_total":0.25}

#Instatiate necessary classes
MetricsClass = AggregateMetrics(
    smoothening_parameters=smoothening_parameters,
    awareness_metrics=awareness_metrics,
    saliency_metrics=saliency_metrics,
    affinity_metrics=affinity_metrics,
    weigths=weights)

@st.cache_data()
def get_weighted(df,df_total_uns,weighted_avg,weighted_total,brand_replacement,user_to_equity,affinity_labels,join_data_average,join_data_total,list_fix,order_list,rename_all):
    #------------------------------------------------------------------------------------------------------------------------------------------------------
    df.rename(columns=user_to_equity,inplace=True)

    df_total_uns.rename(columns=user_to_equity,inplace=True)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # drop any nan values
    df.dropna(inplace=True)
    df_total_uns.dropna(inplace=True)

    df_total_uns.brand = df_total_uns.brand.replace(brand_replacement)

    replacements = {"weeks":"Weeks","months":"Months","quarters":"Quarters","semiannual":"Semiannual","years":"Years"}
    df_total_uns["time_period"] = df_total_uns["time_period"].replace(replacements)
    
    affinity_labels = affinity_labels
    
    # Doing the percentual in total_unsmoothened
    for aff in affinity_labels:
        grouped = df_total_uns.groupby(["time","time_period"])[aff].transform("sum")
        df_total_uns["total"] = grouped
        df_total_uns[aff] = df_total_uns[aff] / df_total_uns['total'] * 100

    # Let's join by time and brand
    join_data = pd.merge(df,df_total_uns,on=["time","brand","time_period"],suffixes=("_average","_total"))

    #splitting them 

    final_average = join_data[join_data_average]


    final_total = join_data[join_data_total]

    list_fix = list_fix

    #Getting first the fixed stuff
    weighted_average_equity = final_average[list_fix]

    for aff_pilar in affinity_labels:
        weighted_average_equity["weighted_" + aff_pilar] = 0
        for index,row in final_average.iterrows():
            weighted_average_equity["weighted_" + aff_pilar][index] = round(((weighted_avg * final_average[aff_pilar + "_average"][index]) + (weighted_total * final_total[aff_pilar + "_total"][index])),2)
        
    # Select columns that start with 'weighted_AF_'
    affinity_columns = [col for col in weighted_average_equity.columns if col.startswith('weighted_AF_')]

    # Calculate the weighted Framework Affinity
    weighted_average_equity["weighted_Framework_Affinity"] = round(weighted_average_equity[affinity_columns].mean(axis=1), 2)
    

    # getting the new total equity

    weighted_average_equity["Total_Equity"] = round((weighted_average_equity["weighted_Framework_Affinity"] + weighted_average_equity["Framework_Awareness_average"] + weighted_average_equity["Framework_Saliency_average"])/3,2) 

    #ordering
    order = order_list
    weighted_average_equity = weighted_average_equity[order]

    weighted_average_equity.rename(columns=rename_all,inplace=True)

    return weighted_average_equity


#---------------------------------------------------------------------------------------////--------------------------------------------------------------------------------------------------

# Market_share_weighted_average
def weighted_brand_calculation(df_original,weights_joined,years, value_columns,framework_to_user):
    concat_data=[]
    for year,weights in zip(years,weights_joined):
        #filter by year
        df = df_original[(df_original.time >= f"{year}-01-01") & (df_original.time <= f"{year}-12-31")]
        df.rename(columns=framework_to_user,inplace=True)
        # Convert value columns to numeric, replacing non-numeric values with NaN
        for col in value_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Apply weights to each brand
        for brand, weight in weights.items():
            mask = (df['brand'] == brand)
            df.loc[mask, value_columns] = df.loc[mask, value_columns].multiply(weight)
        
        # Group by time_period and time, then normalize
        def normalize_group(group):
            totals = group[value_columns].sum()
            for col in value_columns:
                if totals[col] == 0:
                    group[col] = 0
                else:
                    group[col] = round((group[col] / totals[col]) * 100,2)
            return group

        result_df = df.groupby(['time_period', 'time']).apply(normalize_group).reset_index(drop=True)
        
        
        concat_data.append(result_df)

    final_df = pd.concat(concat_data,axis=0)
    return final_df
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def equity_info(data,market_flag):
    if market_flag == "uk":
        market_flag = "danone_uk_equity_"
    for x in os.listdir(data):
        if market_flag in x:
            filepath_equity = os.path.join(data,x)
            info_number = [x for x in x.split("_") if x >= "0" and x <="9"]
            year_equity,month_equity,day_equity,hour_equity,minute_equity = info_number[:5]
            second_equity = info_number[-1].split(".")[0]
    
    return filepath_equity,year_equity,month_equity,day_equity,hour_equity,minute_equity,second_equity

def equity_options(df,brand_mapping,categories_changed,framework_options_):
         df.brand = df.brand.replace(brand_mapping)
         
         
         df["Category"] = df["Category"].replace(categories_changed)
         category_options = df["Category"].unique()
         
         replacements = {"weeks":"Weeks","months":"Months","quarters":"Quarters","semiannual":"Semiannual","years":"Years"}
         df["time_period"] = df["time_period"].replace(replacements)
         time_period_options = df["time_period"].unique()
         
         framework_options = framework_options_
         
         return (category_options,time_period_options,framework_options)
         
#-----------------------------------------------------------------------------------------------------//-----------------------------------------------------------------------------------------
# Equity_plot
def Equity_plot(df,categories,time_frames,frameworks,sheet_name,framework_to_user,brand_color_mapping,category):
    if sheet_name == "Average Smoothening":
        name = "Average"
    if sheet_name == "Total Unsmoothening":
        name = "Absolute"
    if sheet_name == "Market Share Weighted":
        name = "Market Share Weighted"

    df.rename(columns=framework_to_user,inplace=True)

    
    st.subheader(f"Final Equity plot - {name}")

    # creating the columns for the app
    right_column_1,right_column_2,left_column_1,left_column_2 = st.columns(4)
    
    with right_column_1:
    #getting the date
        start_date = st.date_input("Select start date",value=datetime(2021, 2, 15))
        end_date =  st.date_input("Select end date")
        #convert our dates
        ws = start_date.strftime('%Y-%m-%d')
        we = end_date.strftime('%Y-%m-%d')
    # getting the parameters
    with right_column_2:
        category = category
        
    with left_column_1:    
        time_frame = st.radio('Choose  time frame:', time_frames)
    
    with left_column_2:
        framework = st.selectbox('Choose  metric:', frameworks)
    
    #filtering
    df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    
    df_filtered = df_filtered.sort_values(by="time")
    
    
    # color stuff
    #all_brands = [x for x in df["brand"].unique()]
    #colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","yellow","olive","silver","darkviolet","grey"]

    #brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
    
    fig = px.line(df_filtered, x="time", y=framework, color="brand", color_discrete_map=brand_color_mapping)

    
    if time_frame == "Months":
        unique_months = df_filtered['time'].dt.to_period('M').unique()

        # Customize the x-axis tick labels to show one label per month
        tickvals = [f"{m.start_time}" for m in unique_months]
        ticktext = [m.strftime("%B %Y") for m in unique_months]

        # Update x-axis ticks
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
        
        return fig

    if time_frame == "Quarters":

        unique_quarters = df_filtered['time'].dt.to_period('Q').unique()

        # Customize the x-axis tick labels to show one label per quarter
        tickvals = [f"{q.start_time}" for q in unique_quarters]
        ticktext = [f"Q{q.quarter} {q.year}" for q in unique_quarters]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
        
        return fig


    if time_frame =="Years":
        # Extract unique years from the "time" column
        unique_years = df_filtered['time'].dt.year.unique()

        # Customize the x-axis tick labels to show only one label per year
        fig.update_xaxes(tickvals=[f"{year}-01-01" for year in unique_years], ticktext=unique_years, tickangle=45)
        
        return fig


    if time_frame == "Weeks":
        # Extract unique weeks from the "time" column
        unique_weeks = pd.date_range(start=ws, end=we, freq='W').date

        # Customize the x-axis tick labels to show the start date of each week
        tickvals = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]
        ticktext = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)

        return fig

    else:
        # Extract unique semiannual periods from the "time" column
        unique_periods = pd.date_range(start=ws, end=we, freq='6M').date

        # Customize the x-axis tick labels to show the start date of each semiannual period
        tickvals = [period.strftime('%Y-%m-%d') for period in unique_periods]
        ticktext = [f"Semiannual {i // 2 + 1} - {period.strftime('%Y')}" for i, period in enumerate(unique_periods)]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)

        return fig


#-----------------------------------------------------------------------------------------------//----------------------------------------------------------------------------------------------


# Equity_plot for market share weighted average

def Equity_plot_market_share_(df,category,time_frame,framework,ws,we,brand_color_mapping):
   
    #filtering
    df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    
    df_filtered = df_filtered.sort_values(by="time")
    
    
    # color stuff
    #all_brands = [x for x in df["brand"].unique()]
    #colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","yellow","olive","silver","darkviolet","grey"]

    #brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
    
    fig = px.line(df_filtered, x="time", y=framework, color="brand", color_discrete_map=brand_color_mapping)

    
    if time_frame == "Months":
        unique_months = df_filtered['time'].dt.to_period('M').unique()

        # Customize the x-axis tick labels to show one label per month
        tickvals = [f"{m.start_time}" for m in unique_months]
        ticktext = [m.strftime("%B %Y") for m in unique_months]

        # Update x-axis ticks
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
        
        return fig

    if time_frame == "Quarters":

        unique_quarters = df_filtered['time'].dt.to_period('Q').unique()

        # Customize the x-axis tick labels to show one label per quarter
        tickvals = [f"{q.start_time}" for q in unique_quarters]
        ticktext = [f"Q{q.quarter} {q.year}" for q in unique_quarters]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
        
        return fig


    if time_frame =="Years":
        # Extract unique years from the "time" column
        unique_years = df_filtered['time'].dt.year.unique()

        # Customize the x-axis tick labels to show only one label per year
        fig.update_xaxes(tickvals=[f"{year}-01-01" for year in unique_years], ticktext=unique_years, tickangle=45)
        
        return fig


    if time_frame == "Weeks":
        # Extract unique weeks from the "time" column
        unique_weeks = pd.date_range(start=ws, end=we, freq='W').date

        # Customize the x-axis tick labels to show the start date of each week
        tickvals = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]
        ticktext = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)

        return fig

    else:
        # Extract unique semiannual periods from the "time" column
        unique_periods = pd.date_range(start=ws, end=we, freq='6M').date

        # Customize the x-axis tick labels to show the start date of each semiannual period
        tickvals = [period.strftime('%Y-%m-%d') for period in unique_periods]
        ticktext = [f"Semiannual {i // 2 + 1} - {period.strftime('%Y')}" for i, period in enumerate(unique_periods)]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)

        return fig
#-----------------------------------------------------------------------------------------------//----------------------------------------------------------------------------------------------


#Used to comparing the Equity from different sheets
def Comparing_Equity(df,df_total_uns,weighted_df,categories,time_frames,frameworks,brand_replacement,affinity_to_user,categories_changed,general_equity_to_user,category):
    st.subheader(f"Compare Average, Absolute and Market Share Weighted")
    
    # ------------------------------------------------------------------------------------------------Aesthetic changes-------------------------------------------------------------------------
    #changing the names of the filtered  columns
    ################################################################## df ####################################################################################################
    df.rename(columns=affinity_to_user,inplace=True)

    df.brand = df.brand.replace(brand_replacement)
    
    df.rename(columns=general_equity_to_user,inplace=True)

    ################################################################## df_total_uns ####################################################################################################

    df_total_uns.rename(columns=affinity_to_user,inplace=True)


    df_total_uns.brand = df_total_uns.brand.replace(brand_replacement)

    replacements = {"weeks":"Weeks","months":"Months","quarters":"Quarters","semiannual":"Semiannual","years":"Years"}
    df_total_uns["time_period"] = df_total_uns["time_period"].replace(replacements)


    df_total_uns["Category"] = df_total_uns["Category"].replace(categories_changed)

    df_total_uns.rename(columns=general_equity_to_user,inplace=True)

    ################################################################## weighted_df ####################################################################################################


    weighted_df.rename(columns=affinity_to_user,inplace=True)

    weighted_df.brand = weighted_df.brand.replace(brand_replacement)

    weighted_df.rename(columns=general_equity_to_user,inplace=True)

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # creating the columns for the app
    right_column_1,right_column_2,left_column_1,left_column_2 = st.columns(4)
    
    with right_column_1:
    #getting the date
        start_date = st.date_input("Select start date",value=datetime(2021, 2, 15),key="test_1")
        end_date =  st.date_input("Select end date",key='test_2')
        #convert our dates
        ws = start_date.strftime('%Y-%m-%d')
        we = end_date.strftime('%Y-%m-%d')
    # getting the parameters
    with right_column_2:
        category = category
        
    with left_column_1:    
        time_frame = st.radio('Choose  time frame:', time_frames,key="test_4")
    
    with left_column_2:
        framework = st.selectbox('Choose  framework:', frameworks,key="test_5")
        my_brand = st.multiselect('Choose  brand',df.brand.unique())
    
    #filtering all the dataframes
    #Average
    df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    df_filtered = df_filtered.sort_values(by="time")
    df_filtered = df_filtered[df_filtered["brand"].isin(my_brand)]

    
    #Total Unsmoothening
    df_filtered_uns =  df_total_uns[(df_total_uns["Category"] == category) & (df_total_uns["time_period"] == time_frame)]
    df_filtered_uns = df_filtered_uns[(df_filtered_uns['time'] >= ws) & (df_filtered_uns['time'] <= we)]
    df_filtered_uns = df_filtered_uns.sort_values(by="time")
    df_filtered_uns = df_filtered_uns[df_filtered_uns["brand"].isin(my_brand)]

    #Weighted
    df_filtered_weighted =  weighted_df[(weighted_df["Category"] == category) & (weighted_df["time_period"] == time_frame)]
    df_filtered_weighted = df_filtered_weighted[(df_filtered_weighted['time'] >= ws) & (df_filtered_weighted['time'] <= we)]
    df_filtered_weighted = df_filtered_weighted.sort_values(by="time")
    df_filtered_weighted = df_filtered_weighted[df_filtered_weighted["brand"].isin(my_brand)]
    
    # color stuff
    all_brands = [x for x in df["brand"].unique()]
    colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","yellow","olive","silver","darkviolet","grey"]

    brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
    
    fig = px.line()

    # Add traces for the first dataset (Average Smoothing)
    for brand in df_filtered["brand"].unique():
        brand_data = df_filtered[df_filtered["brand"] == brand]
        fig.add_trace(go.Scatter(
            x=brand_data["time"],
            y=brand_data[framework],
            mode="lines",
            name=f"{brand} (Average)",
            line=dict(color=brand_color_mapping[brand]),
        ))

    # Add traces for the second dataset (Total Unsmoothing)
    for brand in df_filtered_uns["brand"].unique():
        brand_data = df_filtered_uns[df_filtered_uns["brand"] == brand]
        fig.add_trace(go.Scatter(
            x=brand_data["time"],
            y=brand_data[framework],
            mode="lines",
            name=f"{brand} (Absolute)",
            line=dict(color=brand_color_mapping[brand], dash= "dot"),

        ))

    # Add traces for the third dataset (Weighted)
    for brand in df_filtered_weighted["brand"].unique():
        brand_data = df_filtered_weighted[df_filtered_weighted["brand"] == brand]
        fig.add_trace(go.Scatter(
            x=brand_data["time"],
            y=brand_data[framework],
            mode="markers",
            name=f"{brand} (Weighted)",
            line=dict(color=brand_color_mapping[brand]),

        ))

    
    if time_frame == "Months":
        unique_months = df_filtered['time'].dt.to_period('M').unique()

        # Customize the x-axis tick labels to show one label per month
        tickvals = [f"{m.start_time}" for m in unique_months]
        ticktext = [m.strftime("%B %Y") for m in unique_months]

        # Update x-axis ticks
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
        
        return fig

    if time_frame == "Quarters":

        unique_quarters = df_filtered['time'].dt.to_period('Q').unique()

        # Customize the x-axis tick labels to show one label per quarter
        tickvals = [f"{q.start_time}" for q in unique_quarters]
        ticktext = [f"Q{q.quarter} {q.year}" for q in unique_quarters]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
        
        return fig


    if time_frame =="Years":
        # Extract unique years from the "time" column
        unique_years = df_filtered['time'].dt.year.unique()

        # Customize the x-axis tick labels to show only one label per year
        fig.update_xaxes(tickvals=[f"{year}-01-01" for year in unique_years], ticktext=unique_years, tickangle=45)
        
        return fig


    if time_frame == "Weeks":
        # Extract unique weeks from the "time" column
        unique_weeks = pd.date_range(start=ws, end=we, freq='W').date

        # Customize the x-axis tick labels to show the start date of each week
        tickvals = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]
        ticktext = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)

        return fig

    else:
        # Extract unique semiannual periods from the "time" column
        unique_periods = pd.date_range(start=ws, end=we, freq='6M').date

        # Customize the x-axis tick labels to show the start date of each semiannual period
        tickvals = [period.strftime('%Y-%m-%d') for period in unique_periods]
        ticktext = [f"Semiannual {i // 2 + 1} - {period.strftime('%Y')}" for i, period in enumerate(unique_periods)]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)

        return fig




def smoothening_weeks(df,variables,affinity_to_user,framework_to_user,original_category,categories_changed,brand_mapping,window,method= 'average'): 
    columns_to_multiply = [x for x in df.columns if "AA" in x  or "AS" in x  or "AF" in x ]
    

    # Aplicar isto desde o início. 
    df_weeks = df[df.time_period == "Weeks"]
    
    
  
    for variable in variables:
        for brand in df.brand.unique():
            df_weeks.loc[df_weeks.brand == brand, variable] = (df_weeks[df_weeks.brand == brand][variable].rolling(window=window).mean())


    final_week = df_weeks
    final_week["Category"] = original_category
    final_week['Total Equity'] = final_week[['Awareness', 'Saliency', 'Affinity']].mean(axis=1)

    #calculate the montly
    monthly_output = MetricsClass.calculate_monthly_metrics(final_week, method)
    monthly_output["Category"] = original_category
    monthly_output['Total Equity'] = monthly_output[['Awareness', 'Saliency', 'Affinity']].mean(axis=1)
    monthly_output[columns_to_multiply] = monthly_output[columns_to_multiply].apply(lambda x: x*100)
    monthly_output = monthly_output.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)


    #calculate the quarterly
    quarterly_output = MetricsClass.calculate_quarterly_metrics(final_week, method)
    quarterly_output["Category"] = original_category
    quarterly_output['Total Equity'] = quarterly_output[['Awareness', 'Saliency', 'Affinity']].mean(axis=1)
    quarterly_output[columns_to_multiply] = quarterly_output[columns_to_multiply].apply(lambda x: x*100)
    quarterly_output = quarterly_output.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)



    #calculate the semiannual
    semiannual_output = MetricsClass.calculate_halfyearly_metrics(final_week, method)
    semiannual_output["Category"] = original_category
    semiannual_output['Total Equity'] = semiannual_output[['Awareness', 'Saliency', 'Affinity']].mean(axis=1)
    semiannual_output[columns_to_multiply] = semiannual_output[columns_to_multiply].apply(lambda x: x*100)
    semiannual_output = semiannual_output.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)


    #calculate the yearly
    yearly_output = MetricsClass.calculate_yearly_metrics(final_week, method)
    yearly_output["Category"] = original_category
    yearly_output['Total Equity'] = yearly_output[['Awareness', 'Saliency', 'Affinity']].mean(axis=1)
    yearly_output[columns_to_multiply] = yearly_output[columns_to_multiply].apply(lambda x: x*100)
    yearly_output = yearly_output.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)



    #getting the final smoothened data
    final_df_smoothened = pd.concat([final_week,monthly_output,quarterly_output,semiannual_output,yearly_output],axis=0)

    #-------------------------------------------------------------// --------------------------------------------------------------------
    #doing some transformations
    final_df_smoothened.rename(columns=affinity_to_user,inplace=True)

    final_df_smoothened.rename(columns=framework_to_user,inplace=True)
       
    final_df_smoothened["Category"] = final_df_smoothened["Category"].replace(categories_changed)

    final_df_smoothened["brand"]= final_df_smoothened["brand"].replace(brand_mapping)

    replacements = {"weeks":"Weeks","months":"Months","quarters":"Quarters","semiannual":"Semiannual","years":"Years"}
    final_df_smoothened["time_period"] = final_df_smoothened["time_period"].replace(replacements)


    final_df_smoothened["Category"] = final_df_smoothened["Category"].replace(categories_changed)
    final_df_smoothened["Category"] = final_df_smoothened["Category"].replace("Baby milk","Baby Milk")
  #-------------------------------------------------------------//----------------------------------------------------------------------
    return final_df_smoothened





## ----------------------------------------------------------------------   Equity Analysis -----------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import ttest_ind



def check_affinity_high_level(eq, df, brand,time_period, period_pre, period_start, period_end,aw_metrics):
    st.header("High Level Affinity")
    
    def highlight_row_before(x):
        if x.name == 0:  # Highlight the row with index 2
            return ['background-color: blue'] * len(x)
        else:
            return [''] * len(x)

    def highlight_row_during(x):
        if x.name == 0:  # Highlight the row with index 2
            return ['background-color: blue'] * len(x)
        else:
            return ['background-color: darkred'] * len(x)

    def change_format(x):
        x = f"{x:.1e}"
        return x 


    #creating brands
    brands = [x for x in eq.brand.unique()]
    
    eq = eq.rename(columns=user_to_equity)
    

    # Check if the metric is in awareness metrics
    aw_metrics = aw_metrics

    
    # Filter data accordingly with the data time periods
    df_var = eq[(eq.time >= period_start) & (eq.time <= period_end)]
    df_pre_var = eq[(eq.time >= period_pre) & (eq.time <= period_start)]
    df_all = eq[(eq.time >= period_pre) & (eq.time <= period_end)]
 
    # Initialize Plotly figure
    fig = go.Figure()

    # Plot each brand's metric over time
    max_values = []
    df_all_time = df_all[df_all.time_period == time_period]
    #df_brand = df_all_time[df_all_time['brand'] == brand]
    
    # sorted by time
    df_all_time = df_all_time.sort_values(by="time")
    df_all_time = df_all_time[df_all_time.brand ==brand]
    
    
    df_brand = df_all_time
    for inv_metric in aw_metrics:
        max_values.append(df_brand[inv_metric].max())
        fig.add_trace(go.Scatter(x=df_brand["time"], y=df_brand[inv_metric], mode='lines', name=inv_metric))

    # Add vertical lines for spike start and end
    #fig.add_shape(type="line", x0=period_start, y0=0, x1=period_start, y1=max(max_values) + 5, line=dict(color="red", width=2, dash="dash"), name='Spike Start')
    #fig.add_shape(type="line", x0=period_end, y0=0, x1=period_end, y1=max(max_values) + 5, line=dict(color="green", width=2, dash="dash"), name='Spike End')

    # shaded version
    # Adding a shaded region (shadow) to represent the spike period
    fig.add_shape(type="rect",
              x0=period_start, x1=period_end,
              y0=0, y1=max(max_values) + 5,
              fillcolor="rgba(255, 0, 0, 0.2)",  # Red color with transparency
              line=dict(color="rgba(255, 0, 0, 0.0)"))  # No border
    
    # Update layout
    fig.update_layout(
        title=f'{brand} - Affinity over time -',
        xaxis_title='Time Period',
        yaxis_title='Scores',
        hovermode='x unified'
    )

    # Show plot
    st.plotly_chart(fig)
    st.subheader("Wich high level affinity pillar changed the most")

    #doing the t-test for each brand
    rows_before=[]
    rows_during = []
    t_stats = {}
    for inv_metric in aw_metrics:
        data_brand_pre_var = df_pre_var[df_pre_var['brand'] == brand]
        data_brand_var = df_var[df_var['brand'] == brand]

        mean_before =  str(round(data_brand_pre_var[inv_metric].mean(),2))
        mean_after = str(round(data_brand_var[inv_metric].mean(),2))
        
        t_statistic, p_value = ttest_ind(data_brand_pre_var[inv_metric], data_brand_var[inv_metric])
        t_stats[inv_metric] = (t_statistic, p_value)
        #st.write(f"{inv_metric}: t-statistic = {t_statistic}, p-value = {p_value}")
        row_before = {"time period":"Before","metric":inv_metric,"mean":mean_before,"t-statistic":t_statistic,"p-value":p_value}
        row_during = {"time period":"During","metric":inv_metric,"mean":mean_after,"t-statistic":t_statistic,"p-value":p_value}


        rows_before.append(row_before)
        rows_during.append(row_during)

        

    column_1,column_2 = st.columns(2)


    #Before column

    new_data_before = pd.DataFrame(rows_before)
    new_data_during = pd.DataFrame(rows_during)
    
    with column_1:
        new_data_before = new_data_before.sort_values(by="p-value",ascending=True).reset_index(drop=True)

        new_data_before["p-value"] = new_data_before["p-value"].apply(change_format)
        new_data_before["t-statistic"] = new_data_before["t-statistic"].apply(change_format)

        new_data_before = new_data_before.reset_index(drop=True)
        
        new_data_before = new_data_before.style.apply(highlight_row_before, axis=1)

        st.dataframe(new_data_before,hide_index=True)
    
    with column_2:
        new_data_during = new_data_during.sort_values(by="p-value",ascending=True).reset_index(drop=True)

        new_data_during["p-value"] = new_data_during["p-value"].apply(change_format)
        new_data_during["t-statistic"] = new_data_during["t-statistic"].apply(change_format)

        new_data_during = new_data_during.reset_index(drop=True)

        new_data_during = new_data_during.style.apply(highlight_row_during, axis=1)

        st.dataframe(new_data_during,hide_index=True)

    
    most_changed_metric = max(t_stats, key=lambda k: abs(t_stats[k][0]))
    most_changed_t_stat, most_changed_p_value = t_stats[most_changed_metric]

    

    st.write(f"\nThe metric that has changed the most is **'{most_changed_metric}'** with a t-statistic of **{most_changed_t_stat:.1e}** and a p-value of **{most_changed_p_value:.1e}**.")

  
   
    most_changed_metric = st.selectbox("Metric to study",[mt for mt in aw_metrics])
    
    if most_changed_metric == "AF_Entry_point":
        most_changed_metric = "AF_Life_Entry"
    
    if most_changed_metric == "AF_Brand_Love":
        most_changed_metric = "AF_Brand_Prest_Lov"
    
    if most_changed_metric == "AF_Value_for_Money":
        most_changed_metric = "AF_VFM"

    return most_changed_metric


#Affinity low level


def check_affinity_low_level(df, eq, period_pre,period_start,period_end,metric, brand, channel,pre_year,post_year,brand_mapping):
    st.subheader("Low level affinity")
    

    
    def highlight_row_before(x):
        if x.name == 0:  # Highlight the row with index 2
            return ['background-color: blue'] * len(x)
        else:
            return [''] * len(x)

    def highlight_row_during(x):
        if x.name == 0:  # Highlight the row with index 2
            return ['background-color: blue'] * len(x)
        else:
            return ['background-color: darkred'] * len(x)


    def change_format(x):
        x = f"{x:.1e}"
        return x 




    # Creating 'Week Commencing' column
    df['Week Commencing'] = df['created_time'].apply(lambda x: (x - timedelta(days=x.weekday())).replace(hour=0, minute=0, second=0, microsecond=0))

  

    aff_metrics = aff_metrics_analysis

    aff_metric_pillars = aff_metrics_pillars_analysis
    
    inv_metric = metric
   
    df["brand"] = df["brand"].replace(brand_mapping)

    inv_brand = brand
    if channel != None:
        channel_filter = [channel]
    else:
        channel_filter =  ["None"]

    period_pre = period_pre
    period_start = period_start
    period_end = period_end

    if inv_metric in aff_metric_pillars:
        inv_sub_metrics = [mt for mt in aff_metrics if inv_metric in mt]
    elif inv_metric == 'framework_affinity':
        inv_sub_metrics = aff_metrics

    df_var = df[(df.brand == inv_brand) & (df.created_time >= period_start) & (df.created_time <= period_end)
                & (~df.message_type.isin(channel_filter))]
    df_pre_var = df[(df.brand == inv_brand) & (df.created_time >= period_pre) & (df.created_time <= period_start)
                    & (~df.message_type.isin(channel_filter))]
    df_all = df[(df.brand == inv_brand) & (df.created_time >= period_pre) & (df.created_time <= period_end)
                & (~df.message_type.isin(channel_filter))]


    
    # Plotting sub-metrics over time
    subm_df = pd.DataFrame(df_all.groupby(['Week Commencing'])[inv_sub_metrics].sum()).reset_index()

    fig = go.Figure()
    for mt in inv_sub_metrics:
        fig.add_trace(go.Scatter(x=subm_df['Week Commencing'], y=subm_df[mt], mode='lines', name=mt))
    
    
    # Add a shaded area between period_start and period_end
    fig.add_shape(type="rect",
              x0=period_start, x1=period_end,
              y0=0, y1=1,  # The y-axis range should cover the whole plot; 'yref' will be 'paper' to cover the full height
              xref="x", yref="paper",  # 'xref' set to "x" means the x-coordinates are in data space, 'yref' set to "paper" means y is relative to the plot (0 to 1)
              fillcolor="rgba(255, 0, 0, 0.2)",  # Red color with transparency for the shaded area
              line=dict(color="rgba(255, 0, 0, 0.0)"))  # No border line

    # Update layout and show the plot
    fig.update_layout(title='Sub-metrics Over Time (Positive sentiment)', 
                  xaxis_title='Week Commencing', 
                  yaxis_title='Scores', 
                  legend_title='Metrics')

       
    st.plotly_chart(fig)
    
    

    # Sentiment normalised score
    plus_benefits_scores = df_all[df_all['sentiment'].isin(['Positive', "Neutral"])].groupby(['Week Commencing'])[inv_sub_metrics].apply(lambda x: x.astype(int).sum())
    for column in plus_benefits_scores:
        new_name = "Plus_" + str(column)
        plus_benefits_scores = plus_benefits_scores.rename(columns={column: new_name})
    minus_benefits_scores = df_all[df_all['sentiment'].isin(['Negative'])].groupby(['Week Commencing'])[inv_sub_metrics].apply(lambda x: x.astype(int).sum())
    for column in minus_benefits_scores:
        new_name = "Minus_" + str(column)
        minus_benefits_scores = minus_benefits_scores.rename(columns={column: new_name})
    benefits_scores = pd.concat([plus_benefits_scores, minus_benefits_scores], axis=1).fillna(0)
    for benefit in inv_sub_metrics:
        benefits_scores["Net_" + str(benefit)] = benefits_scores["Plus_" + str(benefit)] - benefits_scores["Minus_" + str(benefit)]

    benefits_scores = benefits_scores[[x for x in benefits_scores if "Net_" in x]]

    fig = go.Figure()
    for mt in benefits_scores.columns:
        fig.add_trace(go.Scatter(x=benefits_scores.index, y=benefits_scores[mt], mode='lines', name=mt))
    #fig.add_vline(x=datetime.strptime(period_start, '%Y-%m-%d'), line=dict(color='red', dash='dash'), name='Spike Start')
    #fig.add_vline(x=datetime.strptime(period_end, '%Y-%m-%d'), line=dict(color='green', dash='dash'), name='Spike End')
    #fig.update_layout(title='Sentiment Normalised Sub-metrics Over Time', xaxis_title='Week Commencing', yaxis_title='Scores', legend_title='Metrics')
    #fig.show()
    
     # Add a shaded area between period_start and period_end
    fig.add_shape(type="rect",
              x0=period_start, x1=period_end,
              y0=0, y1=1,  # The y-axis range should cover the whole plot; 'yref' will be 'paper' to cover the full height
              xref="x", yref="paper",  # 'xref' set to "x" means the x-coordinates are in data space, 'yref' set to "paper" means y is relative to the plot (0 to 1)
              fillcolor="rgba(255, 0, 0, 0.2)",  # Red color with transparency for the shaded area
              line=dict(color="rgba(255, 0, 0, 0.0)"))  # No border line

    # Update layout and show the plot
    fig.update_layout(title='Sentiment Normalised Sub-metrics Over Time ( (Positive + Neutral) - negative', 
                  xaxis_title='Time Commencing', 
                  yaxis_title='Scores', 
                  legend_title='Metrics')
    st.plotly_chart(fig)



    # Identify the sub-metric that has changed the most
    st.subheader("Wich sub-metric changed the most")
    rows_before=[]
    rows_during = []
    t_stats = {}

    for metric in inv_sub_metrics:
        data_brand_pre_var = df_pre_var[df_pre_var['brand'] == brand]
        data_brand_var = df_var[df_var['brand'] == brand]

        mean_before =  str(round(data_brand_pre_var[metric].mean(),2))
        mean_after = str(round(data_brand_var[metric].mean(),2))
        
        t_statistic, p_value = ttest_ind(data_brand_pre_var[metric], data_brand_var[metric])
        t_stats[inv_metric] = (t_statistic, p_value)
        #st.write(f"{inv_metric}: t-statistic = {t_statistic}, p-value = {p_value}")
        row_before = {"time period":"Before","metric":metric,"mean":mean_before,"t-statistic":t_statistic,"p-value":p_value}
        row_during = {"time period":"During","metric":metric,"mean":mean_after,"t-statistic":t_statistic,"p-value":p_value}


        rows_before.append(row_before)
        rows_during.append(row_during)

    column_1,column_2 = st.columns(2)
    
    #Before column
    new_data_before = pd.DataFrame(rows_before)
    new_data_during = pd.DataFrame(rows_during)
    
    with column_1:
        new_data_before = new_data_before.sort_values(by="p-value",ascending=True).reset_index(drop=True)

        new_data_before["p-value"] = new_data_before["p-value"].apply(change_format)
        new_data_before["t-statistic"] = new_data_before["t-statistic"].apply(change_format)

        new_data_before = new_data_before.reset_index(drop=True)
        most_changed_sub_metric = new_data_before["metric"].iloc[0]
        new_data_before = new_data_before.style.apply(highlight_row_before, axis=1)

        st.dataframe(new_data_before,hide_index=True)
    
    with column_2:
        new_data_during = new_data_during.sort_values(by="p-value",ascending=True).reset_index(drop=True)

        new_data_during["p-value"] = new_data_during["p-value"].apply(change_format)
        new_data_during["t-statistic"] = new_data_during["t-statistic"].apply(change_format)

        new_data_during = new_data_during.reset_index(drop=True)

        new_data_during = new_data_during.style.apply(highlight_row_during, axis=1)

        st.dataframe(new_data_during,hide_index=True)

    
    st.write(f"\nThe sub-metric that has changed the most is **'{most_changed_sub_metric}'** ")

    most_changed_sub_metric = st.selectbox("Metric to study",[mt for mt in inv_sub_metrics])



    # Analyzing which channel is causing the spike

    st.subheader("Wich channel has seen the highest change")

    rows_before=[]
    rows_during = []

    channels = df['message_type'].unique()
    for channel in channels:
        channel_pre_var_data = df_pre_var[df_pre_var['message_type'] == channel]
        channel_var_data = df_var[df_var['message_type'] == channel]

        #for metric in inv_sub_metrics:
        metric = most_changed_sub_metric
        data_brand_pre_var = channel_pre_var_data[channel_pre_var_data['brand'] == brand]
        data_brand_var = channel_var_data[channel_var_data['brand'] == brand]

        mean_before =  str(round(data_brand_pre_var[metric].mean(),2))
        mean_after = str(round(data_brand_var[metric].mean(),2))
        
        len_before = len(channel_pre_var_data)
        len_during = len(channel_var_data)

        t_statistic, p_value = ttest_ind(data_brand_pre_var[metric], data_brand_var[metric])
        t_stats[metric] = (t_statistic, p_value)
        #st.write(f"{inv_metric}: t-statistic = {t_statistic}, p-value = {p_value}")
        row_before = {"time period":"before","Metric":metric,"channel":channel,"mean":mean_before,"number of mentions":len_before,"t-statistic":t_statistic,"p-value":p_value}
        row_during =  {"time period":"during","Metric":metric,"channel":channel,"mean":mean_after,"number of mentions":len_during,"t-statistic":t_statistic,"p-value":p_value}


        rows_before.append(row_before)
        rows_during.append(row_during)

    column_1,column_2 = st.columns(2)


    with column_1:
            df_mean_before_channel = pd.DataFrame(rows_before)
            df_mean_before_channel = df_mean_before_channel.sort_values(by="p-value",ascending=True).reset_index(drop=True)
            df_mean_before_channel["t-statistic"] = df_mean_before_channel["t-statistic"].apply(change_format)
            df_mean_before_channel["p-value"] = df_mean_before_channel["p-value"].apply(change_format)
            most_changed_channel = df_mean_before_channel.channel.iloc[0]
            #Aggregate
            df_mean_before_channel = df_mean_before_channel.style.apply(highlight_row_before,axis=1)
            st.dataframe(df_mean_before_channel,hide_index=True)

    with column_2:
        df_mean_during_channel = pd.DataFrame(rows_during)
        df_mean_during_channel = df_mean_during_channel.sort_values(by="p-value",ascending=True).reset_index(drop=True)
        df_mean_during_channel["t-statistic"] = df_mean_during_channel["t-statistic"].apply(change_format)
        df_mean_during_channel["p-value"] = df_mean_during_channel["p-value"].apply(change_format)
        #Aggregate
        df_mean_during_channel = df_mean_during_channel.style.apply(highlight_row_during,axis=1)
        st.dataframe(df_mean_during_channel,hide_index=True)

    
    st.write(f"\nThe channel that has changed the most is **'{most_changed_channel}'**")


    st.dataframe(df_all[df_all.message_type == most_changed_channel],hide_index=True)

    
    # getting the mean of the metric in this case the followers, and comparing before and during the time. 
    
    st.header("Sentiment Analysis")

    st.subheader(f"How has sentiment impacted the affinity metric")
     

    x_metric = most_changed_sub_metric
    try:
        x_metric = st.selectbox("select sub-metric",[mt for mt in inv_sub_metrics])
        st.write(f"Analysing for **{x_metric}**")

        #Positive + Neutral
        rows_before_pos= []
        rows_during_pos = []

        rows_before_neu= []
        rows_during_neu = []

        rows_before_neg= []
        rows_during_neg= []
        
        #Positive
        df_counting_pre_positive = df_pre_var[(df_pre_var[x_metric] == 1)  &  (df_pre_var.sentiment == "Positive") ]
        df_counting_during_positive = df_var[(df_var[x_metric] == 1)  & (df_var.sentiment == "Positive")]
        
        len_positive_pre = len(df_counting_pre_positive)
        len_positive_pos= len(df_counting_during_positive)

        mean_positive_pre = round(len_positive_pre / len(df_pre_var),2)
        mean_positive_during = round(len_positive_pos / len(df_var),2)

        t_statistic, p_value = ttest_ind(mean_positive_pre, mean_positive_during)


        row_before = {"time period":"before","sub metric":x_metric,"sentiment":"positive","number of mentions":len_positive_pre}
        row_during = {"time period":"during","sub metric":x_metric,"sentiment":"positive","number of mentions":len_positive_pos}


        rows_before_pos.append(row_before)
        rows_during_pos.append(row_during)


        #Neutral 
        df_counting_pre_neutral = df_pre_var[(df_pre_var[x_metric] == 1)  &  (df_pre_var.sentiment == "Neutral") ]
        df_counting_during_neutral = df_var[(df_var[x_metric] == 1)  & (df_var.sentiment == "Neutral")]
        
        len_neutral_pre = len(df_counting_pre_neutral)
        len_neutral_pos= len(df_counting_during_neutral)

        mean_neutral_pre = round(len_neutral_pre / len(df_pre_var),2)
        mean_neutral_during = round(len_neutral_pos / len(df_var),2)

        t_statistic, p_value = ttest_ind(mean_neutral_pre, mean_neutral_during)


        row_before = {"time period":"before","sub metric":x_metric,"sentiment":"neutral","number of mentions":len_neutral_pre}
        row_during = {"time period":"during","sub metric":x_metric,"sentiment":"neutral","number of mentions":len_neutral_pos}

        rows_before_neu.append(row_before)
        rows_during_neu.append(row_during)

        #Negative
        df_counting_pre_negative = df_pre_var[(df_pre_var[x_metric] == 1)  &  (df_pre_var.sentiment == "Negative") ]
        df_counting_during_negative = df_var[(df_var[x_metric] == 1)  & (df_var.sentiment == "Negative")]
        

        len_negative_pre = len(df_counting_pre_negative)
        len_negative_pos= len(df_counting_during_negative)


        mean_negative_pre = round(len_negative_pre / len(df_pre_var),2)
        mean_negative_during = round(len_negative_pos / len(df_var),2)

        t_statistic, p_value = ttest_ind(mean_negative_pre, mean_negative_during)

        row_before = {"time period":"before","sub metric":x_metric,"sentiment":"negative","number of mentions":len_negative_pre}
        row_during = {"time period":"during","sub metric":x_metric,"sentiment":"negative","number of mentions":len_negative_pos}

        rows_before_neg.append(row_before)
        rows_during_neg.append(row_during)
        

    except:
        st.warning("Not able to calculate the sentiment analysis, maybe due to several reasons - bad choice of the periods, not enough data... - ")
    

    df_all = pd.DataFrame(columns=["time period","sub metric","sentiment","number of mentions"])




    column_1,column_2 = st.columns(2)


    #Before column

    new_data_before_positive = pd.DataFrame(rows_before_pos)
    new_data_during_positive = pd.DataFrame(rows_during_pos)
    
    new_data_before_neutral = pd.DataFrame(rows_before_neu)
    new_data_during_neutral = pd.DataFrame(rows_during_neu)
    
    new_data_before_negative = pd.DataFrame(rows_before_neg)
    new_data_during_negative = pd.DataFrame(rows_during_neg)


    df_before_all = pd.concat([df_all,new_data_before_positive,new_data_before_neutral,new_data_before_negative],axis=0)
    df_during_all = pd.concat([df_all,new_data_during_positive,new_data_during_neutral,new_data_during_negative],axis=0)


    with column_1:
        st.dataframe(df_before_all,hide_index=True)
    with column_2:
        st.dataframe(df_during_all,hide_index=True)



# Awareness
def check_awareness_high_level(eq, df, brand,time_period, period_pre, period_start, period_end):
    st.header("Awareness")

    st.subheader("Wich sub-metric changed the most in that period of time ?")


    def highlight_row_before(x):
        if x.name == 0:  # Highlight the row with index 2
            return ['background-color: blue'] * len(x)
        else:
            return [''] * len(x)

    def highlight_row_during(x):
        if x.name == 0:  # Highlight the row with index 2
            return ['background-color: blue'] * len(x)
        else:
            return ['background-color: darkred'] * len(x)



    def change_format(x):
        x = f"{x:.1e}"
        return x 

    #creating brands
    brands = [x for x in eq.brand.unique()]
    
    
    # Check if the metric is in awareness metrics
    aw_metrics = ['AA_eSoV', 'AA_Reach', 'AA_Brand_Breadth']

    aw_renaming = {"eSoV" : "AA_eSoV", "Reach":"AA_Reach","Brand Breadth":"AA_Brand_Breadth"}

    eq.rename(columns=aw_renaming,inplace=True)

    # Filter data according to time periods
    df_var = eq[(eq.time >= period_start) & (eq.time <= period_end)]
    df_pre_var = eq[(eq.time >= period_pre) & (eq.time <= period_start)]
    df_all = eq[(eq.time >= period_pre) & (eq.time <= period_end)]

    # Initialize Plotly figure
    fig = go.Figure()

    # Plot each brand's metric over time
    max_values = []
    df_all_time = df_all[df_all.time_period == time_period]
    df_brand = df_all_time[df_all_time['brand'] == brand]
    
    
    for inv_metric in aw_metrics:
        max_values.append(df_brand[inv_metric].max())
        fig.add_trace(go.Scatter(x=df_brand["time"], y=df_brand[inv_metric], mode='lines', name=inv_metric))


    fig.add_shape(
    type="rect",
    x0=period_start,
    y0=0,
    x1=period_end,
    y1=max(max_values) + 5,
    fillcolor="red",  # You can choose any color you like
    opacity=0.3,  # Adjust the opacity as needed
    line=dict(width=0),  # No border line for the shaded area
    layer="below"  # Place the shaded area below other shapes/lines
    )



    # Update layout
    fig.update_layout(
        title=f'{brand} - Awareness over time -',
        xaxis_title='Time Period',
        yaxis_title='Scores',
        hovermode='x unified'
    )


    # Show plot
    st.plotly_chart(fig)

    
    #doing the t-test for each metric
    rows_before=[]
    rows_during = []
    t_stats = {}
    for inv_metric in aw_metrics:
        data_brand_pre_var = df_pre_var[df_pre_var['brand'] == brand]
        data_brand_var = df_var[df_var['brand'] == brand]

        mean_before =  str(round(data_brand_pre_var[inv_metric].mean(),2))
        mean_after = str(round(data_brand_var[inv_metric].mean(),2))
        
        t_statistic, p_value = ttest_ind(data_brand_pre_var[inv_metric], data_brand_var[inv_metric])
        t_stats[inv_metric] = (t_statistic, p_value)
        #st.write(f"{inv_metric}: t-statistic = {t_statistic}, p-value = {p_value}")
        row_before = {"time period":"Before","metric":inv_metric,"mean":mean_before,"t-statistic":t_statistic,"p-value":p_value}
        row_during = {"time period":"During","metric":inv_metric,"mean":mean_after,"t-statistic":t_statistic,"p-value":p_value}


        rows_before.append(row_before)
        rows_during.append(row_during)



    column_1,column_2 = st.columns(2)


    #Before column

    new_data_before = pd.DataFrame(rows_before)
    new_data_during = pd.DataFrame(rows_during)
    
    with column_1:
        new_data_before = new_data_before.sort_values(by="p-value",ascending=True).reset_index(drop=True)

        new_data_before["p-value"] = new_data_before["p-value"].apply(change_format)
        new_data_before["t-statistic"] = new_data_before["t-statistic"].apply(change_format)

        new_data_before = new_data_before.reset_index(drop=True)
        
        new_data_before = new_data_before.style.apply(highlight_row_before, axis=1)

        st.dataframe(new_data_before,hide_index=True)
    
    with column_2:
        new_data_during = new_data_during.sort_values(by="p-value",ascending=True).reset_index(drop=True)

        new_data_during["p-value"] = new_data_during["p-value"].apply(change_format)
        new_data_during["t-statistic"] = new_data_during["t-statistic"].apply(change_format)

        new_data_during = new_data_during.reset_index(drop=True)

        new_data_during = new_data_during.style.apply(highlight_row_during, axis=1)

        st.dataframe(new_data_during,hide_index=True)

    most_changed_metric = max(t_stats, key=lambda k: abs(t_stats[k][0]))
    most_changed_t_stat, most_changed_p_value = t_stats[most_changed_metric]

    

    st.write(f"\nThe metric that has changed the most is **'{most_changed_metric}'** with a t-statistic of **{most_changed_t_stat:.1e}** and a p-value of **{most_changed_p_value:.1e}**.")

  
    # _____________________________________________________ Not being used ________________________________________________

    # Analyzing which channel is causing the spike

    sub_pillars_options = {'AA_eSoV':['mentions'], 'AA_Reach':['followers']}
    
    if most_changed_metric == "AA_Brand_Breadth":
        pass
    else:
        sub_pillars_awareness = sub_pillars_options[most_changed_metric]
               

        df = df
        # Filter data according to time periods
        df_var = df[(df.created_time >= period_start) & (df.created_time <= period_end)]
        df_pre_var = df[(df.created_time >= period_pre) & (df.created_time <= period_start)]
        df_all = df[(df.created_time >= period_pre) & (df.created_time <= period_end)]
                
        df_awareness_channels = pd.DataFrame(columns=["Sub-pillars","t_statistic","p-value","channel"])
        df_awareness_channels_show = pd.DataFrame(columns=["Sub-pillars","t_statistic","p-value","channel"])

        rows=[]
        rows_show = []
        df_channel_stats = pd.DataFrame(columns=["metric","t_statistic","p-value","channel"])
        channels = df['message_type'].unique()
        for channel in channels:
            channel_pre_var_data = df_pre_var[df_pre_var['message_type'] == channel]
            channel_var_data = df_var[df_var['message_type'] == channel]


            channel_t_stats = {}
            for metric in sub_pillars_awareness:
                if not channel_pre_var_data[metric].empty and not channel_var_data[metric].empty:
                    if metric == "mentions":
                        
                        channel_pre_var_data =   (channel_pre_var_data.groupby(['Week Commencing', 'brand'])['mentions'].sum()/channel_pre_var_data.groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'eSoV'})
                        channel_var_data = (channel_var_data.groupby(['Week Commencing', 'brand'])['mentions'].sum()/channel_var_data.groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'eSoV'}) 
                        
                        t_statistic, p_value = ttest_ind(channel_pre_var_data.eSoV, channel_var_data.eSoV)
                    else:

                         t_statistic, p_value = ttest_ind(channel_pre_var_data[metric], channel_var_data[metric])

                    if str(t_statistic) == "nan" or str(t_statistic) == "inf":
                        pass
                    else:
                        channel_t_stats[metric] = (t_statistic, p_value)
                        row = {"Sub-pillars":metric,"t_statistic":t_statistic,"p-value":p_value,"channel":channel}
                        row_show = {"Sub-pillars":metric,"t_statistic":t_statistic,"p-value":p_value,"channel":channel} 
                        rows.append(row)
                        rows_show.append(row_show)
                        
        
        new_data = pd.DataFrame(rows)
        new_data_show = pd.DataFrame(rows_show)
        df_awareness_channels = pd.concat([df_awareness_channels,new_data],ignore_index=True)
        df_awareness_channels_show = pd.concat([df_awareness_channels_show,new_data_show],ignore_index=True)

    
        df_awareness_channels_show = df_awareness_channels_show.sort_values(by="t_statistic",ascending=False).reset_index(drop=True)

        df_awareness_channels_show["t_statistic"] = df_awareness_channels_show["t_statistic"].apply(change_format)
        df_awareness_channels_show["p-value"] = df_awareness_channels_show["p-value"].apply(change_format)

        # st.write(df_awareness_channels_show)
        # _____________________________________________________ Not being used ________________________________________________


        st.subheader("Wich channel has seen the highest change?")


        # getting the absolute values
        df_awareness_channels["t_statistic"] = abs(df_awareness_channels["t_statistic"])
  
        st.write(f"Mean (in each channel) -Before vs During - {sub_pillars_awareness[0]}")


        if most_changed_metric == "AA_eSoV":
            column_name = "mentions"
        else:
            column_name = "followers"
        

        rows_before = []
        rows_during = []
        for channel in channels:
            df_pre_var_channel = df_pre_var[df_pre_var["message_type"] == channel ]
            df_var_channel = df_var[df_var["message_type"] == channel ]
            
            if sub_pillars_awareness[0] == "mentions":

                df_pre_var_channel = (df_pre_var_channel.groupby(['Week Commencing', 'brand'])['mentions'].sum()/df_pre_var_channel.groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'eSoV'})
                df_var_channel = (df_var_channel.groupby(['Week Commencing', 'brand'])['mentions'].sum()/df_var_channel.groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'eSoV'}) 
        

                len_before = len(df_pre_var_channel)
                len_during = len(df_var_channel)

                mean_before = str(round(df_pre_var_channel.eSoV.mean(),2))
                mean_after = str(round(df_var_channel.eSoV.mean(),2))

                t_statistic, p_value = ttest_ind(df_pre_var_channel.eSoV, df_var_channel.eSoV)
            
            else:

                len_before = len(df_pre_var_channel)
                len_during = len(df_var_channel)

                mean_before = str(round(df_pre_var_channel[sub_pillars_awareness[0]].mean(),2))
                mean_after = str(round(df_var_channel[sub_pillars_awareness[0]].mean(),2))

                t_statistic, p_value = ttest_ind(df_pre_var_channel[metric], df_var_channel[metric])
            


            row_before = {"time period":"before","channel":channel,"mean":mean_before,"number of mentions":len_before,column_name:round(len(df_pre_var_channel),1),"t-statistic":t_statistic,"p-value":p_value}
            row_during =  {"time period":"during","channel":channel,"mean":mean_after,"number of mentions":len_during,column_name:round(len(df_var_channel),1),"t-statistic":t_statistic,"p-value":p_value}

            rows_before.append(row_before)
            rows_during.append(row_during)


        column_1,column_2 = st.columns(2)


        with column_1:
            df_mean_before_channel = pd.DataFrame(rows_before)
            df_mean_before_channel = df_mean_before_channel.sort_values(by="p-value",ascending=True).reset_index(drop=True)
            df_mean_before_channel["t-statistic"] = df_mean_before_channel["t-statistic"].apply(change_format)
            df_mean_before_channel["p-value"] = df_mean_before_channel["p-value"].apply(change_format)
            most_changed_channel = df_mean_before_channel.channel.iloc[0]
            #Aggregate
            df_mean_before_channel = df_mean_before_channel.style.apply(highlight_row_before,axis=1)
            st.dataframe(df_mean_before_channel,hide_index=True)

        with column_2:
            df_mean_during_channel = pd.DataFrame(rows_during)
            df_mean_during_channel = df_mean_during_channel.sort_values(by="p-value",ascending=True).reset_index(drop=True)
            df_mean_during_channel["t-statistic"] = df_mean_during_channel["t-statistic"].apply(change_format)
            df_mean_during_channel["p-value"] = df_mean_during_channel["p-value"].apply(change_format)
            #Aggregate
            df_mean_during_channel = df_mean_during_channel.style.apply(highlight_row_during,axis=1)
            st.dataframe(df_mean_during_channel,hide_index=True)

        
        st.write(f"\nThe channel that has changed the most is **'{most_changed_channel}'**")
        

        #filtered table. 
        df_var_show = df_var[df_var.message_type == most_changed_channel]

        st.dataframe(df_var_show)


#Saliency 
def check_saliency_high_level(eq, df, brand,time_period, period_pre, period_start, period_end):
    st.subheader("Saliency")

    st.subheader("Wich sub-metric changed the most in that period of time ?")

    def highlight_row_before(x):
        if x.name == 0:  # Highlight the row with index 2
            return ['background-color: blue'] * len(x)
        else:
            return [''] * len(x)

    def highlight_row_during(x):
        if x.name == 0:  # Highlight the row with index 2
            return ['background-color: blue'] * len(x)
        else:
            return ['background-color: darkred'] * len(x)



    def change_format(x):
        x = f"{x:.1e}"
        return x 

   
    
    
    # Check if the metric is in awareness metrics
    as_metrics =  ['AS_Average_Engagement', 'AS_Usage_SoV','AS_Search_Index','AS_Brand_Centrality']

    as_renaming = {"Average Engagement" :"AS_Average_Engagement","Usage SoV":"AS_Usage_SoV","Trial":"AS_Trial_Sov","Search Index":"AS_Search_Index","Brand Centrality":"AS_Brand_Centrality"}

    eq.rename(columns=as_renaming,inplace=True)


    # Filter data according to time periods
    df_var = eq[(eq.time >= period_start) & (eq.time <= period_end)]
    df_pre_var = eq[(eq.time >= period_pre) & (eq.time <= period_start)]
    df_all = eq[(eq.time >= period_pre) & (eq.time <= period_end)]


    # Initialize Plotly figure
    fig = go.Figure()

    # Plot each brand's metric over time
    max_values = []
    df_all_time = df_all[df_all.time_period == time_period]
    df_brand = df_all_time[df_all_time['brand'] == brand]
        
    for inv_metric in as_metrics:
        max_values.append(df_brand[inv_metric].max())
        fig.add_trace(go.Scatter(x=df_brand["time"], y=df_brand[inv_metric], mode='lines', name=inv_metric))


    fig.add_shape(
    type="rect",
    x0=period_start,
    y0=0,
    x1=period_end,
    y1=max(max_values) + 5,
    fillcolor="red",  # You can choose any color you like
    opacity=0.3,  # Adjust the opacity as needed
    line=dict(width=0),  # No border line for the shaded area
    layer="below"  # Place the shaded area below other shapes/lines
    )



    # Update layout
    fig.update_layout(
        title=f'{brand} - Awareness over time -',
        xaxis_title='Time Period',
        yaxis_title='Scores',
        hovermode='x unified'
    )


    # Show plot
    st.plotly_chart(fig)

    
    #doing the t-test for each metric
    df_saliency = pd.DataFrame(columns=["Pillars","t-statistic","p-value"])
    rows_before=[]
    rows_during = []
    t_stats = {}
    for inv_metric in as_metrics:
        data_brand_pre_var = df_pre_var[df_pre_var['brand'] == brand]
        data_brand_var = df_var[df_var['brand'] == brand]

        mean_before =  str(round(data_brand_pre_var[inv_metric].mean(),2))
        mean_after = str(round(data_brand_var[inv_metric].mean(),2))
        
        t_statistic, p_value = ttest_ind(data_brand_pre_var[inv_metric], data_brand_var[inv_metric])
        t_stats[inv_metric] = (t_statistic, p_value)
        #st.write(f"{inv_metric}: t-statistic = {t_statistic}, p-value = {p_value}")
        row_before = {"time period":"Before","metric":inv_metric,"mean":mean_before,"t-statistic":t_statistic,"p-value":p_value}
        row_during = {"time period":"During","metric":inv_metric,"mean":mean_after,"t-statistic":t_statistic,"p-value":p_value}


        rows_before.append(row_before)
        rows_during.append(row_during)

    column_1,column_2 = st.columns(2)

    #Before column

    new_data_before = pd.DataFrame(rows_before)
    new_data_during = pd.DataFrame(rows_during)
    
    with column_1:
        new_data_before = new_data_before.sort_values(by="p-value",ascending=True).reset_index(drop=True)

        new_data_before["p-value"] = new_data_before["p-value"].apply(change_format)
        new_data_before["t-statistic"] = new_data_before["t-statistic"].apply(change_format)

        new_data_before = new_data_before.reset_index(drop=True)
        
        new_data_before = new_data_before.style.apply(highlight_row_before, axis=1)

        st.dataframe(new_data_before,hide_index=True)
    
    with column_2:
        new_data_during = new_data_during.sort_values(by="p-value",ascending=True).reset_index(drop=True)

        new_data_during["p-value"] = new_data_during["p-value"].apply(change_format)
        new_data_during["t-statistic"] = new_data_during["t-statistic"].apply(change_format)

        new_data_during = new_data_during.reset_index(drop=True)

        new_data_during = new_data_during.style.apply(highlight_row_during, axis=1)

        st.dataframe(new_data_during,hide_index=True)

    most_changed_metric = max(t_stats, key=lambda k: abs(t_stats[k][0]))
    most_changed_t_stat, most_changed_p_value = t_stats[most_changed_metric]

    

    st.write(f"\nThe metric that has changed the most is **'{most_changed_metric}'** with a t-statistic of **{most_changed_t_stat:.1e}** and a p-value of **{most_changed_p_value:.1e}**.")




# _______________________________________________________ Not being used ______________________________________________________
    if most_changed_metric == "AS_Search_Index" or most_changed_metric == "AS_Brand_Centrality":
        pass
    else:
        sub_pillars_options = {'AS_Average_Engagement':['earned_engagements'], 'AS_Usage_SoV':['Usage'],'AS_Trial_Sov':['Trial or Experimentation']}
        sub_pillars_saliency = sub_pillars_options[most_changed_metric]
        
        df = df
        # Filter data according to time periods
        df_var = df[(df.created_time >= period_start) & (df.created_time <= period_end)]
        df_pre_var = df[(df.created_time >= period_pre) & (df.created_time <= period_start)]
        df_all = df[(df.created_time >= period_pre) & (df.created_time <= period_end)]
        

        df_saliency_channels = pd.DataFrame(columns=["Sub-pillars","t_statistic","p-value","channel"])
        channels = df['message_type'].unique()
        
        # rows=[]
        # rows_show = []
        #channels = df['message_type'].unique()
        # for channel in channels:
        #     channel_pre_var_data = df_pre_var[df_pre_var['message_type'] == channel]
        #     channel_var_data = df_var[df_var['message_type'] == channel]

        #     channel_t_stats = {}
        #     for metric in sub_pillars_saliency:
        #         if not channel_pre_var_data[metric].empty and not channel_var_data[metric].empty:
        #             t_statistic, p_value = ttest_ind(channel_pre_var_data[metric], channel_var_data[metric])
        #             if str(t_statistic) == "nan" or str(t_statistic) == "inf":
        #                 pass
        #             else:
        #                 channel_t_stats[metric] = (t_statistic, p_value)
        #                 row = {"Sub-pillars":metric,"t_statistic":t_statistic,"p-value":p_value,"channel":channel}
        #                 row_show = {"Sub-pillars":metric,"t_statistic":t_statistic,"p-value":p_value,"channel":channel} 
        #                 rows.append(row)
        #                 rows_show.append(row_show)
                        
        
        # new_data = pd.DataFrame(rows)
        # new_data_show = pd.DataFrame(rows_show)
        # df_saliency_channels = pd.concat([df_saliency_channels,new_data],ignore_index=True)
        # df_saliency_channels_show = pd.concat([df_saliency_channels_show,new_data_show],ignore_index=True)

    
        # df_saliency_channels_show = df_saliency_channels_show.sort_values(by="t_statistic",ascending=False).reset_index(drop=True)

        # df_saliency_channels_show["t_statistic"] = df_saliency_channels_show["t_statistic"].apply(change_format)
        # df_saliency_channels_show["p-value"] = df_saliency_channels_show["p-value"].apply(change_format)

        #df_saliency_channels_show = df_saliency_channels_show.style.apply(highlight_row, axis=1)

        #st.write(df_saliency_channels_show)

    # _______________________________________________________ Not being used ______________________________________________________


        st.subheader("Wich channel has seen the highest change?")


        # getting the absolute values
        df_saliency_channels["t_statistic"] = abs(df_saliency_channels["t_statistic"])

        
        
        st.write(f"Mean (in each channel) -Before vs During - {sub_pillars_saliency[0]}")


        # # getting the mean of the metric in this case the followers, and comparing before and during the time. 
        st.subheader(f"Mean - Before vs During - {sub_pillars_saliency[0]}")


        if most_changed_metric == "AS_Average_Engagement":
            column_name = "earned_engagements"
        elif most_changed_metric == "AS_Usage_SoV":
            column_name = "Usage"
        elif most_changed_metric == "AS_Trial_Sov":
            column_name ="Trial/Experimentation"


        
        rows_before = []
        rows_during = []
        for channel in channels:
            df_pre_var_channel = df_pre_var[df_pre_var["message_type"] == channel ]
            df_var_channel = df_var[df_var["message_type"] == channel ]
            
            for metric in sub_pillars_saliency:
                if sub_pillars_saliency[0] == "Usage":

                    df_pre_var_channel = (df_pre_var_channel[df_pre_var_channel['journey_predictions'].isin(["Usage"])].groupby(['Week Commencing', 'brand'])['mentions'].sum()/df_pre_var_channel[df_pre_var_channel['journey_predictions'].isin(["Usage"])].groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'Usage_SoV'})
                    df_var_channel = (df_var_channel[df_var_channel['journey_predictions'].isin(["Usage"])].groupby(['Week Commencing', 'brand'])['mentions'].sum()/df_var_channel[df_var_channel['journey_predictions'].isin(["Usage"])].groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'Usage_SoV'})

                    len_before = len(df_pre_var_channel)
                    len_during = len(df_var_channel)

                    mean_before = str(round(df_pre_var_channel.Usage_SoV.mean(),2))
                    mean_after = str(round(df_var_channel.Usage_SoV.mean(),2))

                    sum_before = str(round(df_pre_var_channel.Usage_SoV.sum(),2))
                    sum_after = str(round(df_var_channel.Usage_SoV.sum(),2))


                    t_statistic, p_value = ttest_ind(df_pre_var_channel.Usage_SoV, df_var_channel.Usage_SoV)
                    
                
                elif sub_pillars_saliency[0] == "Trial or Experimentation":
                    
                    df_pre_var_channel = (df_pre_var_channel[df_pre_var_channel['journey_predictions'].isin(["Trial or Experimentation"])].groupby(['Week Commencing', 'brand'])['mentions'].sum()/df_pre_var_channel[df_pre_var_channel['journey_predictions'].isin(["Trial or Experimentation"])].groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'Trial_SoV'})
                    df_var_channel =(df_var_channel[df_var_channel['journey_predictions'].isin(["Trial or Experimentation"])].groupby(['Week Commencing', 'brand'])['mentions'].sum()/df_var_channel[df_var_channel['journey_predictions'].isin(["Trial or Experimentation"])].groupby(['Week Commencing'])['mentions'].sum()).reset_index().rename(columns={'mentions': 'Trial_SoV'})

                    len_before = len(df_pre_var_channel)
                    len_during = len(df_var_channel)

                    mean_before = str(round(df_pre_var_channel.Trial_SoV.mean(),2))
                    mean_after = str(round(df_var_channel.Trial_SoV.mean(),2))

                    sum_before = str(round(df_pre_var_channel.Trial_SoV.sum(),2))
                    sum_after = str(round(df_var_channel.Trial_SoV.sum(),2))


                    t_statistic, p_value = ttest_ind(df_pre_var_channel.Trial_SoV, df_var_channel.Trial_SoV)
                    
                else:
                    len_before = len(df_pre_var_channel)
                    len_during = len(df_var_channel)

                    mean_before = str(round(df_pre_var_channel[sub_pillars_saliency[0]].mean(),2))
                    mean_after = str(round(df_var_channel[sub_pillars_saliency[0]].mean(),2))

                    sum_before = str(round(df_pre_var_channel[sub_pillars_saliency[0]].sum(),2))
                    sum_after = str(round(df_var_channel[sub_pillars_saliency[0]].sum(),2))
                    
                    t_statistic, p_value = ttest_ind(df_pre_var_channel[metric], df_var_channel[metric])
                    

                row_before = {"time period":"before","channel":channel,"mean":mean_before,"number of mentions":len_before,column_name:sum_before,"t-statistic":t_statistic,"p-value":p_value}
                row_during =  {"time period":"during","channel":channel,"mean":mean_after,"number of mentions":len_during,column_name:sum_after,"t-statistic":t_statistic,"p-value":p_value}

                rows_before.append(row_before)
                rows_during.append(row_during)


        column_1,column_2 = st.columns(2)

        with column_1:
            df_mean_before_channel = pd.DataFrame(rows_before)
            df_mean_before_channel = df_mean_before_channel.sort_values(by="p-value",ascending=True).reset_index(drop=True)
            df_mean_before_channel["t-statistic"] = df_mean_before_channel["t-statistic"].apply(change_format)
            df_mean_before_channel["p-value"] = df_mean_before_channel["p-value"].apply(change_format)
            most_changed_channel = df_mean_before_channel.channel.iloc[0]
            #Aggregate
            df_mean_before_channel = df_mean_before_channel.style.apply(highlight_row_before,axis=1)
            st.dataframe(df_mean_before_channel,hide_index=True)

        with column_2:
            df_mean_during_channel = pd.DataFrame(rows_during)
            df_mean_during_channel = df_mean_during_channel.sort_values(by="p-value",ascending=True).reset_index(drop=True)
            df_mean_during_channel["t-statistic"] = df_mean_during_channel["t-statistic"].apply(change_format)
            df_mean_during_channel["p-value"] = df_mean_during_channel["p-value"].apply(change_format)
            #Aggregate
            df_mean_during_channel = df_mean_during_channel.style.apply(highlight_row_during,axis=1)
            st.dataframe(df_mean_during_channel,hide_index=True)

        
        st.write(f"\nThe channel that has changed the most is **'{most_changed_channel}'**")
        

        #filtered table. 
        df_var_show = df_var[df_var.message_type == most_changed_channel]

        st.dataframe(df_var_show)


    


def check_all(eq,df,brand,channel,time_period,period_pre,period_start,period_end,pre_year,post_year,aw_metrics,brand_mapping,metrics_to_see):
    for x in metrics_to_see:
    
        if x == "Awareness":

            #check Awareness
            check_awareness_high_level(eq,df,brand,time_period,period_pre,period_start,period_end)

        if x == "Saliency":

            #Checking Saliency 
            check_saliency_high_level(eq,df,brand,time_period,period_pre,period_start,period_end)

        if x =="Affinity":

            #Checking affinity
            most_changed_metric_high_level= check_affinity_high_level(eq, df, brand,time_period, period_pre, period_start, period_end,aw_metrics)
            st.write("\n")
            #print(most_changed_metric_high_level)
            most_changed_metric_high_level = most_changed_metric_high_level.lower()
            

            # Regular expression to match any string starting with "A" followed by any character, and capturing the rest
            pattern = r"a._(.+)"

            # Use re.search to find the pattern in the string
            metric = re.search(pattern,most_changed_metric_high_level).group(1)
            st.write(f"Analysing the sub-pillars for {metric}  . . . . . ")
            


            if metric:
                st.write("\n")
                #print(f"{brand} low level affinity analysis")
                # Print statement with bold text
                st.write(f" **{brand}** low level affinity analysis . . .")


                st.write("\n")
                check_affinity_low_level(df, eq, period_pre,period_start,period_end,metric, brand, channel,pre_year,post_year,brand_mapping)
            




#-------------------------------------------------------------------Equity Analysis-----------------------------------------------------------------------------------------------













#------------------------------------------------------------------------app---------------------------------------------------------------------------------------------------------------------#
def main():   
         if 'button' not in st.session_state:
                  st.session_state.button = False
         
         if 'fig' not in st.session_state:
                  st.session_state.fig = False

         # Initialize session state if not already initialized
         logout_container = st.container()
         if "inputs" not in st.session_state:
                  st.session_state.inputs = {}

         # Initialize session state variables
         if 'access' not in st.session_state:
                  st.session_state.access = False
         
         if 'login_clicked' not in st.session_state:
                  st.session_state.login_clicked = False
         
         if 'user_email' not in st.session_state:
                  st.session_state.user_email = None
         
         if not st.session_state.access:                  
                  login()
                  # Check for authorization code in URL
                  params = st.query_params
                  if "code" in params:
                           code = params["code"]
                           token = get_token_from_code(code)
                           if token:
                                    st.session_state.access_token = token
                                    st.session_state.user_email = get_user_info(st.session_state.access_token)
                                    st.query_params.clear()
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

                                    st.session_state.access = True
                                    st.rerun()
#----------------------------------------------------------------------------------------------// Logout //------------------------------------------------------------------------------------------------ 

         #if logged in
         else:
                  with st.sidebar:
                           st.image(image)
                           # user input for equity and mmm file. 
                           markets_available = ["UK"]
                           column_1,column_2 = st.columns(2)
                           
                           with column_1:
                                    market = st.selectbox('Markets', markets_available)
                                    market = market.lower()
                                    
                           if market == "uk":
                                    slang ="MMM_UK_"
                                    brand_mapping = {"aptamil":"APTAMIL" , "cow&gate": "COW & GATE", "sma": "SMA", "kendamil": "KENDAMIL", "hipp_organic": "HIPP ORGANIC"}
                                    weights_values_for_average_2021 = {"APTAMIL":0 , "COW & GATE": 0, "SMA": 0, "KENDAMIL": 0, "HIPP ORGANIC": 0}
                                    weights_values_for_average_2022 = {"APTAMIL":0 , "COW & GATE": 0, "SMA": 0, "KENDAMIL": 0, "HIPP ORGANIC": 0}
                                    weights_values_for_average_2023 = {"APTAMIL":0 , "COW & GATE": 0, "SMA": 0, "KENDAMIL": 0, "HIPP ORGANIC": 0}
                                    weights_values_for_average_2024 = {"APTAMIL":0 , "COW & GATE": 0, "SMA": 0, "KENDAMIL": 0, "HIPP ORGANIC": 0}
                                    brand_list = ["APTAMIL","COW & GATE","SMA","KENDAMIL","HIPP ORGANIC"]
                                    master_parquet = pd.read_parquet(r"uk_data_tagged_2024_09_02_14_59_00.parquet")
                           
                           # getting our equity    
                           filepath_equity,year_equity,month_equity,day_equity,hour_equity,minute_equity,second_equity = equity_info(data,market)
                           
                           # reading the equity file
                           df = reading_df(filepath_equity,sheet_name="average_smoothened")
                           df_total_uns = reading_df(filepath_equity,sheet_name="total_unsmoothened")
                           df_total_smooth = reading_df(filepath_equity,sheet_name="total_smoothened")
                           df_avg_unsmooth = reading_df(filepath_equity,sheet_name="average_unsmoothened")
                           df_significance = reading_df(filepath_equity,sheet_name="significance")
                           df_perc_changes = reading_df(filepath_equity,sheet_name="perc_changes")
                           
                           st.write(df_total_uns.head())
                           #Equity options
                           category_options,time_period_options,framework_options = equity_options(df,brand_mapping,categories_changed,framework_options_)
                           
                           #creating the market_share_weighted
                           value_columns  = value_columns_

                           with column_2:
                             category =  st.radio('Choose  category:', category_options,key='test7')

#--------------------------------------------------------------------------------------// transformations ----------------------------------------------------------------------------------
                           #creating a copy of our dataframes.
                           df_copy = df.copy()
                           df_total_uns_copy = df_total_uns.copy()
                           # Aesthetic changes --------------------------------------------------------------------------------------------------
                           #changing the names of the filtered  columns
                           ################################################################## df ####################################################################################################
                           df_copy.rename(columns=affinity_to_user,inplace=True)
                           
                           
                           
                           df_copy.brand = df_copy.brand.replace(brand_mapping)
                           
                           df_copy.rename(columns=general_equity_to_user,inplace=True)
                           
                           ################################################################## df_total_uns ####################################################################################################
                           
                           df_total_uns_copy.rename(columns=affinity_to_user,inplace=True)
                           
                           
                           df_total_uns_copy.brand = df_total_uns_copy.brand.replace(brand_mapping)
                           
                           replacements = {"weeks":"Weeks","months":"Months","quarters":"Quarters","semiannual":"Semiannual","years":"Years"}
                           df_total_uns_copy["time_period"] = df_total_uns_copy["time_period"].replace(replacements)
                           
                           
                           df_total_uns_copy["Category"] = df_total_uns_copy["Category"].replace(categories_changed)
                           
                           
                           df_total_uns_copy.rename(columns=general_equity_to_user,inplace=True)

################################################################## ##################################################################################################################
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------// Market Share Weighted----------------------------------------------------------------------------------
                  
                  with st.container():
                    tab2,tab3,tab4,tab5 = st.tabs(["📈 Market Share Weighted","🔍Average, absolute and market share weighted","📕 Final Equity plot","🎲 Equity Analysis"])
                          
                  with tab2:
                            #chosing the sheet name 
                           column_1,column_2,column_3,_ = st.columns(4)
                           with column_1:
                             sheet_name = st.selectbox("Select sheet",["Average","Absolute"])
                                    
                           with column_2:
                             smoothening_type = st.selectbox("Smoothened/ Not Smoothened",["Not Smoothened","Smoothened"])

                           if smoothening_type == "Smoothened":
                             with column_3:
                               smoothening_parameters["window_size"] = st.number_input("Window size",value=12)

                          
                           st.subheader(f"Equity Metrics Plot - Market Share Weighted {sheet_name}")
                  
                  
                           if sheet_name == "Average":
                                    sheet_name = "Average Smoothening"
                                    sheet_name_download = 'average'
                                    df_for_weighted = df_copy
                           
                           if sheet_name == "Absolute":
                                    sheet_name = "Total Unsmoothening"
                                    sheet_name_download = "total"
                                    df_for_weighted = df_total_uns_copy
                                             
                           
                           
                           
                           # getting the individual years
                           years_filtered = df_for_weighted[df_for_weighted.time_period == "Years"]            
                           years_filtered = years_filtered.time.dt.year.unique()
                           years_cols = [str(year) for year in years_filtered if year != 2020]
                  


#--------------------------------------------------------------------------------------------------------------------------// //-----------------------------------------------------
                           weights_joined = []
                           join_brand_year= []
                           keys = [ key for key in brand_mapping.keys()]
                           
                           for year in years_cols:
                                    join_brand_year.append((year,keys))
                           
                           # Assuming you want one column per key in brand_mapping
                           num_columns = len(join_brand_year)
                           num_brand = len(brand_mapping.keys())
                              
                           #colunas
                           cols = st.columns(num_columns)
                           
                           for index,col in zip(range(num_columns),cols[:]):
                                  
                                  workspace = join_brand_year[index]
                                  year,brand = workspace[0],workspace[1]
                                  st.write(year)
                                  # Assuming you want one column per key in brand_mapping
                                  num_columns = len(brand_mapping.keys())
                                  # Create the columns
                                  cols = st.columns(num_columns)
                                  if year == "2021":
                                      # Iterate over the columns and keys simultaneously
                                      for col, key in zip(cols, weights_values_for_average_2021.keys()):
                                                  year_key = f"{key}_{year}"
                                                  with col:
                                                          number = st.number_input(f"Weight for {key}", min_value=0, max_value=100, value=10,key=year_key)
                                                          weights_values_for_average_2021[key] = number / 100
                           
                                  if year == "2022":
                                      # Iterate over the columns and keys simultaneously
                                      for col, key in zip(cols, weights_values_for_average_2022.keys()):
                                                  year_key = f"{key}_{year}"
                                                  with col:
                                                          number = st.number_input(f"Weight for {key}", min_value=0, max_value=100, value=10,key=year_key)
                                                          weights_values_for_average_2022[key] = number / 100
                           
                           
                                  if year == "2023":
                                      # Iterate over the columns and keys simultaneously
                                      for col, key in zip(cols, weights_values_for_average_2023.keys()):
                                                  year_key = f"{key}_{year}"
                                                  with col:
                                                          number = st.number_input(f"Weight for {key}", min_value=0, max_value=100, value=10,key=year_key)
                                                          weights_values_for_average_2023[key] = number / 100
                           
                           
                                  if year == "2024":
                                      # Iterate over the columns and keys simultaneously
                                      for col, key in zip(cols, weights_values_for_average_2024.keys()):
                                                  year_key = f"{key}_{year}"
                                                  with col:
                                                          number = st.number_input(f"Weight for {key}", min_value=0, max_value=100, value=10,key=year_key)
                                                          weights_values_for_average_2024[key] = number / 100
                           
                           
                           
                           weights_joined.append(weights_values_for_average_2021)
                           weights_joined.append(weights_values_for_average_2022)
                           weights_joined.append(weights_values_for_average_2023)
                           weights_joined.append(weights_values_for_average_2024)

#--------------------------------------------------------------------------------------------------------------------------// //-----------------------------------------------------
                           
                           #creating the market_share_weighted
                           market_share_weighted =  weighted_brand_calculation(df_for_weighted, weights_joined,years_cols,value_columns,framework_to_user)

                           # color stuff
                           all_brands = [x for x in brand_list]
                           colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","yellow","olive","silver","darkviolet","grey"]
              
                           brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}


                           #getting the min value
                           try:
                             market_share_weighted.drop(columns=["weights","Unnamed: 24","Unnamed: 25"],inplace=True)
                           except:
                             pass
              
                   
                           #Just for this case!. 
                           market_share_weighted["Category"] = market_share_weighted["Category"].replace("Baby milk", "Baby Milk")
                         
                           # creating the columns for the app
                           right_column_1,right_column_2,left_column_1,left_column_2 = st.columns(4)
                           
                           with right_column_1:
                           #getting the date
                                    start_date = st.date_input("Select start date",value=datetime(2021, 2, 16),key='start_date')
                                    end_date =  st.date_input("Select end date",key='test1')
                           
                           # getting the parameters
                           
                           with right_column_2:
                                    st.session_state.category = category
                                    
                                    if smoothening_type == "Smoothened":
                                      market_share_weighted = smoothening_weeks(market_share_weighted,smoothening_weeks_list,affinity_to_user,framework_to_user,st.session_state.category,categories_changed,brand_mapping,smoothening_parameters["window_size"],method= 'average')
                                    else:
                                      market_share_weighted = market_share_weighted
                           

                                    market_share_weighted.dropna(inplace=True)
                                    mask = market_share_weighted["eSoV"] == 0
                                    market_share_weighted = market_share_weighted[~mask]

                           with left_column_1:    
                                    st.session_state.time_frame = st.radio('Choose  time frame:', time_period_options,key='test4')
                           
                           with left_column_2:
                                    framework = st.selectbox('Choose  framework:', value_columns,key='test5')
                           
                           
                           if st.session_state.button == False:
                                    if st.button("Run!"):
                                             #convert our dates
                                             ws = start_date.strftime('%Y-%m-%d')
                                             we = end_date.strftime('%Y-%m-%d')
                                             
                                             st.session_state.fig = Equity_plot_market_share_(market_share_weighted, st.session_state.category, st.session_state.time_frame,framework,ws,we,brand_color_mapping)
                                             st.session_state.button = True
                           else:
                                    if st.button("Run!"):
                                             #convert our dates
                                             ws = start_date.strftime('%Y-%m-%d')
                                             we = end_date.strftime('%Y-%m-%d')
                                             
                                             st.session_state.fig = Equity_plot_market_share_(market_share_weighted, st.session_state.category, st.session_state.time_frame,framework,ws,we,brand_color_mapping)
                                    
                           
                           
                           if st.session_state.button == False:
                                    pass
                           else:
                                    st.plotly_chart(st.session_state.fig,use_container_width=True)
                           
                           
                           buffer = io.BytesIO()
                           with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                    df.to_excel(writer, sheet_name='average_smoothened', index=False)
                                    
                                    df_avg_unsmooth.to_excel(writer, sheet_name='average_unsmoothened', index=False)
                                    
                                    df_total_uns.to_excel(writer, sheet_name='total_unsmoothened', index=False)
                                    
                                    df_total_smooth.to_excel(writer, sheet_name='total_smoothened', index=False)
                                    
                                    market_share_weighted.to_excel(writer,sheet_name=f'market_share_{sheet_name_download}',index=False)
                                    
                                    df_significance.to_excel(writer,sheet_name='significance',index=False)
                                    
                                    df_perc_changes.to_excel(writer,sheet_name='perc_changes',index=False)
                           
                           
                           st.download_button(
                                    label="📤",
                                    data=buffer,
                                    file_name=f"Equity_danone_{market}_{datetime.today()}.xlsx",
                                    mime="application/vnd.ms-excel")
                 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

#-------------------------------------------------------------------------------------------------------------// Compare plot//------------------------------------------------------------------------------------
                  with tab3:
                           #creating the weighted file and the plot  
                           column_1,column_2 = st.columns([1,1])
                           with column_1:
                                    sheet_name_1 = st.selectbox("Select sheet 1",["Average","Absolute", "Mkt Share Weighted"])
                           with column_2:
                                    sheet_name_2 = st.selectbox("Select sheet 2",["Absolute","Average", "Mkt Share Weighted"])
                           
                           if sheet_name_1 == "Average":
                                    sheet_1 = df
                           if sheet_name_1 == "Absolute":
                                    sheet_1 = df_total_uns
                           
                           if sheet_name_1 == "Mkt Share Weighted":
                                    sheet_1 = market_share_weighted
                           
                           if sheet_name_2 == "Average":
                                    sheet_2 = df
                           if sheet_name_2 == "Absolute":
                                    sheet_2 = df_total_uns
                           
                           if sheet_name_2 == "Mkt Share Weighted":
                                    sheet_2 = market_share_weighted
                           
                           column_1,column_2 = st.columns([1,1])
                           with column_1:
                                    weighted_1_page = st.number_input("sheet 1 weight (%)", min_value=0, max_value=100, value=75, step=5, key="sheet 1")
                           with column_2:
                                    weighted_2_page = st.number_input("sheet 2 weight (%)", min_value=0, max_value=100, value=75, step=5, key="sheet 2")
                           
                           if weighted_1_page + weighted_2_page != 100:
                                    st.warning("The values of the weights need to be equal to 100 %")
                           else:
                                    weighted_1_page = weighted_1_page/100
                                    weighted_2_page = weighted_2_page/100
                           
                           
                           df_weighted = get_weighted(sheet_1,sheet_2,weighted_1_page,weighted_2_page,brand_mapping,user_to_equity,affinity_labels,join_data_average,join_data_total,list_fix,order_list,rename_all)
                           # Comparing all the sheets
                           fig = Comparing_Equity(df,df_total_uns,df_weighted,category_options,time_period_options,framework_options,brand_mapping,affinity_to_user,categories_changed,general_equity_to_user,category)
                           st.plotly_chart(fig,use_container_width=True)
                           
                           buffer = io.BytesIO()
                           with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                    df_weighted.to_excel(writer, sheet_name=f'weighted_combined', index=False)
                           
                           
                           new_file_name = f"{sheet_name_1}_{sheet_name_2}_weighted_{datetime.today()}.xlsx"
                           
                           st.download_button(
                           label="📤",
                           data=buffer,
                           file_name=new_file_name)

                           
#--------------------------------------------------------------------------------------// Equity plot //----------------------------------------------------------------------------------
                  with tab4:
                           #chosing the sheet name 
                           column_1,_,_,_ = st.columns(4)
                           with column_1:
                                    sheet_name = st.selectbox("Select sheet",["Average","Absolute", "Mkt Share Weighted"])
                           
                           if sheet_name == "Average":
                                    sheet_name = "Average Smoothening"
                           if sheet_name == "Absolute":
                                    sheet_name = "Total Unsmoothening"

                           if sheet_name =="Mkt Share Weighted":
                                    sheet_name = "Market Share Weighted"


                           # color stuff
                           all_brands = [x for x in brand_list]
                           colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","yellow","olive","silver","darkviolet","grey"]
              
                           brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
                
                                           
                           if sheet_name == "Average Smoothening":
                                    fig = Equity_plot(df,category_options,time_period_options,framework_options,sheet_name,framework_to_user,brand_color_mapping,category)
                                    st.plotly_chart(fig,use_container_width=True)
                           
                           if sheet_name == "Total Unsmoothening":
                                    fig = Equity_plot(df_total_uns,category_options,time_period_options,framework_options,sheet_name,framework_to_user,brand_color_mapping,category)
                                    st.plotly_chart(fig,use_container_width=True)
                           
                           if sheet_name == "Market Share Weighted":
                                    fig = Equity_plot(market_share_weighted,category_options,time_period_options,framework_options,sheet_name,framework_to_user,brand_color_mapping,category)
                                    st.plotly_chart(fig,use_container_width=True)

#--------------------------------------------------------------------------------------Equity Analysis--------------------------------------------------------------------------
        
                  #Equity analysis tab
                  with tab5:
                      column_1,column_2,column_3 = st.columns([1,1,1])
                      
                      
                      with column_1:
                          brand = st.selectbox("Brand",brand_list)
          
                          channel_filter = "News"
          
                          time_period = "Weeks"
                      with column_2:
                          
                          test_ = df.loc[df.eSoV> 0]
              
                          period_start =  st.date_input("Select start period",value=datetime(2022, 2, 16),key="Equity_analysis")
                          period_start = pd.to_datetime(period_start)
          
                          period_end =  st.date_input("Select end period",key="Equity_analysis_2",value=test_["time"].iloc[-1])
                          period_end = pd.to_datetime(period_end)
                      
                          difference_of_time = period_end - period_start
                          difference_in_days = difference_of_time.days
          
                          period_pre = period_start - timedelta(days=difference_in_days)
          
          
                      with column_3:
                          
                          master_parquet["year"] = master_parquet["year"].astype(int)
          
                          pre_year = period_start.year
          
                          post_year = period_end.year
          
                          if pre_year == post_year:
                              st.warning("Previous Year and Post year are the same")
                              st.warning("Going to take one year from the period end")
                              pre_year = post_year - 1 
          
                          
                          metrics_to_see = st.multiselect("Metric to analyse",["Awareness","Saliency","Affinity"])
          
          
                      check_all(df,master_parquet,brand,channel_filter,time_period,period_pre,period_start,period_end,pre_year,post_year,aw_metrics,brand_mapping,metrics_to_see)



         
                  # Custom CSS to push the logout button to the right and style it
                  # Custom CSS
                  st.markdown("""
                  <style>
                  #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 2rem;}
                  .stButton > button.logout-button {
                    padding: 0.25rem 0.5rem !important;
                    font-size: 0.1rem !important;
                    min-height: 0px !important;
                    height: auto !important;
                    line-height: normal !important;
                  }
                  </style>
                  """, unsafe_allow_html=True)
                  
                  # Custom HTML for spacing
                  html_code = """
                  <div style="margin-left: 20px;">
                  </div>
                  """
                  
                  with logout_container:
                           col1, col2, col3 = st.columns([6,1,1])
                  with col2:
                    components.html(html_code, height=3)
                    st.markdown(f'<p style="font-size:12px;">{st.session_state.user_email}</p>', unsafe_allow_html=True)
                  
                  with col3:
                    components.html(html_code, height=3)
                    if st.session_state.get('access', False):
                        if st.button("Logout", key="small_button", type="secondary", use_container_width=False, 
                                        help="Click to logout", kwargs={"class": "small_button"}):
                            st.markdown("""
                            <meta http-equiv="refresh" content="0; url='https://equitytrackingplots-jqwdds7kl4pnw98dcmxfbz.streamlit.app/'" />
                            """, unsafe_allow_html=True)
                                               
if __name__=="__main__":
    main()   













