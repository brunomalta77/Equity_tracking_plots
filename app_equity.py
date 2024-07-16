import pandas as pd
import numpy as np
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


#page config
st.set_page_config(page_title="Equity Tracking plots app",page_icon="💼",layout="wide")
logo_path = r"data/brand_logo.png"
logo_microsoft_path =  r"https://www.shareicon.net/data/256x256/2015/09/15/101518_microsoft_512x512.png"
image = Image.open(logo_path)
#image_microsoft = Image.open(logo_microsoft_path)
#colors used for the plots
colors = ["blue", "green", "red", "purple", "orange","teal","black","paleturquoise","indigo","darkseagreen","gold","darkviolet","firebrick","navy","deeppink",
         "orangered"]


# creating a user database type for getting access to the app -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Microsoft Azure AD configurations
CLIENT_ID = "1363ef83-1b3e-4bd7-add0-9f1bd40d69ba"
CLIENT_SECRET = "142545f0-6624-4c08-9288-73e78b8906d0"
AUTHORITY = "https://login.microsoftonline.com/68421f43-a2e1-4c77-90f4-e12a5c7e0dbc"
SCOPE = ["User.Read"]
REDIRECT_URI = "https://equitytrackingplots-idpmnwwksvjnrgdu5rmitk.streamlit.app" # This should match your Azure AD app configuration

# Initialize MSAL application
app = msal.ConfidentialClientApplication(
    CLIENT_ID, authority=AUTHORITY,
    client_credential=CLIENT_SECRET)

def get_auth_url():
    return app.get_authorization_request_url(SCOPE, redirect_uri=REDIRECT_URI)

def get_token_from_code(code):
    result = app.acquire_token_by_authorization_code(code, SCOPE, redirect_uri=REDIRECT_URI)
    return result.get("access_token")

def get_user_info(access_token):
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get('https://graph.microsoft.com/v1.0/me', headers=headers)
    return response.json()

def login():
         auth_url = get_auth_url()
         #st.image(image_microsoft, width=20)
         #st.markdown(f'[Log in with microsoft]({auth_url})')

         html_string = f"""
         <a href="{auth_url}">
             <img src="{logo_microsoft_path}" style="width: 20px; height: 20px; vertical-align: middle;">
                Log in with Microsoft
         </a>
         """

         # Use st.markdown to render the HTML
         st.markdown(html_string, unsafe_allow_html=True)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


col1, col2 = st.columns([4, 1])  # Adjust the width ratios as needed

# Logo on the left
with col2:
    st.image(image)  # Adjust the width as needed

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
    return df

#mmm file 
@st.cache_data() 
def processing_mmm(filepath):
    df_vol = pd.read_excel(filepath)
    sales = list(df_vol[df_vol["Metric"] == "SalesValue"]["Value"])
    volume = list(df_vol[df_vol["Metric"] == "SalesVol"]["Value"])
    res = [round(x / y,2) if y != 0 else 0 for x, y in zip(sales, volume)]
    df_vol = df_vol[df_vol["Metric"]== "SalesVol"]
    df_vol["Price_change"] = res
    df_vol.rename(columns={"Value":"Volume_share"},inplace=True)
    df_vol=df_vol.groupby(['time','Y','H','QT','M','W','brand','Metric','Category'])[['Volume_share','Price_change']].sum().reset_index()
    return df_vol

# Media files
@st.cache_data()
def media_plan(filepath,sheet_spend,sheet_week):
         df_uk_spend = pd.read_excel(filepath,sheet_name=sheet_spend)
         df_uk_weeks = pd.read_excel(filepath,sheet_name=sheet_week)
         return (df_uk_spend,df_uk_weeks)



@st.cache_data()
def get_weighted(df,df_total_uns,weighted_avg,weighted_total):
    # drop any nan values
    df.dropna(inplace=True)
    df_total_uns.dropna(inplace=True)
    
    affinity_labels = ['AF_Entry_point','AF_Brand_Love','AF_Baby_Milk','AF_Adverts_Promo','AF_Value_for_Money','AF_Buying_Exp','AF_Prep_Milk','AF_Baby_exp']
    
    # Doing the percentual in total_unsmoothened
    for aff in affinity_labels:
        grouped = df_total_uns.groupby(["time","time_period"])[aff].transform("sum")
        df_total_uns["total"] = grouped
        df_total_uns[aff] = df_total_uns[aff] / df_total_uns['total'] * 100

    # Let's join by time and brand
    join_data = pd.merge(df,df_total_uns,on=["time","brand","time_period"],suffixes=("_average","_total"))

    #splitting them 

    final_average = join_data[['time', 'time_period', 'brand', 'AA_eSoV_average', 'AA_Reach_average',
                'AA_Brand_Breadth_average', 'AS_Average_Engagement_average',
                'AS_Usage_SoV_average', 'AS_Search_Index_average',
                'AS_Brand_Centrality_average','AF_Entry_point_average', 'AF_Brand_Love_average', 'AF_Baby_Milk_average','AF_Adverts_Promo_average','AF_Value_for_Money_average','AF_Buying_Exp_average',
                'AF_Prep_Milk_average','AF_Baby_exp_average',
                'Framework_Awareness_average', 'Framework_Saliency_average',
                'Framework_Affinity_average', 'Total_Equity_average',
                'Category_average']]


    final_total = join_data[['time', 'time_period', 'brand', 'AA_eSoV_total', 'AA_Reach_total',
                  'AA_Brand_Breadth_total', 'AS_Average_Engagement_total',
                  'AS_Usage_SoV_total', 'AS_Search_Index_total',
                  'AS_Brand_Centrality_total','AF_Entry_point_total', 'AF_Brand_Love_total', 'AF_Baby_Milk_total','AF_Adverts_Promo_total','AF_Value_for_Money_total','AF_Buying_Exp_total',
                  'AF_Prep_Milk_total','AF_Baby_exp_total',
                  'Framework_Awareness_total', 'Framework_Saliency_total',
                  'Framework_Affinity_total', 'Total_Equity_total', 'Category_total']]

    list_fix = ['time', 'time_period', 'brand', 'AA_eSoV_average', 'AA_Reach_average',
                  'AA_Brand_Breadth_average', 'AS_Average_Engagement_average',
                  'AS_Usage_SoV_average', 'AS_Search_Index_average',
                  'AS_Brand_Centrality_average','Framework_Awareness_average', 'Framework_Saliency_average','Total_Equity_average',
                  'Category_average']
            

    #Getting first the fixed stuff
    weighted_average_equity = final_average[list_fix]

    for aff_pilar in affinity_labels:
        weighted_average_equity["weighted_" + aff_pilar] = 0
        for index,row in final_average.iterrows():
            weighted_average_equity["weighted_" + aff_pilar][index] = round(((weighted_avg * final_average[aff_pilar + "_average"][index]) + (weighted_total * final_total[aff_pilar + "_total"][index])),2)
        
    #getting the new framework affinity
    weighted_average_equity["weighted_Framework_Affinity"] = round((weighted_average_equity["weighted_AF_Entry_point"] + weighted_average_equity["weighted_AF_Brand_Love"] + weighted_average_equity["weighted_AF_Baby_Milk"] +weighted_average_equity["weighted_AF_Adverts_Promo"] + weighted_average_equity["weighted_AF_Value_for_Money"] + weighted_average_equity["weighted_AF_Buying_Exp"] +  weighted_average_equity["weighted_AF_Prep_Milk"] + weighted_average_equity["weighted_AF_Baby_exp"]  )/8,2)

    # getting the new total equity

    weighted_average_equity["Total_Equity"] = round((weighted_average_equity["weighted_Framework_Affinity"] + weighted_average_equity["Framework_Awareness_average"] + weighted_average_equity["Framework_Saliency_average"])/3,2) 

    #ordering
    order = ['time', 'time_period', 'brand', 'AA_eSoV_average', 'AA_Reach_average',
       'AA_Brand_Breadth_average', 'AS_Average_Engagement_average',
       'AS_Usage_SoV_average', 'AS_Search_Index_average',
       'AS_Brand_Centrality_average','weighted_AF_Entry_point','weighted_AF_Brand_Love','weighted_AF_Baby_Milk','weighted_AF_Adverts_Promo','weighted_AF_Value_for_Money','weighted_AF_Buying_Exp','weighted_AF_Prep_Milk','weighted_AF_Baby_exp',
        'Framework_Awareness_average',
       'Framework_Saliency_average','weighted_Framework_Affinity','Total_Equity',"Category_average"]
    weighted_average_equity = weighted_average_equity[order]

    weighted_average_equity.rename(columns={'AA_eSoV_average':'AA_eSoV', 'AA_Reach_average':'AA_Reach',
       'AA_Brand_Breadth_average':'AA_Brand_Breadth', 'AS_Average_Engagement_average':'AS_Average_Engagement',
       'AS_Usage_SoV_average':'AS_Usage_SoV', 'AS_Search_Index_average':'AS_Search_Index',
       'AS_Brand_Centrality_average':'AS_Brand_Centrality','weighted_AF_Entry_point':'AF_Entry_point','weighted_AF_Brand_Love':'AF_Brand_Love',
       'weighted_AF_Brand_Love':'AF_Brand_Love','weighted_AF_Baby_Milk':'AF_Baby_Milk','weighted_AF_Buying_Exp':'AF_Buying_Exp','weighted_AF_Prep_Milk':'AF_Prep_Milk'
       ,'weighted_AF_Baby_exp':'AF_Baby_exp',
       'weighted_AF_Adverts_Promo':'AF_Adverts_Promo',
       'weighted_AF_Value_for_Money':'AF_Value_for_Money','Framework_Awareness_average':'Framework_Awareness',
       'Framework_Saliency_average':'Framework_Saliency','weighted_Framework_Affinity':'Framework_Affinity','Category_average':'Category'},inplace=True)

    return weighted_average_equity


#---------------------------------------------------------------------------------------////------------------------------------------------------------------------

# Market_share_weighted_average
def weighted_brand_calculation(df, weights, value_columns):
    # Ensure brand names in the dataframe match the keys in the weights dictionary
    df['brand'] = df['brand'].str.lower()
    
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
    
    return result_df
#------------------------------------------------------------------------------------------------------------------------------------------------------------------






#merged file
@st.cache_data() 
def merged_file(df,df_vol):
    df_merged = pd.merge(df_vol,df,on=["time","brand","Category"],how="inner")
    return df_merged

# Function to calculate confidence intervals
def calculate_confidence_intervals(data, confidence=0.90):
    if confidence == 0.90:
        multi = 1.645
    if confidence == 0.95:
        multi = 1.960
    else:
        multi = 2.576
    
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    margin_of_error = std_err * 1.645  # 1.96 for 95% confidence interval (z-score)
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound

def equity_info(data,market_flag):
    if market_flag == "UK":
        market_flag = "UK_equity_"
    for x in os.listdir(data):
        if market_flag in x:
            filepath_equity = os.path.join(data,x)
            info_number = [x for x in x.split("_") if x >= "0" and x <="9"]
            year_equity,month_equity,day_equity,hour_equity,minute_equity = info_number[:5]
            second_equity = info_number[-1].split(".")[0]
    
    return filepath_equity,year_equity,month_equity,day_equity,hour_equity,minute_equity,second_equity


def mmm_info(data,slang_flag):
    for x in os.listdir(data):
        if slang_flag in x:
            filepath_mmm = os.path.join(data,x)
            info_number = [x for x in x.split("_") if x >= "0" and x <="9"]
            day_mmm,month_mmm,year_mmm,hour_mmm,minute_mmm = info_number[:5]
            second_mmm = info_number[-1].split(".")[0]
    
    return filepath_mmm,year_mmm,month_mmm,day_mmm,hour_mmm,minute_mmm,second_mmm


# media processing
def media_spend_processed(df,df_spend,df_weeks):
         # making everything as a string and in lower case
         df_weeks["w"] = df_weeks["w"].apply(lambda x : str(x).lower())
         df_weeks["y"] = df_weeks["y"].apply(lambda x : str(x).lower())
         df_weeks["m"] = df_weeks["m"].apply(lambda x : str(x).lower())
         df_weeks["qt"] = df_weeks["qt"].apply(lambda x : str(x).lower())
         df_weeks["h"] = df_weeks["h"].apply(lambda x : str(x).lower())
         
         # renaming and put everything into string and lower case
         df_spend.rename(columns={"Group Type":"Media_spend"},inplace=True)
         df_spend.rename(columns={"time_frame":"time_period"},inplace=True)
         df_spend["Date"] = df_spend["Date"].apply(lambda x : str(x).lower())
         
         # mapping our codes from the y,h,qt,m,w into time
         trans_dic = {}
         trans_dic.update({k: i for k, i in zip(df_weeks["w"], df_weeks["time"])})
         trans_dic.update({k: i for k, i in zip(df_weeks["y"], df_weeks["time"])})
         trans_dic.update({k: i for k, i in zip(df_weeks["m"], df_weeks["time"])})
         trans_dic.update({k: i for k, i in zip(df_weeks["qt"], df_weeks["time"])})
         trans_dic.update({k: i for k, i in zip(df_weeks["h"], df_weeks["time"])})
         
         #creating the time column
         df_spend["time"] = [trans_dic[x] for x in df_spend["Date"]]
         
         #grouping by
         df_spend=df_spend.groupby(['Date','time_period','brand','Media_spend','Category','time'])['value'].sum().reset_index()
         
         merged_df = pd.merge(df_spend,df,on=["time","brand","Category"],how="inner")
         
         return merged_df



def equity_options(df):
    category_options = df["Category"].unique()
    time_period_options = df["time_period"].unique()
    framework_options = ['AF_Entry_point','AF_Brand_Love','AF_Baby_Milk','AF_Adverts_Promo','AF_Value_for_Money','AF_Buying_Exp','AF_Prep_Milk','AF_Baby_exp',"Framework_Awareness","Framework_Saliency","Framework_Affinity","Total_Equity"]
    return (category_options,time_period_options,framework_options)



def merged_options(df):
    category_options_merged = df["Category"].unique()
    time_period_options_merged = df["time_period"].unique()
    framework_options_merged = ['AF_Entry_point', 'AF_Brand_Love', 'AF_Adverts_Promo','AF_Prep_Meal','AF_Experience','AF_Value_for_Money', "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity"]
    framework_options_value = ["Volume_share","Price_change"]
    return(category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value)


def merged_options_media(df):
         category_options_merged_media = df["Category"].unique()
         time_period_options_merged_media = df["time_period_x"].unique()
         framework_options_media = ['AF_Entry_point', 'AF_Brand_Love', 'AF_Adverts_Promo','AF_Prep_Meal','AF_Experience','AF_Value_for_Money', "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity"]
         framework_options_value_media = ["value"]
         return(category_options_merged_media,time_period_options_merged_media,framework_options_media, framework_options_value_media)
         

# Equity_plot
def Equity_plot(df,categories,time_frames,frameworks,sheet_name):
    if sheet_name == "Average Smoothening":
        name = "Average"
    if sheet_name == "Total Unsmoothening":
        name = "Absolute"
    if sheet_name == "Weighted":
        name = "Weighted"

    st.subheader(f"Equity Metrics Plot - {name}")

    # creating the columns for the app
    right_column_1,right_column_2,left_column_1,left_column_2 = st.columns(4)
    
    with right_column_1:
    #getting the date
        start_date = st.date_input("Select start date",value=datetime(2020, 1, 1))
        end_date =  st.date_input("Select end date")
        #convert our dates
        ws = start_date.strftime('%Y-%m-%d')
        we = end_date.strftime('%Y-%m-%d')
    # getting the parameters
    with right_column_2:
        category = st.radio('Choose your category:', categories)
        
    with left_column_1:    
        time_frame = st.radio('Choose your time frame:', time_frames)
    
    with left_column_2:
        framework = st.selectbox('Choose your framework:', frameworks)
    
    #filtering
    df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    
    df_filtered = df_filtered.sort_values(by="time")
    
    
    # color stuff
    all_brands = [x for x in df["brand"].unique()]
    colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","yellow","olive","silver","darkviolet","grey"]

    brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
    
    fig = px.line(df_filtered, x="time", y=framework, color="brand", color_discrete_map=brand_color_mapping)

    
    if time_frame == "months":
        unique_months = df_filtered['time'].dt.to_period('M').unique()

        # Customize the x-axis tick labels to show one label per month
        tickvals = [f"{m.start_time}" for m in unique_months]
        ticktext = [m.strftime("%B %Y") for m in unique_months]

        # Update x-axis ticks
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
        
        return fig

    if time_frame == "quarters":

        unique_quarters = df_filtered['time'].dt.to_period('Q').unique()

        # Customize the x-axis tick labels to show one label per quarter
        tickvals = [f"{q.start_time}" for q in unique_quarters]
        ticktext = [f"Q{q.quarter} {q.year}" for q in unique_quarters]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
        
        return fig


    if time_frame =="years":
        # Extract unique years from the "time" column
        unique_years = df_filtered['time'].dt.year.unique()

        # Customize the x-axis tick labels to show only one label per year
        fig.update_xaxes(tickvals=[f"{year}-01-01" for year in unique_years], ticktext=unique_years, tickangle=45)
        
        return fig


    if time_frame == "weeks":
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


#-----------------------------------------------------------------------------------------------//


# Equity_plot for market share weighted average
def Equity_plot_market_share_(df,categories,time_frames,frameworks,sheet_name):
    # creating the columns for the app
    right_column_1,right_column_2,left_column_1,left_column_2 = st.columns(4)
    
    with right_column_1:
    #getting the date
        start_date = st.date_input("Select start date",value=datetime(2020, 1, 1),key='start_date')
        end_date =  st.date_input("Select end date",key='test1')
        #convert our dates
        ws = start_date.strftime('%Y-%m-%d')
        we = end_date.strftime('%Y-%m-%d')
    # getting the parameters
    with right_column_2:
        category = st.radio('Choose your category:', categories,key='test3')
        
    with left_column_1:    
        time_frame = st.radio('Choose your time frame:', time_frames,key='test4')
    
    with left_column_2:
        framework = st.selectbox('Choose your framework:', frameworks,key='test5')
    
    #filtering
    df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    
    df_filtered = df_filtered.sort_values(by="time")
    
    
    # color stuff
    all_brands = [x for x in df["brand"].unique()]
    colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","yellow","olive","silver","darkviolet","grey"]

    brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
    
    fig = px.line(df_filtered, x="time", y=framework, color="brand", color_discrete_map=brand_color_mapping)

    
    if time_frame == "months":
        unique_months = df_filtered['time'].dt.to_period('M').unique()

        # Customize the x-axis tick labels to show one label per month
        tickvals = [f"{m.start_time}" for m in unique_months]
        ticktext = [m.strftime("%B %Y") for m in unique_months]

        # Update x-axis ticks
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
        
        return fig

    if time_frame == "quarters":

        unique_quarters = df_filtered['time'].dt.to_period('Q').unique()

        # Customize the x-axis tick labels to show one label per quarter
        tickvals = [f"{q.start_time}" for q in unique_quarters]
        ticktext = [f"Q{q.quarter} {q.year}" for q in unique_quarters]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
        
        return fig


    if time_frame =="years":
        # Extract unique years from the "time" column
        unique_years = df_filtered['time'].dt.year.unique()

        # Customize the x-axis tick labels to show only one label per year
        fig.update_xaxes(tickvals=[f"{year}-01-01" for year in unique_years], ticktext=unique_years, tickangle=45)
        
        return fig


    if time_frame == "weeks":
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
#-----------------------------------------------------------------------------------------------//--------------------------------------






#Used to comparing the Equity from different sheets
def Comparing_Equity(df,df_total_uns,weighted_df,categories,time_frames,frameworks):
    st.subheader(f"Compare Average vs Absolute vs Weighted Affinity")

    # creating the columns for the app
    right_column_1,right_column_2,left_column_1,left_column_2 = st.columns(4)
    
    with right_column_1:
    #getting the date
        start_date = st.date_input("Select start date",value=datetime(2020, 1, 1),key="test_1")
        end_date =  st.date_input("Select end date",key='test_2')
        #convert our dates
        ws = start_date.strftime('%Y-%m-%d')
        we = end_date.strftime('%Y-%m-%d')
    # getting the parameters
    with right_column_2:
        category = st.radio('Choose your category:', categories,key="test_3")
        
    with left_column_1:    
        time_frame = st.radio('Choose your time frame:', time_frames,key="test_4")
    
    with left_column_2:
        framework = st.selectbox('Choose your framework:', frameworks,key="test_5")
        my_brand = st.multiselect('Choose your brand',df.brand.unique())
    
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

    
    if time_frame == "months":
        unique_months = df_filtered['time'].dt.to_period('M').unique()

        # Customize the x-axis tick labels to show one label per month
        tickvals = [f"{m.start_time}" for m in unique_months]
        ticktext = [m.strftime("%B %Y") for m in unique_months]

        # Update x-axis ticks
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
        
        return fig

    if time_frame == "quarters":

        unique_quarters = df_filtered['time'].dt.to_period('Q').unique()

        # Customize the x-axis tick labels to show one label per quarter
        tickvals = [f"{q.start_time}" for q in unique_quarters]
        ticktext = [f"Q{q.quarter} {q.year}" for q in unique_quarters]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
        
        return fig


    if time_frame =="years":
        # Extract unique years from the "time" column
        unique_years = df_filtered['time'].dt.year.unique()

        # Customize the x-axis tick labels to show only one label per year
        fig.update_xaxes(tickvals=[f"{year}-01-01" for year in unique_years], ticktext=unique_years, tickangle=45)
        
        return fig


    if time_frame == "weeks":
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





def market_share_plot(df,categories):
         # creating the columns layout
         left_column, right_column = st.columns(2)

         with left_column:
                  #getting the date
                  start_date = st.date_input("Select start date",key="1",value=datetime(2020, 1, 1))
                  end_date =  st.date_input("Select end date",key="2")
                  #convert our dates
                  ws = start_date.strftime('%Y-%m-%d')
                  we = end_date.strftime('%Y-%m-%d')
         
         with right_column:
                  # getting the parameters
                  category = st.radio('Choose your category:', categories)
         
         #filtering
         df_filtered =  df[(df["Category"] == category)]
         df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
         
         all_brands = [x for x in df["brand"].unique()]
         brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
         
         fig = px.line(df_filtered, x="time", y="Volume_share",color="brand",color_discrete_map=brand_color_mapping)

         # Extract unique quarters from the "time" column
         unique_quarters = df_filtered['time'].dt.to_period('Q').unique()
         
         # Customize the x-axis tick labels to show one label per quarter
         tickvals = [f"{q.start_time}" for q in unique_quarters]
         ticktext = [f"Q{q.quarter} {q.year}" for q in unique_quarters]
         
         fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
         fig.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}')
         
         
         return fig


def buble_plot(df,categories,time_frames,frameworks,values):
         st.write("4 dimensions: Time(x) /Equity(y)/ Brands(color), Volume share or Price change (buble size)")
         # getting the columns
         left_column_1,right_column_1,right_column_2 = st.columns(3)
         
         with left_column_1:
                  #getting the date
                  start_date = st.date_input("Select start date",key="3",value=datetime(2020, 1, 1))
                  end_date =  st.date_input("Select end date",key="4")
                  #convert our dates
                  ws = start_date.strftime('%Y-%m-%d')
                  we = end_date.strftime('%Y-%m-%d')
         
         with right_column_1:
                  # getting the parameters
                  category = st.radio('Choose your category:', categories,key="90")
                  time_frame = st.radio('Choose your time frame:', time_frames,key="6")
         with right_column_2:
                  framework = st.selectbox('Choose your framework:', frameworks,key="7")
                  value = st.selectbox('Choose  Price Change / Volume share:', values,key="8")
         
         #filter
         df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
         df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
         
         df_filtered = df_filtered.sort_values(by="time")

         
         all_brands = [x for x in df["brand"].unique()]
         brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
         
         fig = px.scatter(df_filtered, x="time", y=framework, color="brand",color_discrete_map=brand_color_mapping ,size=value,color_discrete_sequence=["blue", "green", "red", "purple", "orange"])
         
         if time_frame == "months":
                  # Extract unique months from the "time" column
                  unique_months = df_filtered['time'].dt.to_period('M').unique()
                  
                  # Customize the x-axis tick labels to show one label per month
                  tickvals = [f"{m.start_time}" for m in unique_months]
                  ticktext = [m.strftime("%B %Y") for m in unique_months]
                  
                  fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
                  fig.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}<br>Size: %{marker.size:.2s}')
                  return fig
         
         if time_frame == "quarters":
                  # Extract unique quarters from the "time" column
                  unique_quarters = df_filtered['time'].dt.to_period('Q').unique()
         
                 # Customize the x-axis tick labels to show one label per quarter
                  tickvals = [f"{q.start_time}" for q in unique_quarters]
                  ticktext = [f"Q{q.quarter} {q.year}" for q in unique_quarters]
         
                  fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
                  fig.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}<br>Size: %{marker.size:.2s}')

                  return fig
         
         if time_frame =="years":
                  # Extract unique years from the "time" column
                  unique_years = df_filtered['time'].dt.year.unique()
                  
                  # Customize the x-axis tick labels to show only one label per year
                  fig.update_xaxes(tickvals=[f"{year}-01-01" for year in unique_years], ticktext=unique_years, tickangle=45)
                  fig.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}<br>Size: %{marker.size:.2s}')


                  return fig
         
         if time_frame=="weeks" :
                  # Extract unique weeks from the "time" column
                  unique_weeks = pd.date_range(start=ws, end=we, freq='W').date
                  
                  # Customize the x-axis tick labels to show the start date of each week
                  tickvals = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]
                  ticktext = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]
                  
                  fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
                  fig.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}<br>Size: %{marker.size:.2s}')


                  return fig

         else:
                  # Extract unique semiannual periods from the "time" column
                  unique_periods = pd.date_range(start=ws, end=we, freq='6M').date
                  
                  # Customize the x-axis tick labels to show the start date of each semiannual period
                  tickvals = [period.strftime('%Y-%m-%d') for period in unique_periods]
                  ticktext = [f"Semiannual {i} - {period.strftime('%Y')}" for i, period in enumerate(unique_periods)]
                  
                  fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
                  fig.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}<br>Size: %{marker.size:.2s}')


                  return fig


# Creating the Subplots
def sub_plots(df,categories,time_frames,frameworks,values):
         st.write("First plot with 3 dimensions (Time(x)/Equity(y)/brands(color) second plot only with the volume_share as histogram")
         #getting our columns layout
         left_column_1,right_column_1,right_column_2 = st.columns(3)
         
         with left_column_1:
                  #getting the date
                  start_date = st.date_input("Select start date",key="9",value=datetime(2020, 1, 1))
                  end_date =  st.date_input("Select end date",key="10")
                  #convert our dates
                  ws = start_date.strftime('%Y-%m-%d')
                  we = end_date.strftime('%Y-%m-%d')
         with right_column_1:
                  # getting the parameters
                  category = st.radio('Choose your category:', categories,key="11")
                  time_frame = st.radio('Choose your time frame:', time_frames,key="12")
         with right_column_2:
                  framework = st.selectbox('Choose your framework:', frameworks,key="13")
                  value = st.selectbox('Choose  Price Change / Volume share:', values,key="14")
         
         #filter
         df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
         df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
         df_filtered = df_filtered.sort_values(by="time")


         
         sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
         
         
         all_brands = [x for x in df["brand"].unique()]
         brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
         
         line_plot = px.line(df_filtered, x="time", y=framework, color="brand",color_discrete_map=brand_color_mapping ,color_discrete_sequence=["blue", "green", "red", "purple", "orange"])
         histogram = px.histogram(df_filtered,x="time",y=value,color="brand",color_discrete_map=brand_color_mapping ,color_discrete_sequence=["blue", "green", "red", "purple", "orange"],nbins=200)


         line_plot.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}')

         # Update custom hover template for histogram (scatter plot)
         histogram.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}')
         
         if time_frame == "months":
                  # Extract unique months from the "time" column
                  unique_months = df_filtered['time'].dt.to_period('M').unique()
                  
                  # Customize the x-axis tick labels to show one label per month
                  tickvals = [f"{m.start_time}" for m in unique_months]
                  ticktext = [m.strftime("%B %Y") for m in unique_months]
                  
                  # Create subplots with separate figures
                  sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                  
                  # Add line plot to the first subplot
                  for trace in line_plot.data:
                     sub_fig.add_trace(trace, row=1, col=1)
                  
                  # Add histogram to the second subplot
                  for trace in histogram.data:
                     sub_fig.add_trace(trace, row=2, col=1)
                  
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=1, col=1)
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=2, col=1)

         if time_frame == "quarters":
                  unique_quarters = df_filtered['time'].dt.to_period('Q').unique()
                  
                  # Customize the x-axis tick labels to show one label per quarter
                  tickvals = [f"{q.start_time}" for q in unique_quarters]
                  ticktext = [f"Q{q.quarter} {q.year}" for q in unique_quarters]
                  
                  # Create subplots with separate figures
                  sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                  
                  # Add line plot to the first subplot
                  for trace in line_plot.data:
                     sub_fig.add_trace(trace, row=1, col=1)
                  
                  # Add histogram to the second subplot
                  for trace in histogram.data:
                     sub_fig.add_trace(trace, row=2, col=1)
                  
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=1, col=1)
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=2, col=1)

         if time_frame =="years":
                  # Extract unique years from the "time" column
                  unique_years = df_filtered['time'].dt.year.unique()
                  
                  # Create subplots with separate figures
                  sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                  
                  # Add line plot to the first subplot
                  for trace in line_plot.data:
                           sub_fig.add_trace(trace, row=1, col=1)
                  
                  # Add histogram to the second subplot
                  for trace in histogram.data:
                           sub_fig.add_trace(trace, row=2, col=1)
                  
                  sub_fig.update_xaxes(tickvals=[f"{year}-01-01" for year in unique_years], ticktext=unique_years, tickangle=45, row=1, col=1)
                  sub_fig.update_xaxes(tickvals=[f"{year}-01-01" for year in unique_years], ticktext=unique_years, tickangle=45, row=2, col=1)

         if time_frame== "weeks":
                  # Extract unique weeks from the "time" column
                  unique_weeks = pd.date_range(start=ws, end=we, freq='W').date
                  
                  # Customize the x-axis tick labels to show the start date of each week
                  tickvals = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]
                  ticktext = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]
                  
                  # Create subplots with separate figures
                  sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                  
                  # Add line plot to the first subplot
                  for trace in line_plot.data:
                     sub_fig.add_trace(trace, row=1, col=1)
                  
                  # Add histogram to the second subplot
                  for trace in histogram.data:
                     sub_fig.add_trace(trace, row=2, col=1)
                  
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=1, col=1)
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=2, col=1)

         if time_frame == "semiannual":
                  # Extract unique semiannual periods from the "time" column
                  unique_periods = pd.date_range(start=ws, end=we, freq='6M').date
                  
                  # Customize the x-axis tick labels to show the start date of each semiannual period
                  tickvals = [period.strftime('%Y-%m-%d') for period in unique_periods]
                  ticktext = [f"Semiannual {i} - {period.strftime('%Y')}" for i, period in enumerate(unique_periods)]
                  
                  sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                  
                  # Add line plot to the first subplot
                  for trace in line_plot.data:
                     sub_fig.add_trace(trace, row=1, col=1)
                  
                  # Add histogram to the second subplot
                  for trace in histogram.data:
                     sub_fig.add_trace(trace, row=2, col=1)
                  
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=1, col=1)
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=2, col=1)
         
         return sub_fig


# Subplot Media
def sub_plots_media(df,categories,time_frames,frameworks,values):
         st.subheader("Equity vs Media spend")
         left_column_1,right_column_1,right_column_2 = st.columns(3)
         
         with left_column_1:
                  #getting the date
                  start_date = st.date_input("Select start date",key="67567567",value=datetime(2020, 1, 1))
                  end_date =  st.date_input("Select end date",key="567567567")
                  #convert our dates
                  ws = start_date.strftime('%Y-%m-%d')
                  we = end_date.strftime('%Y-%m-%d')
         with right_column_1:
                  # getting the parameters
                  category = st.radio('Choose your category:', categories,key="12756757567")
                  time_frame = st.radio('Choose your time frame:', time_frames,key="54567567567")
         with right_column_2:
                  framework = st.selectbox('Choose your framework:', frameworks,key="43534654567567")
                  value_media_spend = st.selectbox('Media spend', values,key="8273985756")

         #filter
         df_filtered =  df[(df["Category"] == category) & (df["time_period_x"] == time_frame)]
         df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]

         # layout stuff
         if time_frame == "months":
                  df_filtered['time'] = df_filtered['time'].dt.to_period('M').dt.to_timestamp()
         
         if time_frame == "quarters":
                  df_filtered['time'] = df_filtered['time'].dt.to_period('Q').dt.to_timestamp()
         
         if time_frame == "years":
                  df_filtered['time'] = df_filtered['time'].dt.to_period('Y').dt.to_timestamp()
         
         if time_frame == "semiannual":
                  df_filtered['time'] = df_filtered['time'].dt.to_period('Q').dt.start_time + pd.DateOffset(months=6)



         
         df_filtered = df_filtered.sort_values(by="time")
         all_brands = [x for x in df["brand"].unique()]
         brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
         
         line_plot = px.line(df_filtered, x="time", y=framework, color="brand",color_discrete_map=brand_color_mapping)
         histogram = px.histogram(df_filtered,x="time",y=value_media_spend,color="brand",color_discrete_map=brand_color_mapping,nbins=600)
         
         
         line_plot.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}')
         histogram.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}')

         
         if time_frame == "months":
            # Extract unique months from the "time" column
            unique_months = df_filtered['time'].dt.to_period('M').unique()
            
            # Customize the x-axis tick labels to show one label per month
            tickvals = [f"{m.start_time}" for m in unique_months]
            ticktext = [m.strftime("%B %Y") for m in unique_months]
            
            # Create subplots with separate figures
            sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
            
            # Add line plot to the first subplot
            for trace in line_plot.data:
                sub_fig.add_trace(trace, row=1, col=1)
            
            # Add histogram to the second subplot
            for trace in histogram.data:
                sub_fig.add_trace(trace, row=2, col=1)
            
            sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=1, col=1)
            sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=2, col=1)
         
         if time_frame == "quarters":
            unique_quarters = df_filtered['time'].dt.to_period('Q').unique()
            
            # Customize the x-axis tick labels to show one label per quarter
            tickvals = [f"{q.start_time}" for q in unique_quarters]
            ticktext = [f"Q{q.quarter} {q.year}" for q in unique_quarters]
            
            # Create subplots with separate figures
            sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
            
            # Add line plot to the first subplot
            for trace in line_plot.data:
                sub_fig.add_trace(trace, row=1, col=1)
            
            # Add histogram to the second subplot
            for trace in histogram.data:
                sub_fig.add_trace(trace, row=2, col=1)
            
            sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=1, col=1)
            sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=2, col=1)
         
         if time_frame =="years":
            # Extract unique years from the "time" column
            unique_years = df_filtered['time'].dt.year.unique()
            
            # Create subplots with separate figures
            sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
            
            # Add line plot to the first subplot
            for trace in line_plot.data:
                    sub_fig.add_trace(trace, row=1, col=1)
            
            # Add histogram to the second subplot
            for trace in histogram.data:
                    sub_fig.add_trace(trace, row=2, col=1)
            
            sub_fig.update_xaxes(tickvals=[f"{year}-01-01" for year in unique_years], ticktext=unique_years, tickangle=45, row=1, col=1)
            sub_fig.update_xaxes(tickvals=[f"{year}-01-01" for year in unique_years], ticktext=unique_years, tickangle=45, row=2, col=1)
         
         if time_frame== "weeks":
            # Extract unique weeks from the "time" column
            unique_weeks = pd.date_range(start=ws, end=we, freq='W').date
            
            # Customize the x-axis tick labels to show the start date of each week
            tickvals = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]
            ticktext = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]
            
            # Create subplots with separate figures
            sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
            
            # Add line plot to the first subplot
            for trace in line_plot.data:
                sub_fig.add_trace(trace, row=1, col=1)
            
            # Add histogram to the second subplot
            for trace in histogram.data:
                sub_fig.add_trace(trace, row=2, col=1)
            
            sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=1, col=1)
            sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=2, col=1)
         
         if time_frame == "semiannual":
            # Extract unique semiannual periods from the "time" column
            unique_periods = pd.date_range(start=ws, end=we, freq='6M').date
            
            # Customize the x-axis tick labels to show the start date of each semiannual period
            tickvals = [period.strftime('%Y-%m-%d') for period in unique_periods]
            ticktext = [f"Semiannual {i} - {period.strftime('%Y')}" for i, period in enumerate(unique_periods)]
            
            sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
            
            # Add line plot to the first subplot
            for trace in line_plot.data:
                sub_fig.add_trace(trace, row=1, col=1)
            
            # Add histogram to the second subplot
            for trace in histogram.data:
                sub_fig.add_trace(trace, row=2, col=1)
            
            sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=1, col=1)
            sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=2, col=1)
         
         return sub_fig


# Sub-plots for comparing the weighted vs the unweighted
def sub_plots_w(df,df_weighted,categories,time_frames,frameworks):
         # getting the columns
         left_column_1, left_column_2,right_column_1,right_column_2 = st.columns(4)

         with left_column_1:
                  #getting the date
                  start_date = st.date_input("Select start date",key="20",value=datetime(2020, 1, 1))
                  end_date =  st.date_input("Select end date",key="21")
                  #convert our dates
                  ws = start_date.strftime('%Y-%m-%d')
                  we = end_date.strftime('%Y-%m-%d')
         
         with left_column_2:
                  # getting the parameters
                  category = st.radio('Choose your category:', categories,key="22")
         with right_column_1:
                  time_frame = st.radio('Choose your time frame:', time_frames,key="23")
         with right_column_2:
                  framework = st.selectbox('Choose your framework:', frameworks,key="24")
         
         #filter df 
         df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
         df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
         df_filtered = df_filtered.sort_values(by="time")

         #filter df_weighted
         df_filtered_w =  df_weighted[(df_weighted["Category"] == category) & (df_weighted["time_period"] == time_frame)]
         df_filtered_w = df_filtered_w[(df_filtered_w['time'] >= ws) & (df_filtered_w['time'] <= we)]
         df_filtered_w = df_filtered_w.sort_values(by="time")

         
         all_brands = [x for x in df["brand"].unique()]
         
         brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
         
         line_plot = px.line(df_filtered, x="time", y=framework,color="brand", color_discrete_map=brand_color_mapping)
         line_plot_w = px.line(df_filtered_w,x="time",y=framework,color="brand",color_discrete_map=brand_color_mapping)

         
         line_plot.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}')

         # Update custom hover template for histogram (scatter plot)
         line_plot_w.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}')

         
         if time_frame == "months":
                  # Extract unique months from the "time" column
                  unique_months = df_filtered['time'].dt.to_period('M').unique()
                  
                  # Customize the x-axis tick labels to show one label per month
                  tickvals = [f"{m.start_time}" for m in unique_months]
                  ticktext = [m.strftime("%B %Y") for m in unique_months]

                   # Create subplots with separate figures
                  sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                  
                  # Add line plot to the first subplot
                  for trace in line_plot.data:
                     sub_fig.add_trace(trace, row=1, col=1)
                  
                  # Add histogram to the second subplot
                  for trace in line_plot_w.data:
                     sub_fig.add_trace(trace, row=2, col=1)

                  
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=1, col=1)
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=2, col=1)
                  sub_fig.update_yaxes(title_text="Unweighted Plot",title_font=dict(color="blue"), row=1, col=1)
                  sub_fig.update_yaxes(title_text="Weighted Plol",title_font=dict(color="red"), row=2, col=1)
                  return sub_fig

         
         if time_frame == "quarters":
                  unique_quarters = df_filtered['time'].dt.to_period('Q').unique()
                  
                  # Customize the x-axis tick labels to show one label per quarter
                  tickvals = [f"{q.start_time}" for q in unique_quarters]
                  ticktext = [f"Q{q.quarter} {q.year}" for q in unique_quarters]
                  
                  # Create subplots with separate figures
                  sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                  
                  # Add line plot to the first subplot
                  for trace in line_plot.data:
                     sub_fig.add_trace(trace, row=1, col=1)
                  
                  # Add histogram to the second subplot
                  for trace in line_plot_w.data:
                     sub_fig.add_trace(trace, row=2, col=1)
                  
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=1, col=1)
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=2, col=1)
                  sub_fig.update_yaxes(title_text="Unweighted Plot",title_font=dict(color="blue"), row=1, col=1)
                  sub_fig.update_yaxes(title_text="Weighted Plol",title_font=dict(color="red"), row=2, col=1)
                  return sub_fig

         if time_frame == "years":
                  # Extract unique years from the "time" column
                  unique_years = df_filtered['time'].dt.year.unique()
                  
                  # Customize the x-axis tick labels to show one label per year
                  tickvals = [f"{year}-01-01" for year in unique_years]
                  ticktext = unique_years
                  
                  # Create subplots with separate figures
                  sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                  
                  # Add line plot to the first subplot
                  for trace in line_plot.data:
                      sub_fig.add_trace(trace, row=1, col=1)
                  
                  # Add histogram to the second subplot
                  for trace in line_plot_w.data:
                      sub_fig.add_trace(trace, row=2, col=1)
                  
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=1, col=1)
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=2, col=1)
                  sub_fig.update_yaxes(title_text="Unweighted Plot", title_font=dict(color="blue"), row=1, col=1)
                  sub_fig.update_yaxes(title_text="Weighted Plot", title_font=dict(color="red"), row=2, col=1)
                  
                  return sub_fig
                  
         if time_frame == "weeks":
                   # Extract unique weeks from the "time" column
                  unique_weeks = pd.date_range(start=ws, end=we, freq='W').date
                  
                  # Customize the x-axis tick labels to show the start date of each week
                  tickvals = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]
                  ticktext = [week.strftime('%Y-%m-%d') for i, week in enumerate(unique_weeks) if i % 4 == 0]
                  
                  # Create subplots with separate figures
                  sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                  
                  # Add line plot to the first subplot
                  for trace in line_plot.data:
                     sub_fig.add_trace(trace, row=1, col=1)
                  
                  # Add histogram to the second subplot
                  for trace in line_plot_w.data:
                     sub_fig.add_trace(trace, row=2, col=1)
                  
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=1, col=1)
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=2, col=1)
                  sub_fig.update_yaxes(title_text="Unweighted Plot",title_font=dict(color="blue"), row=1, col=1)
                  sub_fig.update_yaxes(title_text="Weighted Plol",title_font=dict(color="red"), row=2, col=1)
                  return sub_fig


         if time_frame =="semiannual":
                   # Extract unique semiannual periods from the "time" column
                  unique_periods = pd.date_range(start=ws, end=we, freq='6M').date
                  
                  # Customize the x-axis tick labels to show the start date of each semiannual period
                  tickvals = [period.strftime('%Y-%m-%d') for period in unique_periods]
                  ticktext = [f"Semiannual {i} - {period.strftime('%Y')}" for i, period in enumerate(unique_periods)]
                  
                  # Create subplots with separate figures
                  sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                  
                  # Add line plot to the first subplot
                  for trace in line_plot.data:
                     sub_fig.add_trace(trace, row=1, col=1)
                  
                  # Add histogram to the second subplot
                  for trace in line_plot_w.data:
                     sub_fig.add_trace(trace, row=2, col=1)
                  
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=1, col=1)
                  sub_fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, row=2, col=1)
                  sub_fig.update_yaxes(title_text="Unweighted Plot",title_font=dict(color="blue"), row=1, col=1)
                  sub_fig.update_yaxes(title_text="Weighted Plol",title_font=dict(color="red"), row=2, col=1)
                  return sub_fig







                  
# Significance Plot
def Significance_plot(df,brands,frameworks):
         # getting the columns for the layouts
         left_column_1, right_column_1,right_column_2 = st.columns(3)
         
         
         
         
         with left_column_1:
                  #getting the date
                  start_date = st.date_input("Select start date",key="15",value=datetime(2020, 1, 1))
                  end_date =  st.date_input("Select end date",key="16")
                  #convert our dates
                  ws = start_date.strftime('%Y-%m-%d')
                  we = end_date.strftime('%Y-%m-%d')

         with right_column_1:
                  brand = st.radio('Choose your brand:', brands,key="17")
         with right_column_2:
                  framework = st.selectbox('Choose your framework:', frameworks,key="18")
         
         filtered_data = df[(df["brand"] == brand)]
         filtered_data = filtered_data[(filtered_data['time'] >= ws) & (filtered_data['time'] <= we)]
         
         # Create the main line trace
         main_trace = go.Scatter(x=filtered_data["time"], y=filtered_data[framework], mode='lines', name='Main Line')
         
         #getting the lower and upper boundaries
         lower,upper = calculate_confidence_intervals(filtered_data[framework])
         
         # Create the shaded confidence interval area
         confidence_interval_shape = {
         'type': 'rect',
         'xref': 'paper',  # Use 'paper' for the x-axis reference
         'yref': 'y',
         'x0': 0,  # Start at the left edge of the plot
         'y0': lower,
         'x1': 1,  # End at the right edge of the plot
         'y1': upper,
         'fillcolor': 'rgba(255, 0, 0, 0.2)',
         'line': {'width': 0},
         'opacity': 1,
         'layer': 'above'
         }
         
         # Create the figure and add the traces and shape
         fig = go.Figure(data=[main_trace])
         fig.add_shape(confidence_interval_shape)

         # Extract unique quarters from the "time" column
         unique_quarters = filtered_data['time'].dt.to_period('Q').unique()
         
         # Customize the x-axis tick labels to show one label per quarter
         tickvals = [f"{q.start_time}" for q in unique_quarters]
         ticktext = [f"Q{q.quarter} {q.year}" for q in unique_quarters]
         
         # Update x-axis ticks
         fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)
         
         fig.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}')

         return fig



# Correlation Plot
def correlation_plot(df,brands):

    brand = st.radio('Choose your brand:', brands,key="500")
    
    framework_options_corr = ["Volume_share","AF_Value_for_Money", "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity","Price_change"]
    df_filtered =  df[(df["brand"] == brand)]
    filtered_data_final = df_filtered[framework_options_corr]
                    
    spearman_corr = filtered_data_final.corr(method='spearman')

    custom_colors = ["red","grey","green"]

    fig_spearman = px.imshow(
    spearman_corr.values,
    x=spearman_corr.columns,
    y=spearman_corr.columns,
    color_continuous_scale=custom_colors,
    labels ={"Value": "Custom Scale"},
    title="Spearman's Correlation Heatmap")

    
     # Update the layout with labels and title
    fig_spearman.update_layout(
    title=f'spearman correlation map {brand}',
    height=700,  # Set the height of the figure
    width=700)
    
    return fig_spearman


#------------------------------------------------------------------------app------------------------------------------------------------------------------#
def main():
         
         #logout_container = st.container()
         #st.title("Streamlit App with Microsoft SSO")
         #Global variables
         # Initialize session state variables


         # Initialize session state variables
         #if 'access' not in st.session_state:
         #         st.session_state.access = False
                  
         #if 'login_clicked' not in st.session_state:
         #         st.session_state.login_clicked = False

         #if not st.session_state.access:                  
                                    #login()
                                    # Check for authorization code in URL
                                    #params = st.experimental_get_query_params()
                                    #if "code" in params:
                                             #code = params["code"][0]
                                             #token = get_token_from_code(code)
                                             #if token:
                                                      #st.session_state.access_token = token
                                                      #st.experimental_set_query_params()
                                             
         with st.container():
                  #None Global

                  # user input for equity and mmm file. 
                  markets_available = ["uk"]
                  market = st.selectbox('', markets_available)
                  
                  if market == "germany":
                           slang = "MMM_DE_"
                           res_weighted = None
                           mmm = "Yes"
                           sheet_week = "WeekMap_DE"
                           sheet_spend = "DE_Raw Spend"
         
                  
                  if market == "uk":
                           slang ="MMM_UK_"
                           #res_weighted = None
                           #market_weighted = "uk_equity_age_weighted"
                           mmm = None
                           sheet_week = "WeekMap_UK"
                           sheet_spend = "UK_Raw Spend"
                      
                  if market =="italy":
                           slang ="MMM_IT"
                           res_weighted = None
                           mmm = "Yes"
                           sheet_week = "WeekMap_IT"
                           sheet_spend = "IT_Raw Spend"
         
                      
                  if market =="france":
                           slang = "MMM_FR"
                           res_weighted = None
                           mmm =None
         
         
                  # getting our equity    
                  #if res_weighted == "Yes":
                  #   filepath_equity,year_equity,month_equity,day_equity,hour_equity,minute_equity,second_equity = equity_info(data,market)
                  #   filepath_equity_weighted,year_equity_w,month_equity_w,day_equity_w,hour_equity_w,minute_equity_w,second_equity_w = equity_info(data,market_weighted)
                  #else:
                  filepath_equity,year_equity,month_equity,day_equity,hour_equity,minute_equity,second_equity = equity_info(data,market)
                  
                  if mmm == None:
                     pass 
                  else:
                     filepath_mmm,year_mmm,month_mmm,day_mmm,hour_mmm,minute_mmm,second_mmm = mmm_info(data,slang)
                  
                  #if mmm == None:
                  #         st.write(f"""**Equity file version** {market} : {day_equity}/{month_equity}/{year_equity} - {hour_equity}: {minute_equity}: {second_equity}""")
                     
                  #else:
                     #if res_weighted == "Yes":
                     #    st.write(f"""**Equity file version** {market} : {day_equity}/{month_equity}/{year_equity} - {hour_equity}: {minute_equity}: {second_equity} **Age weighted equity file version** {market_weighted}: {day_equity_w}/{month_equity_w}/{year_equity_w} - {hour_equity_w}: {minute_equity_w}: {second_equity_w} 
                     #    **MMM data version** {market} : {day_mmm}/{month_mmm}/{year_mmm} - {hour_mmm}: {minute_mmm}: {second_mmm}""")
                     #if res_weighted == None or mmm == None:
                         #st.write(f"""**Equity file version** {market}: {day_equity}/{month_equity}/{year_equity} - {hour_equity}: {minute_equity}: {second_equity} **MMM data version** {market} : {day_mmm}/{month_mmm}/{year_mmm}- {hour_mmm} - {minute_mmm}: {second_mmm}""")
         
         
                  # reading the equity file
                  #if res_weighted == "Yes":
                  #   df = reading_df(filepath_equity)
                  #   df_weighted = reading_df(filepath_equity_weighted)
                  #else:
                  df = reading_df(filepath_equity,sheet_name="average_smoothened")
                  df_total_uns = reading_df(filepath_equity,sheet_name="total_unsmoothened")

                  
                  # reading and processing the mmm file
                  if mmm == None:
                     pass
                  else:
                     df_vol = processing_mmm(filepath_mmm)
                  
                  #creating the merged df 
                  if mmm == None:
                     pass
                  else:
                     #if res_weighted == "Yes":
                     #    merged_df = merged_file(df,df_vol)
                     #    merged_df_weighted = merged_file(df_weighted,df_vol)
                     #else:
                           merged_df = merged_file(df,df_vol)
                  
                   # creating the Media merged_df with options ! 
                  if mmm == None:
                           pass
                  else:
                           #if res_weighted == "Yes":
                           #         df_uk_spend,df_uk_weeks = media_plan(media_data,sheet_spend,sheet_week)
                           #         merged_df_media_weighted = media_spend_processed(df_weighted,df_uk_spend,df_uk_weeks)
                           #         category_options_merged_media_w,time_period_options_merged_media_w,framework_options_media_w, framework_options_value_media_w= merged_options_media(merged_df_media_weighted)
                           
                            #        df_uk_spend,df_uk_weeks = media_plan(media_data,sheet_spend,sheet_week)
                            #        merged_df_media = media_spend_processed(df,df_uk_spend,df_uk_weeks)
                            #        category_options_merged_media,time_period_options_merged_media,framework_options_media, framework_options_value_media= merged_options_media(merged_df_media)
                  
                           #else:
                           df_uk_spend,df_uk_weeks = media_plan(media_data,sheet_spend,sheet_week)
                           merged_df_media = media_spend_processed(df,df_uk_spend,df_uk_weeks)
                           category_options_merged_media,time_period_options_merged_media,framework_options_media, framework_options_value_media= merged_options_media(merged_df_media)
         
                  
                  #Equity options
                  #if res_weighted == "Yes":
                  #   category_options,time_period_options,framework_options = equity_options(df)
                  #   category_options_w,time_period_options_w,framework_options_w = equity_options(df_weighted)
                    
                  
                  #else:
                  category_options,time_period_options,framework_options = equity_options(df)
                  
                  
                  # Volume share options
                  if mmm== None:
                     pass
                  else:
                     category_options_vol_share = df_vol["Category"].unique()
                  
                  #Merged options
                  if mmm== None:
                     pass
                  else:
                     category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value = merged_options(merged_df)
                  
                  # Significance options
                  if mmm== None:
                     pass
                  else:
                     Brand_options = merged_df["brand"].unique()
                     framework_options_sig = ["Volume_share","AF_Value_for_Money", "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity","Price_change"]
                     lower,upper = calculate_confidence_intervals(merged_df["Framework_Awareness"])
                  
                  # Correlation options
                  if mmm== None:
                     pass
                  else:
                     Brand_options = merged_df["brand"].unique()
                     framework_options_corr = ["Volume_share","AF_Value_for_Money", "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity","Price_change"]


                  
                 
                  
                  #creating the weighted file
                  weighted_avg = st.number_input("average weight", min_value=0.0, max_value=1.0, value=0.75, step=0.5, key="weighted_avg")
                  weighted_total = 1 - weighted_avg
                  df_weighted = get_weighted(df,df_total_uns,weighted_avg,weighted_total)
                  # Comparing all the sheets
                  fig = Comparing_Equity(df,df_total_uns,df_weighted,category_options,time_period_options,framework_options)
                  st.plotly_chart(fig,use_container_width=True)

                  
                  #Equity plot
                  #if res_weighted == "Yes":
                  #   res_equity_weighted = st.radio("What type do you want to see?", ["Unweighted","Weighted"])
                  #   if res_equity_weighted == "Weighted":
                  #       fig = Equity_plot(df_weighted,category_options,time_period_options,framework_options)
                  #       st.plotly_chart(fig,use_container_width=True)
                  #   else:
                  #         fig = Equity_plot(df,category_options,time_period_options,framework_options)
                  #       st.plotly_chart(fig,use_container_width=True)
                  #else: 
                  


                   #chosing the sheet name 
                  sheet_name = st.selectbox("Select you sheet",["Average","Absolute", "Weighted"])
                  
                  if sheet_name == "Average":
                           sheet_name = "Average Smoothening"
                  if sheet_name == "Absolute":
                           sheet_name = "Total Unsmoothening"
                  
                  if sheet_name == "Average Smoothening":
                           fig = Equity_plot(df,category_options,time_period_options,framework_options,sheet_name=sheet_name)
                           st.plotly_chart(fig,use_container_width=True)
                  
                  if sheet_name == "Total Unsmoothening":
                           fig = Equity_plot(df_total_uns,category_options,time_period_options,framework_options,sheet_name=sheet_name)
                           st.plotly_chart(fig,use_container_width=True)
                  
                  if sheet_name == "Weighted":
                           fig = Equity_plot(df_weighted,category_options,time_period_options,framework_options,sheet_name=sheet_name)
                           st.plotly_chart(fig,use_container_width=True)


                  st.subheader(f"Equity Metrics Plot - Market Share Weighted Average")
                  col1,col2,col3 = st.columns([4,4,8])
                  # creating the average_weighted 
                  weights_values_for_average = {"aptamil":0 , "cow&gate": 0, "sma": 0, "kendamil": 0, "hipp_organic": 0}
                  with col1:
                     for x in list(weights_values_for_average.keys())[:3]:
                         number = st.number_input(f"Weight for the {x}", min_value=0, max_value=100, value=10)
                         number = number/100
                         weights_values_for_average[x]=number
                  
                  with col2:
                     for x in list(weights_values_for_average.keys())[3:5]:
                         number = st.number_input(f"Weight for the {x}", min_value=0, max_value=100, value=10)
                         number = number/100
                         weights_values_for_average[x]=number
                  
                  #creating the market_share_weighted
                  value_columns  = [ 'AA_eSoV', 'AA_Reach',
                  'AA_Brand_Breadth', 'AS_Average_Engagement', 'AS_Usage_SoV',
                  'AS_Search_Index', 'AS_Brand_Centrality', 'AF_Entry_point',
                  'AF_Brand_Love', 'AF_Baby_Milk', 'AF_Adverts_Promo',
                  'AF_Value_for_Money', 'AF_Buying_Exp', 'AF_Prep_Milk', 'AF_Baby_exp',
                  'Framework_Awareness', 'Framework_Saliency', 'Framework_Affinity',
                  'Total_Equity']
                  market_share_weighted =  weighted_brand_calculation(df, weights_values_for_average, value_columns)
                  sheet_name = "Market Share Weighted Average"
                  fig = Equity_plot_market_share_(market_share_weighted,category_options,time_period_options,value_columns,sheet_name=sheet_name)
                  st.plotly_chart(fig,use_container_width=True)












                  
                  # Comparing the weighted vs the unweighted
                  if mmm == None:
                     pass
                  else:
                     st.subheader("Weighted vs Unweighted")
                     if res_weighted == "Yes":
                         fig_weigheted_vs_un = sub_plots_w(df,df_weighted,category_options,time_period_options,framework_options)
                         st.plotly_chart(fig_weigheted_vs_un,use_container_width=True)
                  
                  #Market share Plot 
                  if mmm ==None:
                     pass
                  else:
                     st.subheader("Agreggated Volume Share by Brand Plot")
                     fig_market_share = market_share_plot(df_vol,category_options_vol_share)
                     st.plotly_chart(fig_market_share,use_container_width=True)
                  
                  #Buble plot
                  if mmm ==None:
                     pass
                  else:
                     st.subheader("Equity vs Volume Share (Bubble plot)")
                     if res_weighted == "Yes":
                         res_equity_weighted = st.radio("What type do you want to see?", ["Unweighted","Weighted"],key="44")
                         if res_equity_weighted == "Weighted":
                             fig_buble= buble_plot(merged_df_weighted,category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value)
                             st.plotly_chart(fig_buble,use_container_width=True)
                         else:
                             fig_buble= buble_plot(merged_df,category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value)
                             st.plotly_chart(fig_buble,use_container_width=True)
                     else:
                         fig_buble= buble_plot(merged_df,category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value)
                         st.plotly_chart(fig_buble,use_container_width=True)
                  
                  #Sub_plots
                  if mmm ==None:
                     pass
                  else:
                     st.subheader("Equity vs Volume Share  (histogram)")
                     if res_weighted == "Yes":
                         res_equity_weighted = st.radio("What type do you want to see?", ["Unweighted","Weighted"],key="45")
                         if res_equity_weighted == "Weighted":
                             fig_sub = sub_plots(merged_df_weighted,category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value)
                             st.plotly_chart(fig_sub,use_container_width=True)
                         else:
                             fig_sub = sub_plots(merged_df,category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value)
                             st.plotly_chart(fig_sub,use_container_width=True)
                     else:
                         fig_sub = sub_plots(merged_df,category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value)
                         st.plotly_chart(fig_sub,use_container_width=True)
                  
                  # Media_spend sub-plot. 
                  if mmm== None:
                     pass
                  else:
                     if res_weighted == "Yes":
                         res_equity_weighted = st.radio("What type do you want to see?", ["Unweighted","Weighted"],key="00")
                         if res_equity_weighted == "Weighted":
                             fig_media = sub_plots_media(merged_df_media_weighted,category_options_merged_media_w,time_period_options_merged_media_w,framework_options_media_w, framework_options_value_media_w)
                             st.plotly_chart(fig_media,use_container_width=True)
                         else:
                             fig_media = sub_plots_media(merged_df_media,category_options_merged_media,time_period_options_merged_media,framework_options_media, framework_options_value_media)
                             st.plotly_chart(fig_media,use_container_width=True)
                     else:
                         fig_media = sub_plots_media(merged_df_media,category_options_merged_media,time_period_options_merged_media,framework_options_media, framework_options_value_media)
                         st.plotly_chart(fig_media,use_container_width=True)
                  
                  
                  # Significance Plot
                  if mmm ==None:
                     pass
                  else:
                     st.subheader("Equity Plot w/ Significance (90% confidence interval)")
                     if res_weighted == "Yes":
                         res_equity_weighted = st.radio("What type do you want to see?", ["Unweighted","Weighted"],key="50")
                         if res_equity_weighted == "Weighted":
                             fig_significance = Significance_plot(merged_df_weighted, Brand_options,framework_options_sig)
                             st.plotly_chart(fig_significance,use_container_width=True)
                         else:
                             fig_significance = Significance_plot(merged_df, Brand_options,framework_options_sig)
                             st.plotly_chart(fig_significance,use_container_width=True)
                     
                     else:
                         fig_significance = Significance_plot(merged_df, Brand_options,framework_options_sig)
                         st.plotly_chart(fig_significance,use_container_width=True)
                  
                  # Correlation Plot
                  if mmm ==None:
                     pass
                  else:
                     st.subheader("Correlation Plot between Equity Metrics and Aggregated Sales Volume ")
                     if res_weighted == "Yes":
                         res_equity_weighted = st.radio("What type do you want to see?", ["Unweighted","Weighted"],key="47")
                         if res_equity_weighted == "Weighted":
                             fig_corr = correlation_plot(merged_df_weighted,Brand_options)
                             st.plotly_chart(fig_corr,use_container_width=True)
                         else:
                             fig_corr = correlation_plot(merged_df,Brand_options)
                             st.plotly_chart(fig_corr,use_container_width=True) 
                     else:   
                         fig_corr = correlation_plot(merged_df,Brand_options)
                         st.plotly_chart(fig_corr,use_container_width=True)
         
                  
if __name__=="__main__":
    main()   













