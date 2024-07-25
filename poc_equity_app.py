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
#--------------------------------------------------------------------------------------// Aesthetic Global Variables // -------------------------------------------------------------------------

user_to_equity = {'Entry points & Key Moments':'AF_Entry_point','Brand Prestige & Love':'AF_Brand_Love','Baby Milk':'AF_Baby_Milk','Adverts and Promotions':'AF_Adverts_Promo','Value For Money':'AF_Value_for_Money',
                        'Buying Experience': 'AF_Buying_Exp','Preparing Milk':'AF_Prep_Milk','Baby Experience':'AF_Baby_exp','Total Equity':'Total_Equity',"Awareness":'Framework_Awareness','Saliency':'Framework_Saliency','Affinity':'Framework_Affinity','eSoV':'AA_eSoV', 'Reach':'AA_Reach',
       'Brand Breadth': 'AA_Brand_Breadth', 'Average Engagement':'AS_Average_Engagement', 'Usage SoV':'AS_Usage_SoV',
       'Search Index': 'AS_Search_Index', 'Brand Centrality':'AS_Brand_Centrality'}

affinity_labels = ['AF_Entry_point','AF_Brand_Love','AF_Baby_Milk','AF_Adverts_Promo','AF_Value_for_Money','AF_Buying_Exp','AF_Prep_Milk','AF_Baby_exp']



framework_to_user = {'Total_Equity':'Total Equity','Framework_Awareness':"Awareness",'Framework_Saliency':'Saliency','Framework_Affinity':'Affinity','AA_eSoV':'eSoV', 'AA_Reach':'Reach',
       'AA_Brand_Breadth':'Brand Breadth', 'AS_Average_Engagement':'Average Engagement', 'AS_Usage_SoV':'Usage SoV',
       'AS_Search_Index':'Search Index', 'AS_Brand_Centrality':'Brand Centrality'}


original_category = "baby_milk" 
changed_category = "Baby Milk" 


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
    return df


@st.cache_data()
def get_weighted(df,df_total_uns,weighted_avg,weighted_total,brand_replacement,user_to_equity,affinity_labels):
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
    if market_flag == "UK":
        market_flag = "UK_equity_"
    for x in os.listdir(data):
        if market_flag in x:
            filepath_equity = os.path.join(data,x)
            info_number = [x for x in x.split("_") if x >= "0" and x <="9"]
            year_equity,month_equity,day_equity,hour_equity,minute_equity = info_number[:5]
            second_equity = info_number[-1].split(".")[0]
    
    return filepath_equity,year_equity,month_equity,day_equity,hour_equity,minute_equity,second_equity

def equity_options(df,brand_mapping,original_category,changed_category,framework_options_):
         df.brand = df.brand.replace(brand_mapping)
         
         
         df["Category"] = df["Category"].replace(original_category,changed_category)
         category_options = df["Category"].unique()
         
         replacements = {"weeks":"Weeks","months":"Months","quarters":"Quarters","semiannual":"Semiannual","years":"Years"}
         df["time_period"] = df["time_period"].replace(replacements)
         time_period_options = df["time_period"].unique()
         
         framework_options = framework_options_
         
         return (category_options,time_period_options,framework_options)
         
#-----------------------------------------------------------------------------------------------------//-----------------------------------------------------------------------------------------
# Equity_plot
def Equity_plot(df,categories,time_frames,frameworks,sheet_name,framework_to_user):
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
        category = st.radio('Choose  category:', categories)
        
    with left_column_1:    
        time_frame = st.radio('Choose  time frame:', time_frames)
    
    with left_column_2:
        framework = st.selectbox('Choose  metric:', frameworks)
    
    #filtering
    df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    
    df_filtered = df_filtered.sort_values(by="time")
    
    
    # color stuff
    all_brands = [x for x in df["brand"].unique()]
    colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","yellow","olive","silver","darkviolet","grey"]

    brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
    
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

def Equity_plot_market_share_(df,category,time_frame,framework,ws,we):
   
    #filtering
    df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    
    df_filtered = df_filtered.sort_values(by="time")
    
    
    # color stuff
    all_brands = [x for x in df["brand"].unique()]
    colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","yellow","olive","silver","darkviolet","grey"]

    brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
    
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
def Comparing_Equity(df,df_total_uns,weighted_df,categories,time_frames,frameworks,brand_replacement,affinity_to_user,original_category,changed_category,general_equity_to_user):
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


    df_total_uns["Category"] = df_total_uns["Category"].replace(original_category,changed_category)

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
        category = st.radio('Choose category:', categories,key="test_3")
        
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
                           
                                    
#---------------------------------------------------------------------------------------------------------//General info// ------------------------------------------------------------------------------------- 
                           #with st.container():
                                    #tab2,tab3,tab4 = st.tabs(["📈 Market Share Weighted","🔍Compare Average, Absolute and Market Share Weighted","📕 Final Equity plots"])
                           
                           #with st.sidebar:
                                    #st.image(image)
         
                                    # user input for equity and mmm file. 
                                    markets_available = ["UK"]
                                    #column_1,_ = st.columns(2)
                     
                                    #with column_1:
                                             #market = st.selectbox('Markets', markets_available)
                                             #market = market.lower()
                                             
                                    #if market == "uk":
                                             #slang ="MMM_UK_"
                                             #brand_mapping = {"aptamil":"APTAMIL" , "cow&gate": "COW & GATE", "sma": "SMA", "kendamil": "KENDAMIL", "hipp_organic": "HIPP ORGANIC"}
                                             #weights_values_for_average_2021 = {"APTAMIL":0 , "COW & GATE": 0, "SMA": 0, "KENDAMIL": 0, "HIPP ORGANIC": 0}
                                             #weights_values_for_average_2022 = {"APTAMIL":0 , "COW & GATE": 0, "SMA": 0, "KENDAMIL": 0, "HIPP ORGANIC": 0}
                                             #weights_values_for_average_2023 = {"APTAMIL":0 , "COW & GATE": 0, "SMA": 0, "KENDAMIL": 0, "HIPP ORGANIC": 0}
                                             #weights_values_for_average_2024 = {"APTAMIL":0 , "COW & GATE": 0, "SMA": 0, "KENDAMIL": 0, "HIPP ORGANIC": 0}
                           
                                    # getting our equity    
                                    #filepath_equity,year_equity,month_equity,day_equity,hour_equity,minute_equity,second_equity = equity_info(data,market)
                                    
                           
                                    # reading the equity file
                                    #df = reading_df(filepath_equity,sheet_name="average_smoothened")
                                    #df_total_uns = reading_df(filepath_equity,sheet_name="total_unsmoothened")
                                    #df_total_smooth = reading_df(filepath_equity,sheet_name="total_smoothened")
                                    #df_avg_unsmooth = reading_df(filepath_equity,sheet_name="average_unsmoothened")
                                    #df_significance = reading_df(filepath_equity,sheet_name="significance")
                                    #df_perc_changes = reading_df(filepath_equity,sheet_name="perc_changes")
                           
                                    
                                    #Equity options
                                    #category_options,time_period_options,framework_options = equity_options(df,brand_mapping,original_category,changed_category,framework_options_)
                                    
                                      #creating the market_share_weighted
                                    #value_columns  = value_columns_
                  
#--------------------------------------------------------------------------------------// transformations ----------------------------------------------------------------------------------
                                    #creating a copy of our dataframes.
                                    #df_copy = df.copy()
                                    #df_total_uns_copy = df_total_uns.copy()
                                    # Aesthetic changes --------------------------------------------------------------------------------------------------
                                    #changing the names of the filtered  columns
                                    ################################################################## df ####################################################################################################
                                    #df_copy.rename(columns=affinity_to_user,inplace=True)
                                    
                                    
                                    
                                    #df_copy.brand = df_copy.brand.replace(brand_mapping)
                                    
                                    #df_copy.rename(columns=general_equity_to_user,inplace=True)
                                    
                                    ################################################################## df_total_uns ####################################################################################################
                                    
                                    #df_total_uns_copy.rename(columns=affinity_to_user,inplace=True)
                                    
                                    
                                    #df_total_uns_copy.brand = df_total_uns_copy.brand.replace(brand_mapping)
                                    
                                    #replacements = {"weeks":"Weeks","months":"Months","quarters":"Quarters","semiannual":"Semiannual","years":"Years"}
                                    #df_total_uns_copy["time_period"] = df_total_uns_copy["time_period"].replace(replacements)
                                    
                                    
                                    #df_total_uns_copy["Category"] = df_total_uns_copy["Category"].replace("baby_milk","Baby milk")
                                    
                                    
                                    #df_total_uns_copy.rename(columns=general_equity_to_user,inplace=True)

################################################################## ##################################################################################################################
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------// Market Share Weighted----------------------------------------------------------------------------------
                           #with tab2:
                                    #chosing the sheet name 
                                    #column_1,_,_,_ = st.columns(4)
                                    #with column_1:
                                     #        sheet_name = st.selectbox("Select sheet",["Average","Absolute"])
                                    
                                    #st.subheader(f"Equity Metrics Plot - Market Share weighted {sheet_name}")
                           
                           
                                    #if sheet_name == "Average":
                                       #      sheet_name = "Average Smoothening"
                                      #       sheet_name_download = 'average'
                                     #        df_for_weighted = df_copy
                                    #if sheet_name == "Absolute":
                                             #sheet_name = "Total Unsmoothening"
                                             #sheet_name_download = "total"
                                             #df_for_weighted = df_total_uns_copy
                                    
                           
                                    # getting the individual years
                                    #years_filtered = df_for_weighted[df_for_weighted.time_period == "Years"]            
                                    #years_filtered = years_filtered.time.dt.year.unique()
                                    #years_cols = [str(year) for year in years_filtered if year != 2020]
                                    
#--------------------------------------------------------------------------------------------------------------------------// //-----------------------------------------------------
                                    #weights_joined = []
                                    #join_brand_year= []
                                    #keys = [ key for key in brand_mapping.keys()]
                           
                                    #for year in years_cols:
                                     #        join_brand_year.append((year,keys))
                           
                                    # Assuming you want one column per key in brand_mapping
                                    #num_columns = len(join_brand_year)
                                    #num_brand = len(brand_mapping.keys())
                                       
                                    #colunas
                                    #cols = st.columns(num_columns)
                           
                                    #for index,col in zip(range(num_columns),cols[:]):
                                           #workspace = join_brand_year[index]
                                           #year,brand = workspace[0],workspace[1]
                                           #st.write(year)
                                           # Assuming you want one column per key in brand_mapping
                                           #num_columns = len(brand_mapping.keys())
                                           # Create the columns
                                           #cols = st.columns(num_columns)
                                           #if year == "2021":
                                               # Iterate over the columns and keys simultaneously
                                            #   for col, key in zip(cols, weights_values_for_average_2021.keys()):
                                             #              year_key = f"{key}_{year}"
                                              #             with col:
                                               #                    number = st.number_input(f"Weight for {key}", min_value=0, max_value=100, value=10,key=year_key)
                                                #                   weights_values_for_average_2021[key] = number / 100
                           
                                           #if year == "2022":
                                               # Iterate over the columns and keys simultaneously
                                            #   for col, key in zip(cols, weights_values_for_average_2022.keys()):
                                             #              year_key = f"{key}_{year}"
                                              #             with col:
                                               #                    number = st.number_input(f"Weight for {key}", min_value=0, max_value=100, value=10,key=year_key)
                                                #                   weights_values_for_average_2022[key] = number / 100
                           
                           
                                           #if year == "2023":
                                               # Iterate over the columns and keys simultaneously
                                            #   for col, key in zip(cols, weights_values_for_average_2023.keys()):
                                             #              year_key = f"{key}_{year}"
                                              #             with col:
                                               #                    number = st.number_input(f"Weight for {key}", min_value=0, max_value=100, value=10,key=year_key)
                                                #                   weights_values_for_average_2023[key] = number / 100
                           
                           
                                           #if year == "2024":
                                               # Iterate over the columns and keys simultaneously
                                            #   for col, key in zip(cols, weights_values_for_average_2024.keys()):
                                             #              year_key = f"{key}_{year}"
                                              #             with col:
                                               #                    number = st.number_input(f"Weight for {key}", min_value=0, max_value=100, value=10,key=year_key)
                                                #                   weights_values_for_average_2024[key] = number / 100
                           
                           
                           
                                    #weights_joined.append(weights_values_for_average_2021)
                                    #weights_joined.append(weights_values_for_average_2022)
                                    #weights_joined.append(weights_values_for_average_2023)
                                    #weights_joined.append(weights_values_for_average_2024)                                       
#--------------------------------------------------------------------------------------------------------------------------// //-----------------------------------------------------                   
                                    #creating the market_share_weighted
                                    #market_share_weighted =  weighted_brand_calculation(df_for_weighted, weights_joined,years_cols,value_columns,framework_to_user)
                                    
                                    # creating the columns for the app
                                    #right_column_1,right_column_2,left_column_1,left_column_2 = st.columns(4)
                                    
                                    #with right_column_1:
                                    #getting the date
                                    #         start_date = st.date_input("Select start date",value=datetime(2021, 2, 16),key='start_date')
                                    #         end_date =  st.date_input("Select end date",key='test1')
                                    # getting the parameters
                                    #with right_column_2:
                                     #        st.session_state.category = st.radio('Choose  category:', category_options,key='test3')
                                    
                                    #with left_column_1:    
                                     #        st.session_state.time_frame = st.radio('Choose  time frame:', time_period_options,key='test4')
                                    
                                    #with left_column_2:
                                     #        framework = st.selectbox('Choose  framework:', value_columns,key='test5')
                                    
                                    
                                    #if st.session_state.button == False:
                                    #         if st.button("Run!"):
                                    #                  #convert our dates
                                    #                  ws = start_date.strftime('%Y-%m-%d')
                                    #                  we = end_date.strftime('%Y-%m-%d')
                                                      
                                     #                 st.session_state.fig = Equity_plot_market_share_(market_share_weighted, st.session_state.category, st.session_state.time_frame,framework,ws,we)
                                     #                 st.session_state.button = True
                                    #else:
                                     #        if st.button("Run!"):
                                                      #convert our dates
                                      #                ws = start_date.strftime('%Y-%m-%d')
                                       #               we = end_date.strftime('%Y-%m-%d')
                                                      
                                        #              st.session_state.fig = Equity_plot_market_share_(market_share_weighted, st.session_state.category, st.session_state.time_frame,framework,ws,we)
                                             
                                    
                                    
                                    #if st.session_state.button == False:
                                    #         pass
                                    #else:
                                    #         st.plotly_chart(st.session_state.fig,use_container_width=True)
                                    
                                    
                                    #buffer = io.BytesIO()
                                    #with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                    #         df.to_excel(writer, sheet_name='average_smoothened', index=False)
                                             
                                    #         df_avg_unsmooth.to_excel(writer, sheet_name='average_unsmoothened', index=False)
                                             
                                    #         df_total_uns.to_excel(writer, sheet_name='total_unsmoothened', index=False)
                                             
                                    #         df_total_smooth.to_excel(writer, sheet_name='total_smoothened', index=False)
                                             
                                    #         market_share_weighted.to_excel(writer,sheet_name=f'market_share_{sheet_name_download}',index=False)
                                             
                                    #         df_significance.to_excel(writer,sheet_name='significance',index=False)
                                             
                                    #         df_perc_changes.to_excel(writer,sheet_name='perc_changes',index=False)
                                    
                                    
                                    #st.download_button(
                                    #         label="📤",
                                    #         data=buffer,
                                    #         file_name=f"Equity_danone_{market}_{datetime.today()}.xlsx",
                                    #         mime="application/vnd.ms-excel")
                          
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
         
#-------------------------------------------------------------------------------------------------------------// Compare plot//------------------------------------------------------------------------------------
                           #with tab3:
                                    #creating the weighted file and the plot  
                                    #column_1,column_2 = st.columns([1,1])
                                    #with column_1:
                                    #         sheet_name_1 = st.selectbox("Select sheet 1",["Average","Absolute", "Market Share weighted"])
                                    #with column_2:
                                    #         sheet_name_2 = st.selectbox("Select sheet 2",["Absolute","Average", "Market Share weighted"])
                                    
                                    #if sheet_name_1 == "Average":
                                    #         sheet_1 = df
                                    #if sheet_name_1 == "Absolute":
                                    #         sheet_1 = df_total_uns
                                    #if sheet_name_1 == "Market Share weighted":
                                    #         sheet_1 = market_share_weighted
                                    
                                    #if sheet_name_2 == "Average":
                                    #         sheet_2 = df
                                    #if sheet_name_2 == "Absolute":
                                    #         sheet_2 = df_total_uns
                                    #if sheet_name_2 == "Market Share weighted":
                                             #sheet_2 = market_share_weighted
                                    
                                    #column_1,column_2 = st.columns([1,1])
                                    #with column_1:
                                    #         weighted_1_page = st.number_input("sheet 1 weight (%)", min_value=0, max_value=100, value=75, step=5, key="sheet 1")
                                    #with column_2:
                                    #         weighted_2_page = st.number_input("sheet 2 weight (%)", min_value=0, max_value=100, value=75, step=5, key="sheet 2")
                                    
                                    #if weighted_1_page + weighted_2_page != 100:
                                    #         st.warning("The values of the weights need to be equal to 100 %")
                                    #else:
                                    #         weighted_1_page = weighted_1_page/100
                                    #         weighted_2_page = weighted_2_page/100
                                    
                                    
                                    #df_weighted = get_weighted(sheet_1,sheet_2,weighted_1_page,weighted_2_page,brand_mapping,user_to_equity,affinity_labels)
                                    # Comparing all the sheets
                                    #fig = Comparing_Equity(df,df_total_uns,df_weighted,category_options,time_period_options,framework_options,brand_mapping,affinity_to_user,original_category,changed_category,general_equity_to_user)
                                    #st.plotly_chart(fig,use_container_width=True)
                                    
                                    #buffer = io.BytesIO()
                                    #with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                    #         df_weighted.to_excel(writer, sheet_name=f'weighted_combined', index=False)
                                    
                                    
                                    #new_file_name = f"{sheet_name_1}_{sheet_name_2}_weighted_{datetime.today()}.xlsx"
                                    
                                    #st.download_button(
                                    #label="📤",
                                    #data=buffer,
                                    #file_name=new_file_name)
         
#--------------------------------------------------------------------------------------// Equity plot //----------------------------------------------------------------------------------
                           #with tab4:
                                    #chosing the sheet name 
                                    #column_1,_,_,_ = st.columns(4)
                                    #with column_1:
                                    #         sheet_name = st.selectbox("Select sheet",["Average","Absolute", "Market Share Weighted"])
                                    
                                    #if sheet_name == "Average":
                                    #         sheet_name = "Average Smoothening"
                                    
                                    #if sheet_name == "Absolute":
                                    #         sheet_name = "Total Unsmoothening"

                                    #if sheet_name =="Market Share Weighted":
                                    #         sheet_name = "Market Share Weighted"
                           

                                    
                                    #if sheet_name == "Average Smoothening":
                                    #         fig = Equity_plot(df,category_options,time_period_options,framework_options,sheet_name,framework_to_user)
                                    #         st.plotly_chart(fig,use_container_width=True)
                                    
                                    #if sheet_name == "Total Unsmoothening":
                                    #         fig = Equity_plot(df_total_uns,category_options,time_period_options,framework_options,sheet_name,framework_to_user)
                                    #         st.plotly_chart(fig,use_container_width=True)
                                    
                                    #if sheet_name == "Market Share Weighted":
                                    #         fig = Equity_plot(market_share_weighted,category_options,time_period_options,framework_options,sheet_name,framework_to_user)
                                    #         st.plotly_chart(fig,use_container_width=True)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

                           st.session_state.access = True
                           st.experimental_rerun()
#----------------------------------------------------------------------------------------------// Logout //------------------------------------------------------------------------------------------------ 

         #if logged in
         else:
                  with st.sidebar:
                           st.image(image)
                           # user input for equity and mmm file. 
                           markets_available = ["UK"]
                           column_1,_ = st.columns(2)
                           
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
                           
                           # getting our equity    
                           filepath_equity,year_equity,month_equity,day_equity,hour_equity,minute_equity,second_equity = equity_info(data,market)
                           
                           
                           # reading the equity file
                           df = reading_df(filepath_equity,sheet_name="average_smoothened")
                           df_total_uns = reading_df(filepath_equity,sheet_name="total_unsmoothened")
                           df_total_smooth = reading_df(filepath_equity,sheet_name="total_smoothened")
                           df_avg_unsmooth = reading_df(filepath_equity,sheet_name="average_unsmoothened")
                           df_significance = reading_df(filepath_equity,sheet_name="significance")
                           df_perc_changes = reading_df(filepath_equity,sheet_name="perc_changes")
                           
                           
                           #Equity options
                           category_options,time_period_options,framework_options = equity_options(df,brand_mapping,original_category,changed_category,framework_options_)
                           
                             #creating the market_share_weighted
                           value_columns  = value_columns_
                           
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
                           
                           
                           df_total_uns_copy["Category"] = df_total_uns_copy["Category"].replace("baby_milk","Baby milk")
                           
                           
                           df_total_uns_copy.rename(columns=general_equity_to_user,inplace=True)

################################################################## ##################################################################################################################
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------// Market Share Weighted----------------------------------------------------------------------------------
                  
                  with st.container():
                           tab2,tab3,tab4 = st.tabs(["📈 Market Share Weighted","🔍Compare Average, Absolute and Market Share Weighted","📕 Final Equity plots"])
                  with tab2:
                            #chosing the sheet name 
                           column_1,_,_,_ = st.columns(4)
                           with column_1:
                                    sheet_name = st.selectbox("Select sheet",["Average","Absolute"])
                           
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
                           
                           # creating the columns for the app
                           right_column_1,right_column_2,left_column_1,left_column_2 = st.columns(4)
                           
                           with right_column_1:
                           #getting the date
                                    start_date = st.date_input("Select start date",value=datetime(2021, 2, 16),key='start_date')
                                    end_date =  st.date_input("Select end date",key='test1')
                           # getting the parameters
                           with right_column_2:
                                    st.session_state.category = st.radio('Choose  category:', category_options,key='test3')
                           
                           with left_column_1:    
                                    st.session_state.time_frame = st.radio('Choose  time frame:', time_period_options,key='test4')
                           
                           with left_column_2:
                                    framework = st.selectbox('Choose  framework:', value_columns,key='test5')
                           
                           
                           if st.session_state.button == False:
                                    if st.button("Run!"):
                                             #convert our dates
                                             ws = start_date.strftime('%Y-%m-%d')
                                             we = end_date.strftime('%Y-%m-%d')
                                             
                                             st.session_state.fig = Equity_plot_market_share_(market_share_weighted, st.session_state.category, st.session_state.time_frame,framework,ws,we)
                                             st.session_state.button = True
                           else:
                                    if st.button("Run!"):
                                             #convert our dates
                                             ws = start_date.strftime('%Y-%m-%d')
                                             we = end_date.strftime('%Y-%m-%d')
                                             
                                             st.session_state.fig = Equity_plot_market_share_(market_share_weighted, st.session_state.category, st.session_state.time_frame,framework,ws,we)
                                    
                           
                           
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
                                    sheet_name_1 = st.selectbox("Select sheet 1",["Average","Absolute", "Market Share weighted"])
                           with column_2:
                                    sheet_name_2 = st.selectbox("Select sheet 2",["Absolute","Average", "Market Share weighted"])
                           
                           if sheet_name_1 == "Average":
                                    sheet_1 = df
                           if sheet_name_1 == "Absolute":
                                    sheet_1 = df_total_uns
                           if sheet_name_1 == "Market Share weighted":
                                    sheet_1 = market_share_weighted
                           
                           if sheet_name_2 == "Average":
                                    sheet_2 = df
                           if sheet_name_2 == "Absolute":
                                    sheet_2 = df_total_uns
                           if sheet_name_2 == "Market Share weighted":
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
                           
                           
                           df_weighted = get_weighted(sheet_1,sheet_2,weighted_1_page,weighted_2_page,brand_mapping,user_to_equity,affinity_labels)
                           # Comparing all the sheets
                           fig = Comparing_Equity(df,df_total_uns,df_weighted,category_options,time_period_options,framework_options,brand_mapping,affinity_to_user,original_category,changed_category,general_equity_to_user)
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
                                    sheet_name = st.selectbox("Select sheet",["Average","Absolute", "Market Share Weighted"])
                           
                           if sheet_name == "Average":
                                    sheet_name = "Average Smoothening"
                           if sheet_name == "Absolute":
                                    sheet_name = "Total Unsmoothening"

                           if sheet_name =="Market Share Weighted":
                                    sheet_name = "Market Share Weighted"
                           
                           if sheet_name == "Average Smoothening":
                                    fig = Equity_plot(df,category_options,time_period_options,framework_options,sheet_name,framework_to_user)
                                    st.plotly_chart(fig,use_container_width=True)
                           
                           if sheet_name == "Total Unsmoothening":
                                    fig = Equity_plot(df_total_uns,category_options,time_period_options,framework_options,sheet_name,framework_to_user)
                                    st.plotly_chart(fig,use_container_width=True)
                           
                           if sheet_name == "Market Share Weighted":
                                    fig = Equity_plot(market_share_weighted,category_options,time_period_options,framework_options,sheet_name,framework_to_user)
                                    st.plotly_chart(fig,use_container_width=True)


         
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













