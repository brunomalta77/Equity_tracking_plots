# building a stream lite application
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
import glob
import requests
import dash
import pandas as pd
import plotly.express as px 
from plotly.subplots import make_subplots
import numpy as np 
import plotly.graph_objects as go
import os 
from PIL import Image
from datetime import datetime
from dateutil.relativedelta import relativedelta


#page config
st.set_page_config(page_title="Equity Tracking plots app",page_icon="💼",layout="wide")
logo_path = r"C:\Users\BrunoMalta\OneDrive - Brand Delta\Desktop\Testes\brand_logo.png"
image = Image.open(logo_path)
st.image(image)
st.title("Brand Delta  Equity Tracking plots (V 0.1)")
colors = ["blue", "green", "red", "purple", "orange","teal","black","paleturquoise","indigo","darkseagreen","gold","darkviolet","firebrick","navy","deeppink",
         "orangered"]

# Display the version of pandas
st.write(f"Pandas version: {pd.__version__}")

# Display the version of openpyxl
#st.write(f"Openpyxl version: {openpyxl.__version__}")


# getting the excel file first by user input
data = r"C:\Users\BrunoMalta\OneDrive - Brand Delta\Desktop\Testes\Data_bases_Danone"
media_data = r"C:\Users\BrunoMalta\OneDrive - Brand Delta\Desktop\Testes\Data_bases\Media_invest_all.xlsx"

# equity file
@st.cache_data() 
def reading_df(filepath):
    if ".xlsx" in filepath:
        df = pd.read_excel(filepath)
        return df
    if ".parquet" in filepath:
        df = pd.read_parquet(filepath)
        return df

#campaign file
@st.cache_data()
def get_campaigns(data,res_campaign_list):
    for x in os.listdir(data):
        if "campaigns" in x:
            df_campaign = pd.read_excel(os.path.join(data,x),sheet_name=res_campaign_list)
            return df_campaign 


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
    df_vol=df_vol.groupby(['time','Y','H','QT','M','W','brand','Metric','Category'])['Volume_share','Price_change'].sum().reset_index()
    

    return df_vol

@st.cache_data()
def media_plan(filepath,sheet_spend,sheet_week):
    df_uk_spend = pd.read_excel(filepath,sheet_name=sheet_spend)
    df_uk_weeks = pd.read_excel(filepath,sheet_name=sheet_week)
    return (df_uk_spend,df_uk_weeks)


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
    framework_options = ['AF_Entry_point', 'AF_Brand', 'AF_Adverts_Promo','AF_Prep_Meal','AF_Experience','AF_Value_for_Money',"Framework_Awareness","Framework_Saliency","Framework_Affinity","Total_Equity"]
    return (category_options,time_period_options,framework_options)



def merged_options(df):
    category_options_merged = df["Category"].unique()
    time_period_options_merged = df["time_period"].unique()
    framework_options_merged = ["AF_Value_for_Money", "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity"]
    framework_options_value = ["Volume_share","Price_change"]
    return(category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value)


def merged_options_media(df):
    category_options_merged_media = df["Category"].unique()
    time_period_options_merged_media = df["time_period_x"].unique()
    framework_options_media = ['AF_Entry_point', 'AF_Brand', 'AF_Adverts_Promo','AF_Prep_Meal','AF_Experience','AF_Value_for_Money',"Framework_Awareness","Framework_Saliency","Framework_Affinity","Total_Equity"]
    framework_options_value_media = ["value"]
    return(category_options_merged_media,time_period_options_merged_media,framework_options_media, framework_options_value_media)


# creating the campaign_options to choose from.
def campaign_options(df):
    campaign_list = []
    if not df.loc[df.campaign == 1].empty:
        for x in df.campaign_name.value_counts().keys():
            if x == None:
                pass
            else:
                if df.campaign_name.value_counts()[x] == 1:
                    campaign_list.append(x)
    
    else:
        st.warning("Do not have any campaigns")
        return df
    
    return campaign_list

# Equity_plot
def Equity_plot(df,categories,time_frames,frameworks):
    st.subheader("Equity Metrics Plot")
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

def market_share_plot(df,categories):
    st.subheader("Agreggated Volume Share by Brand Plot")
   
    # creating the columns
    right_column, left_column = st.columns(2)
    
    
    
    with right_column:
    #getting the date
        start_date = st.date_input("Select start date",key="1",value=datetime(2020, 1, 1))
        end_date =  st.date_input("Select end date",key="2")
        #convert our dates
        ws = start_date.strftime('%Y-%m-%d')
        we = end_date.strftime('%Y-%m-%d')
        # getting the parameters
    
    with left_column:
        category = st.radio('Choose your category:', categories)
    
    #filtering
    df_filtered =  df[(df["Category"] == category)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    
    all_brands = [x for x in df["brand"].unique()]
    colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","yellow","olive","silver","darkviolet","grey"]

    brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}


    unique_months = df_filtered['time'].dt.to_period('M').unique()

    # Customize the x-axis tick labels to show one label per month
    tickvals = [f"{m.start_time}" for m in unique_months]
    ticktext = [m.strftime("%B %Y") for m in unique_months]
    fig = px.line(df_filtered, x="time", y="Volume_share", color="brand", color_discrete_map=brand_color_mapping)

    # Update x-axis ticks
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45)

    # Format as integer and take the first three characters

    fig.update_traces(hovertemplate='X: %{x}<br>Y: %{y:.2s}')
    
    return fig


def buble_plot(df,categories,time_frames,frameworks,values):
    st.subheader("Equity vs Market Share (Bubble plot)")
    st.write("4 dimensions: Time(x) /Equity(y)/ Brands(color), Volume share or Price change (buble size)")
    
    left_column_1,left_column_2,right_column_2 = st.columns(3)
    
    with left_column_1:
        #getting the date
        start_date = st.date_input("Select start date",key="3",value=datetime(2020, 1, 1))
        end_date =  st.date_input("Select end date",key="4")
        #convert our dates
        ws = start_date.strftime('%Y-%m-%d')
        we = end_date.strftime('%Y-%m-%d')
    # getting the parameters
    with left_column_2:
        category = st.radio('Choose your category:', categories,key="5")
        time_frame = st.radio('Choose your time frame:', time_frames,key="6")
    with right_column_2:
        framework = st.selectbox('Choose your framework:', frameworks,key="7")
        value = st.selectbox('Choose  Price Change / Volume share:', values,key="8")
        
    #filter
    df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    
    all_brands = [x for x in df["brand"].unique()]
    colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","yellow","olive","silver","darkviolet","grey"]

    brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
    
    fig = px.scatter(df_filtered, x="time", y=framework,color_discrete_map=brand_color_mapping,size=value,color_discrete_sequence=["blue", "green", "red", "purple", "orange"])

    return fig

def sub_plots(df,categories,time_frames,frameworks,values):
    st.write("First plot with 3 dimensions (Time(x)/Equity(y)/brands(color) second plot only with the volume_share as histogram")
    #getting the date
    start_date = st.date_input("Select start date",key="9",value=datetime(2020, 1, 1))
    end_date =  st.date_input("Select end date",key="10")
    #convert our dates
    ws = start_date.strftime('%Y-%m-%d')
    we = end_date.strftime('%Y-%m-%d')
    # getting the parameters
    category = st.radio('Choose your category:', categories,key="11")
    time_frame = st.radio('Choose your time frame:', time_frames,key="12")
    framework = st.selectbox('Choose your framework:', frameworks,key="13")
    value = st.selectbox('Choose  Price Change / Volume share:', values,key="14")
    
    #filter
    df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    
    
    df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    
    all_brands = [x for x in df["brand"].unique()]
    brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}
    
    line_plot = px.line(df_filtered, x="time", y=framework, color="brand",color_discrete_map=brand_color_mapping ,color_discrete_sequence=["blue", "green", "red", "purple", "orange"])
    histogram = px.histogram(df_filtered,x="time",y=value,color="brand",color_discrete_map=brand_color_mapping ,color_discrete_sequence=["blue", "green", "red", "purple", "orange"],nbins=200)

    # Extract unique weeks from the "time" column
    if time_frame =="weeks":
         
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


    if time_frame == "months":
       
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

    return (sub_fig)


def sub_plots_media(df,categories,time_frames,frameworks,values):
    st.subheader("Equity vs Media spend")
    #getting the date
    start_date = st.date_input("Select start date",key="900",value=datetime(2020, 1, 1))
    end_date =  st.date_input("Select end date",key="1000")
    #convert our dates
    ws = start_date.strftime('%Y-%m-%d')
    we = end_date.strftime('%Y-%m-%d')
    # getting the parameters
    category = st.radio('Choose your category:', categories,key="124124")
    time_frame = st.radio('Choose your time frame:', time_frames,key="546456")
    framework = st.selectbox('Choose your framework:', frameworks,key="43534654")
    value_media_spend = st.selectbox('Media spend', values,key="827398")
    all_brands = [x for x in df["brand"].unique()]

    #filter
    df_filtered =  df[(df["Category"] == category) & (df["time_period_x"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    
    if time_frame == "months":
        df_filtered['time'] = df_filtered['time'].dt.to_period('M').dt.to_timestamp()

    if time_frame == "quarters":
        df_filtered['time'] = df_filtered['time'].dt.to_period('Q').dt.to_timestamp()

    if time_frame == "years":
        df_filtered['time'] = df_filtered['time'].dt.to_period('Y').dt.to_timestamp()

    if time_frame == "semiannual":
        df_filtered['time'] = df_filtered['time'].dt.to_period('Q').dt.start_time + pd.DateOffset(months=6)



    df_filtered = df_filtered.sort_values(by="time")

    brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}

    line_plot = px.line(df_filtered, x="time", y=framework, color="brand",color_discrete_map=brand_color_mapping)
    histogram = px.histogram(df_filtered,x="time",y=value_media_spend,color="brand",color_discrete_map=brand_color_mapping,nbins=200)


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
            ticktext = [f"Semiannual  - {period.strftime('%Y')}" for  period in unique_periods]
            
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
    st.subheader("Weighted VS Unweighted")
    st.write("Comparing the weihted vs Unweighted equity")
    #getting the date
    start_date = st.date_input("Select start date",key="20",value=datetime(2020, 1, 1))
    end_date =  st.date_input("Select end date",key="21")
    #convert our dates
    ws = start_date.strftime('%Y-%m-%d')
    we = end_date.strftime('%Y-%m-%d')
    # getting the parameters
    category = st.radio('Choose your category:', categories,key="22")
    time_frame = st.radio('Choose your time frame:', time_frames,key="23")
    framework = st.selectbox('Choose your framework:', frameworks,key="24")
    
    #filter df 
    df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    #filter df_weighted
    df_filtered_w =  df_weighted[(df["Category"] == category) & (df_weighted["time_period"] == time_frame)]
    df_filtered_w = df_filtered_w[(df_filtered_w['time'] >= ws) & (df_filtered['time'] <= we)]


    sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    st.write("(1)Unweighted vs (2) Weighted")


    colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","indigo","olive","silver","darkviolet","grey"]

    all_brands = [x for x in df["brand"].unique()]
    brand_color_mapping = {brand: color for brand, color in zip(all_brands, colors)}

    line_plot = px.line(df_filtered, x="time", y=framework, color_discrete_map=brand_color_mapping,color_discrete_sequence=["blue", "green", "red", "purple", "orange"])
    for trace in line_plot.data:
        sub_fig.add_trace(trace,row=1,col=1)

    histogram = px.line(df_filtered_w,x="time",y=framework,color_discrete_map=brand_color_mapping,color_discrete_sequence=["blue", "green", "red", "purple", "orange"])
    for trace in histogram.data:
        sub_fig.add_trace(trace,row=2,col=1)

    return sub_fig


# Significance Plot
def Significance_plot(df,brands,frameworks):
    st.subheader("Equity Plot w/ Significance (90% confidence interval)")
    #getting the date
    start_date = st.date_input("Select start date",key="15",value=datetime(2020, 1, 1))
    end_date =  st.date_input("Select end date",key="16")
    #convert our dates
    ws = start_date.strftime('%Y-%m-%d')
    we = end_date.strftime('%Y-%m-%d')
    
    brand = st.radio('Choose your brand:', brands,key="17")
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

    return fig



# Correlation Plot
def correlation_plot(df,brands):
    st.subheader("Correlation Plot between Equity Metrics and Aggregated Sales Volume ")

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

#Campaign plot just with the single frameworks -> If I want to check all the framework only. 
# def campaign_plot_just_single_frameworks(df,campaigns):
#     st.subheader("Campaign plot")
#     #Preprocessing ( getting the unique brands)
#     colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","indigo","olive","silver","darkviolet","grey"]
#     brands = df.brand.unique()
    
#     # creating the columns for the app
#     right_column_1,left_column_1,right_column_2,left_column_2 = st.columns(4)
    
#     with right_column_1:
#     #getting the Prior date
#         start_date_prior = st.date_input("Select start date prior",key="start_date_prior",value=datetime(2020, 1, 1))
#         end_date_prior =  st.date_input("Select end date prior ",key="end_date_prior")
#         #convert our dates
#         ws_prior = start_date_prior.strftime('%Y-%m-%d')
#         we_prior = end_date_prior.strftime('%Y-%m-%d')
    
#     with left_column_1:
#     #getting the Post date
#         start_date_post = st.date_input("Select start date pos",key="start_date_prost",value=datetime(2020, 1, 1))
#         end_date_post =  st.date_input("Select end date pos",key="start_date_post")
#         #convert our dates
#         ws_post = start_date_post.strftime('%Y-%m-%d')
#         we_post = end_date_post.strftime('%Y-%m-%d')
    
#     with right_column_2:
#         brand = st.selectbox('Choose your brand:',brands,key="brand_campaign")
#         campaign = st.multiselect('Choose your campaigns',campaigns,key="campaign_campaign")
#     # getting the framework parameters ( stand by ? )
#     # with left_column_2:
#     #     framework = st.selectbox('Choose your framework:', frameworks,key="framework_campaign")

    
#     # filtering brand
#     df_filtered = df[df.brand == brand]

#     #Filtering Time. 
#     df_filtered_prior = df_filtered[(df_filtered['time'] >= ws_prior) & (df_filtered['time'] <= we_prior)]
#     df_filtered_prior = df_filtered_prior.sort_values(by="time")
#     Framework_Awareness_prior = df_filtered_prior.Framework_Awareness.mean() 
#     Framework_Saliency_prior = df_filtered_prior.Framework_Saliency.mean() 
#     Framework_Affinity_prior = df_filtered_prior.Framework_Affinity.mean() 
    

#     df_filtered_post = df_filtered[(df_filtered['time'] >= ws_post) & (df_filtered['time'] <= we_post)]
#     df_filtered_post = df_filtered_post.sort_values(by="time")
#     Framework_Awareness_pos = df_filtered_post.Framework_Awareness.mean() 
#     Framework_Saliency_pos = df_filtered_post.Framework_Saliency.mean() 
#     Framework_Affinity_pos = df_filtered_post.Framework_Affinity.mean()  

#     # Create subplots
#     fig = make_subplots(rows=len(campaign),cols=len(campaign), subplot_titles=campaign, shared_xaxes=False, shared_yaxes=False)

#     # Loop over each campaign
#     for i, name_of_campaign in enumerate(campaign):
#         row_campaign = df_filtered[df_filtered["campaign_name"] == name_of_campaign]
#         start_date_campaign = str(row_campaign.time.iloc[0].date())
#         end_date_campaign = str(row_campaign.end_date.iloc[0])
#         campaign_df = df_filtered[((df_filtered["time"]) >= pd.to_datetime(start_date_campaign)) & (df_filtered["time"] <= pd.to_datetime(end_date_campaign))]

#         # Calculate means
#         campaign_means = [campaign_df[col].mean() for col in ['Framework_Awareness', 'Framework_Saliency', 'Framework_Affinity']]
#         prior_means = [Framework_Awareness_prior, Framework_Saliency_prior, Framework_Affinity_prior]
#         pos_means = [Framework_Awareness_pos, Framework_Saliency_pos, Framework_Affinity_pos]
#         #frameworks
#         frameworks = ['Awareness', 'Saliency', 'Affinity']
       
#        # Specify different widths for each set of bars
#         width_prior = 0.2
#         width_campaign = 0.2
#         width_pos = 0.2
       
#         # Add traces to subplots
#         fig.add_trace(go.Bar(x=frameworks, y=prior_means, name="Prior Period",width =  width_prior,marker_color=colors[i]), row=1, col=i+1)
#         fig.add_trace(go.Bar(x=frameworks, y=campaign_means, name="Campaign",width =  width_campaign,marker_color=colors[i]), row=1, col=i+1)
#         fig.add_trace(go.Bar(x=frameworks, y=pos_means, name="Post Period",width =  width_pos,marker_color=colors[i]), row=1, col=i+1)

#     # Update layout
#     fig.update_layout(title="Campaign Comparison",
#                       height=800*len(campaign),width = 1000,
#                       showlegend=False)

#     # Update y-axis tick format
#     fig.update_yaxes(tickformat=".1f")

#     return(fig)


#Campaign plot just with the campaign plots.
def campaign_plot(data,frameworks_outside):
    
    campaign_list = ["average_smoothened","total_smoothened","average_unsmoothened","total_unsmoothened"]
    res_campaign_list = st.selectbox("Select the sheet taht you want to use",campaign_list)
    st.subheader("Campaign plot")
    
    df = get_campaigns(data,res_campaign_list)
    campaigns =  campaign_options(df)
    #Preprocessing ( getting the unique brands)
    colors = ["blue", "green", "red", "purple", "orange","lightgreen","black","lightgrey","indigo","olive","silver","darkviolet","grey"]
    brands = df.brand.unique()
    time_period = ["weeks","months","years"]
    legend = None
    # creating the columns for the app
    right_column_1,left_column_1,right_column_2,left_column_2 = st.columns(4)
    
    with right_column_1:
    #getting the Prior date
        start_date_prior = st.selectbox("Choose your prior time period",time_period)
        time_period_range_prior =  st.number_input(f"How many {start_date_prior} do you want ?",key="end_date_prior",max_value = 100,value=2,step=1)
        #convert our dates
        #ws_prior = start_date_prior.strftime('%Y-%m-%d')
        #we_prior = end_date_prior.strftime('%Y-%m-%d')
    
    with left_column_1:
    #getting the Post date
        start_date_post = st.selectbox("Choose your post time period",time_period,key="post_date")
        time_period_range_post =  st.number_input(f"How many {start_date_post} do you want ?",key="end_date_post",max_value = 100,value=2,step=1)
        #convert our dates
        #ws_post = start_date_post.strftime('%Y-%m-%d')
        #we_post = end_date_post.strftime('%Y-%m-%d')
    
    with right_column_2:
        brand = st.radio('Choose your brand:',brands,key="brand_campaign")
        campaign = st.multiselect('Choose your campaigns',campaigns,key="campaign_campaign")
    # getting the framework parameters ( stand by ? )
    frameworks = ['Awareness', 'Saliency', 'Affinity']
    with left_column_2:
        framework = st.selectbox('Choose your framework:',frameworks,key="framework_campaign")

        #framework list
        if framework == "Awareness":
            framework_list = ["AA_eSoV","AA_Reach","AA_Brand_Breadth"]
        if framework == "Saliency":
            framework_list = ["AS_Average_Engagement","AS_Usage_SoV","AS_Search_Index","AS_Brand_Centrality"]
        if framework == "Affinity":
            framework_list = ['AF_Entry_point', 'AF_Brand', 'AF_Adverts_Promo','AF_Prep_Meal','AF_Experience','AF_Value_for_Money']

    # filtering brand
    df_filtered = df[df.brand == brand]


    # Grid for the subplots
    if framework == "Saliency" or framework == "Awareness" or framework =="Affinity":
        size = len(framework_list)
    try:   
        # Create subplots
        fig = make_subplots(rows=len(campaign),cols=size,shared_xaxes=False, shared_yaxes=False,row_titles= campaign)

        # Loop over each campaign
        for i, name_of_campaign in enumerate(campaign):
            row_campaign = df_filtered[(df_filtered["campaign_name"] == name_of_campaign) & (df_filtered["time_period"]=="weeks")]
            start_date_campaign = str(row_campaign.time.iloc[0].date())
            end_date_campaign = str(row_campaign.end_date.iloc[0].date())
            campaign_df = df_filtered[((df_filtered["time"]) >= pd.to_datetime(start_date_campaign)) & (df_filtered["time"] < pd.to_datetime(end_date_campaign))]
            campaign_df =  campaign_df[campaign_df["time_period"]== "weeks"]
            
            #write the campaign start and the campaign end for the user to check
            st.markdown(f"**Campaign - {name_of_campaign}** - starts at {start_date_campaign} and ends at {end_date_campaign}")

            
            #Calculating the  prior and post time, with the weeks,years numbers.....
            # filtering by the time_period_prior
            df_filtered_time_period_prior = df_filtered[df_filtered.time_period == start_date_prior]
            if start_date_prior == "weeks":
                ws_prior = datetime.strptime(start_date_campaign, "%Y-%m-%d") - relativedelta(weeks=time_period_range_prior)
            if start_date_prior == "months":
                ws_prior =  datetime.strptime(start_date_campaign, "%Y-%m-%d") - relativedelta(months=time_period_range_prior)
            if start_date_prior == "years":
                ws_prior =  datetime.strptime(start_date_campaign, "%Y-%m-%d") - relativedelta(years=time_period_range_prior)

            df_filtered_prior = df_filtered_time_period_prior[(df_filtered_time_period_prior['time'] >= ws_prior) & (df_filtered_time_period_prior['time'] < start_date_campaign )]
            df_filtered_prior = df_filtered_prior.sort_values(by="time")
        

            # filtering by the time_period_post
            df_filtered_time_period_pos = df_filtered[df_filtered.time_period == start_date_post]
            if start_date_post == "weeks":
                ws_post = datetime.strptime(end_date_campaign, "%Y-%m-%d") + relativedelta(weeks=time_period_range_post)
            if start_date_post == "months":
                ws_post = datetime.strptime(end_date_campaign, "%Y-%m-%d") + relativedelta(months=time_period_range_post)
            if start_date_post == "years":
                ws_post = datetime.strptime(end_date_campaign, "%Y-%m-%d") + relativedelta(years=time_period_range_post)

            df_filtered_post = df_filtered_time_period_pos[(df_filtered_time_period_pos['time'] > end_date_campaign) & (df_filtered_time_period_pos['time'] <= ws_post)]
            df_filtered_post = df_filtered_post.sort_values(by="time")

            # Calculate means
            campaign_means = [campaign_df[col].mean() for col in  framework_list]
            prior_means = [df_filtered_prior[col].mean() for col in  framework_list]
            pos_means = [df_filtered_post[col].mean() for col in  framework_list]
            
            
            # Specify different widths for each set of bars
            width_prior = 0.2
            width_campaign = 0.2
            width_pos = 0.2
        

            #variable for not appearing the graph
            check = False
            if framework == "Saliency" or framework =="Awareness" or framework == "Affinity":
            # Add traces to subplots
                for (index,data) in enumerate(framework_list):
                    if legend == None:
                        fig.add_trace(go.Bar(x=[framework_list[index]], y=[prior_means[index]], name="Prior Period",width =  width_prior,marker_color=colors[5],showlegend=True), row=i+1, col=1+index)
                        fig.add_trace(go.Bar(x=[framework_list[index]], y=[campaign_means[index]], name="Campaign",width =  width_campaign,marker_color=colors[3],showlegend=True), row=i+1, col=1+index)
                        fig.add_trace(go.Bar(x=[framework_list[index]], y=[pos_means[index]], name="Post Period",width =  width_pos,marker_color=colors[8],showlegend=True), row=i+1, col=1+index)
                        legend = True
                        check = True 
                    else:
                        fig.add_trace(go.Bar(x=[framework_list[index]], y=[prior_means[index]], name="Prior Period",width =  width_prior,marker_color=colors[5],showlegend=False), row=i+1, col=1+index)
                        fig.add_trace(go.Bar(x=[framework_list[index]], y=[campaign_means[index]], name="Campaign",width =  width_campaign,marker_color=colors[3],showlegend=False), row=i+1, col=1+index)
                        fig.add_trace(go.Bar(x=[framework_list[index]], y=[pos_means[index]], name="Post Period",width =  width_pos,marker_color=colors[8],showlegend=False), row=i+1, col=1+index)
                        
                    if check == True:
                        fig.update_layout(height=1200 ,width = 1500)
                        # Update y-axis tick format
                        fig.update_yaxes(tickformat=".1f")
                        
                    else:
                        pass

        return(fig)

    except:
        st.warning("Check your data:your equity data does not have campaigns or your brand does not have campaigns!")


#------------------------------------------------------------------------app------------------------------------------------------------------------------------------------------------------#

def main():
    with st.container():
    #None Global

        # user input for equity and mmm file. 
        markets_available = ["UK"]
        market = st.selectbox('Markets currently available:', markets_available)
        st.warning("Choose your market")
        
        
        # if market == "japan":
        #     slang = "MMM_JA"
        #     res_weighted = None
        #     mmm = None
        #     campaign_option = True

        # if market == "germany":
        #     slang = "MMM_DE_"
        #     res_weighted = None
        #     mmm = "Yes"
        #     sheet_week = "WeekMap_DE"
        #     sheet_spend = "DE_Raw Spend"
        #     campaign_option = None

        if market == "UK":
            slang ="MMM_UK_"
            res_weighted = None
            market_weighted = "uk_equity_age_weighted"
            mmm = None
            sheet_week = "WeekMap_UK"
            sheet_spend = "UK_Raw Spend"
            campaign_option = None

        # if market =="italy":
        #     slang ="MMM_IT"
        #     res_weighted = None
        #     mmm= "Yes"
        #     sheet_week = "WeekMap_IT"
        #     sheet_spend = "IT_Raw Spend"
        #     campaign_option = None

        # if market =="france":
        #     slang = "MMM_FR"
        #     res_weighted = None
        #     mmm = None
        #     campaign_option = None
        

        # getting our equity    
        if res_weighted == "Yes":
            filepath_equity,year_equity,month_equity,day_equity,hour_equity,minute_equity,second_equity = equity_info(data,market)
            filepath_equity_weighted,year_equity_w,month_equity_w,day_equity_w,hour_equity_w,minute_equity_w,second_equity_w = equity_info(data,market_weighted)
        else:
            filepath_equity,year_equity,month_equity,day_equity,hour_equity,minute_equity,second_equity = equity_info(data,market)

         
        #getting our mmm
        if mmm ==None:
            pass 
        else:
            filepath_mmm,year_mmm,month_mmm,day_mmm,hour_mmm,minute_mmm,second_mmm = mmm_info(data,slang)


        # for France
        if mmm == None: 
             st.write(f"*From the {market} market - Equity date {day_equity}/{month_equity}/{year_equity}- Hours: {hour_equity}: {minute_equity}: {second_equity}*")            

        else:

            if res_weighted == "Yes":
                st.write(f"*From the {market} market - Equity date {day_equity}/{month_equity}/{year_equity}- Hours: {hour_equity}: {minute_equity}: {second_equity}*")
                st.write(f"*From the {market_weighted} market - Equity date {day_equity_w}/{month_equity_w}/{year_equity_w}- Hours: {hour_equity_w}: {minute_equity_w}: {second_equity_w}*")

                st.write(f"*From the {market} market - MMM date {day_mmm}/{month_mmm}/{year_mmm}- Hours: {hour_mmm}: {minute_mmm}: {second_mmm}*")
            else:
                st.write(f"*From the {market} market - Equity date {day_equity}/{month_equity}/{year_equity}- Hours: {hour_equity}: {minute_equity}: {second_equity}*")
                st.write(f"*From the {market} market - MMM date {day_mmm}/{month_mmm}/{year_mmm}- Hours: {hour_mmm}: {minute_mmm}: {second_mmm}*")


        # reading the equity file
        if res_weighted == "Yes":
            df = reading_df(filepath_equity)
            df_weighted = reading_df(filepath_equity_weighted)
        else:
            df = reading_df(filepath_equity)

        # reading and processing the mmm file
        if mmm == None:
            pass
        else:
            df_vol = processing_mmm(filepath_mmm)
            df_vol = df_vol.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
            st.write(df_vol.head(10))

        #creating the merged df 
        if mmm ==None :
            pass
        else:
            if res_weighted == "Yes":
                merged_df = merged_file(df,df_vol)
                merged_df_weighted = merged_file(df_weighted,df_vol)
            else:
                merged_df = merged_file(df,df_vol)

        # creating the Media options
        if mmm == None:
            pass
        else:
            if res_weighted == "Yes":
                df_uk_spend,df_uk_weeks = media_plan(media_data,sheet_spend,sheet_week)
                merged_df_media_weighted = media_spend_processed(df_weighted,df_uk_spend,df_uk_weeks)
                category_options_merged_media_w,time_period_options_merged_media_w,framework_options_media_w, framework_options_value_media_w= merged_options_media(merged_df_media_weighted)
                
                df_uk_spend,df_uk_weeks = media_plan(media_data,sheet_spend,sheet_week)
                merged_df_media = media_spend_processed(df,df_uk_spend,df_uk_weeks)
                category_options_merged_media,time_period_options_merged_media,framework_options_media, framework_options_value_media= merged_options_media(merged_df_media)
        
            else:
                df_uk_spend,df_uk_weeks = media_plan(media_data,sheet_spend,sheet_week)
                merged_df_media = media_spend_processed(df,df_uk_spend,df_uk_weeks)
                category_options_merged_media,time_period_options_merged_media,framework_options_media, framework_options_value_media= merged_options_media(merged_df_media)



        #Equity options
        if res_weighted == "Yes":
            category_options,time_period_options,framework_options = equity_options(df)
            category_options_w,time_period_options_w,framework_options_w = equity_options(df_weighted)
        else:
            category_options,time_period_options,framework_options = equity_options(df)


        # Volume share options
        if mmm== None:
            pass
        else:
            category_options_vol_share = df_vol["Category"].unique()

            #df_vol['Volume_share'] = df_vol['Volume_share'].apply(lambda x: round(float(x), 2) if isinstance(x, (float, int)) else x)
        if mmm ==None:
            pass
        else:
            category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value = merged_options(merged_df)

        # Significance options
        if mmm ==None:
            pass
        else:
            Brand_options = merged_df["brand"].unique()
            framework_options_sig = ["Volume_share","AF_Value_for_Money", "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity"]
            lower,upper = calculate_confidence_intervals(merged_df["Framework_Awareness"])

        # Correlation options
        if mmm ==None:
            pass
        else:
            Brand_options = merged_df["brand"].unique()
            framework_options_corr = ["Volume_share","AF_Value_for_Money", "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity"]


        # Comparing the weighted vs the unweighted
        if mmm ==None:
            pass
        else:
            if res_weighted == "Yes":
                fig_weigheted_vs_un = sub_plots_w(df,df_weighted,category_options,time_period_options,framework_options)
                st.plotly_chart(fig_weigheted_vs_un,use_container_width=True)



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Equity plot
        if res_weighted == "Yes":
            res_equity_weighted = st.radio("What type do you want to see?", ["Unweighted","Weighted"])
            if res_equity_weighted == "Weighted":
                fig = Equity_plot(df_weighted,category_options,time_period_options,framework_options)
                st.plotly_chart(fig,use_container_width=True)
            else:
                fig = Equity_plot(df,category_options,time_period_options,framework_options)
                st.plotly_chart(fig,use_container_width=True)
        else: 
            fig = Equity_plot(df,category_options,time_period_options,framework_options)
            st.plotly_chart(fig,use_container_width=True)

        #Market share Plot 
        if mmm ==None:
            pass
        else:
            fig_market_share = market_share_plot(df_vol,category_options_vol_share)
            st.plotly_chart(fig_market_share,use_container_width=True)

        #Buble plot
        if mmm ==None:
            pass
        else:
        
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
            if res_weighted == "Yes":
                res_equity_weighted = st.radio("What type do you want to see?", ["Unweighted","Weighted"],key="50")
                if res_equity_weighted == "Weighted":
                    fig_significance = Significance_plot(merged_df_weighted, Brand_options,framework_options_corr)
                    st.plotly_chart(fig_significance,use_container_width=True)
                else:
                    fig_significance = Significance_plot(merged_df, Brand_options,framework_options_corr)
                    st.plotly_chart(fig_significance,use_container_width=True)
            
            else:
                fig_significance = Significance_plot(merged_df, Brand_options,framework_options_corr)
                st.plotly_chart(fig_significance,use_container_width=True)

        # Correlation Plot
        if mmm ==None:
            pass
        else:
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


        #Campaign plot 
        if campaign_option == None:
            pass
        else:
            _,_,frameworks = equity_options(df)
            campaigns =  campaign_options(df)
            fig = campaign_plot(data,frameworks,campaigns)
            try:
                st.plotly_chart(fig)
            except:
                st.warning("Impossible to show any graph")


if __name__=="__main__":
    main()   
    














