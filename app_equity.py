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
import pandas as pd
import plotly.express as px 
from plotly.subplots import make_subplots
import numpy as np 
import plotly.graph_objects as go
import os 

#page config
st.set_page_config(page_title="Equity Tracking plots app",page_icon="💼",layout="wide")
st.title("Brand Delta  Equity Tracking plots (V 0.1)")

# getting the excel file first by user input
data = r"data"

# equity file
@st.cache_data() 
def reading_df(filepath):
    df = pd.read_excel(filepath)
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


def equity_options(df):
    category_options = df["Category"].unique()
    time_period_options = df["time_period"].unique()
    framework_options = ["AF_Value_for_Money", "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity"]
    return (category_options,time_period_options,framework_options)



def merged_options(df):
    category_options_merged = df["Category"].unique()
    time_period_options_merged = df["time_period"].unique()
    framework_options_merged = ["AF_Value_for_Money", "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity"]
    framework_options_value = ["Volume_share","Price_change"]
    return(category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value)



# Equity_plot
def Equity_plot(df,categories,time_frames,frameworks):
    st.subheader("Equity Metrics Plot")
    #getting the date
    start_date = st.date_input("Select start date",value=datetime(2020, 1, 1))
    end_date =  st.date_input("Select end date")
    #convert our dates
    ws = start_date.strftime('%Y-%m-%d')
    we = end_date.strftime('%Y-%m-%d')
    # getting the parameters
    category = st.radio('Choose your category:', categories)
    time_frame = st.radio('Choose your time frame:', time_frames)
    framework = st.selectbox('Choose your framework:', frameworks)

    #filtering
    df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    
    fig = px.line(df_filtered, x="time", y=framework, color="brand")
    return fig


def market_share_plot(df,categories):
    st.subheader("Agreggated Volume Share by Brand Plot")
    #getting the date
    start_date = st.date_input("Select start date",key="1",value=datetime(2020, 1, 1))
    end_date =  st.date_input("Select end date",key="2")
    #convert our dates
    ws = start_date.strftime('%Y-%m-%d')
    we = end_date.strftime('%Y-%m-%d')
    # getting the parameters
    category = st.radio('Choose your category:', categories)
    
    #filtering
    df_filtered =  df[(df["Category"] == category)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    fig = px.line(df_filtered, x="time", y="Volume_share", color="brand")
    return fig


def buble_plot(df,categories,time_frames,frameworks,values):
    st.subheader("Equity vs Market Share (Bubble plot)")
    st.write("4 dimensions: Time(x) /Equity(y)/ Brands(color), Volume share or Price change (buble size)")
    #getting the date
    start_date = st.date_input("Select start date",key="3",value=datetime(2020, 1, 1))
    end_date =  st.date_input("Select end date",key="4")
    #convert our dates
    ws = start_date.strftime('%Y-%m-%d')
    we = end_date.strftime('%Y-%m-%d')
    # getting the parameters
    category = st.radio('Choose your category:', categories,key="5")
    time_frame = st.radio('Choose your time frame:', time_frames,key="6")
    framework = st.selectbox('Choose your framework:', frameworks,key="7")
    value = st.selectbox('Choose  Price Change / Volume share:', values,key="8")
    
    #filter
    df_filtered =  df[(df["Category"] == category) & (df["time_period"] == time_frame)]
    df_filtered = df_filtered[(df_filtered['time'] >= ws) & (df_filtered['time'] <= we)]
    fig = px.scatter(df_filtered, x="time", y=framework, color="brand",size=value,color_discrete_sequence=["blue", "green", "red", "purple", "orange"])

    return fig

# Creating the Subplots
def sub_plots(df,categories,time_frames,frameworks,values):
    st.subheader("Equity vs Market Share  (histogram)")
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
    
    sub_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    line_plot = px.line(df_filtered, x="time", y=framework, color="brand",color_discrete_sequence=["blue", "green", "red", "purple", "orange"])
    for trace in line_plot.data:
        sub_fig.add_trace(trace,row=1,col=1)

    histogram = px.histogram(df_filtered,x="time",y=value,color ="brand",color_discrete_sequence=["blue", "green", "red", "purple", "orange"],nbins=200)
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

    brand = st.radio('Choose your brand:', brands,key="20")
    
    framework_options_corr = ["Volume_share","AF_Value_for_Money", "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity","Price_change"]
    df_filtered =  df[(df["brand"] == brand)]
    filtered_data_final = df_filtered[framework_options_corr]
                    
    spearman_corr = filtered_data_final.corr(method='spearman')

    custom_colors = ["red", "green"]


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





#------------------------------------------------------------------------app---------------------------------#

def main():
    with st.container():
    #None Global
        # user input for equity and mmm file. 
        markets_available = ["germany","UK"]
        market = st.selectbox('Markets currently available:', markets_available)
        st.warning("Choose your market")
        
        if market == "germany":
            slang = "_DE_"
        if market == "UK":
            slang ="_UK_"

        # getting the information and the respective equity and mmm 
        for x in os.listdir(data):
            if market in x:
                filepath_equity = os.path.join(data,x)
                info_number = [x for x in x.split("_") if x >= "0" and x <="9"]
                year_equity,month_equity,day_equity,hour_equity,minute_equity = info_number[:5]
                second_equity = info_number[-1].split(".")[0]
                

        for x in os.listdir(data):
            if slang in x:
                filepath_mmm = os.path.join(data,x)
                info_number = [x for x in x.split("_") if x >= "0" and x <="9"]
                day_mmm,month_mmm,year_mmm,hour_mmm,minute_mmm = info_number[:5]
                second_mmm = info_number[-1].split(".")[0]

        

        st.header(f"From the {market} market - Equity date {day_equity}/{month_equity}/{year_equity},Hours:{hour_equity}:{minute_equity}:{second_equity}")
        st.header(f"From the {market} market - MMM date {day_mmm}/{month_mmm}/{year_mmm},Hours:{hour_mmm}:{minute_mmm}:{second_mmm}")

        # reading the equity file
        df = reading_df(filepath_equity)
        
        # reading and processing the mmm file
        df_vol = processing_mmm(filepath_mmm)
        
        #creating the merged df 
        merged_df = merged_file(df,df_vol)
        
        #Equity options
        category_options,time_period_options,framework_options = equity_options(df)

        # Volume share options
        category_options_vol_share = df_vol["Category"].unique()

        #Merged options
        category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value = merged_options(merged_df)

        # Significance options
        Brand_options = merged_df["brand"].unique()
        framework_options_sig = ["Volume_share","AF_Value_for_Money", "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity"]
        lower,upper = calculate_confidence_intervals(merged_df["Framework_Awareness"])

        # Correlation options
        Brand_options = merged_df["brand"].unique()
        framework_options_corr = ["Volume_share","AF_Value_for_Money", "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity"]

        #Equity plot
        fig = Equity_plot(df,category_options,time_period_options,framework_options)
        st.plotly_chart(fig,use_container_width=True)

        #Market share Plot 
        fig_market_share = market_share_plot(df_vol,category_options_vol_share)
        st.plotly_chart(fig_market_share,use_container_width=True)

        #Buble plot
        fig_buble= buble_plot(merged_df,category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value)
        st.plotly_chart(fig_buble,use_container_width=True)

        #Sub_plots
        fig_sub = sub_plots(merged_df,category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value)
        st.plotly_chart(fig_sub,use_container_width=True)

        # Significance Plot
        fig_significance = Significance_plot(merged_df, Brand_options,framework_options_corr)
        st.plotly_chart(fig_significance,use_container_width=True)

        # Correlation Plot
        fig_corr = correlation_plot(merged_df,Brand_options)
        st.plotly_chart(fig_corr,use_container_width=True)



if __name__=="__main__":
    main()   
    














