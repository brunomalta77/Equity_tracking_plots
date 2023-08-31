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
from PIL import Image

#page config
st.set_page_config(page_title="Equity Tracking plots app",page_icon="💼",layout="wide")
logo_path = r"data/brand_logo.png"
image = Image.open(logo_path)
st.sidebar.image(image, width=150)

st.title("Equity Tracking plots (V 0.1)")

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
    category = st.radio('Choose your category:', categories,key="90")
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

# Sub-plots for comparing the weighted vs the unweighted
def sub_plots_w(df,df_weighted,categories,time_frames,frameworks):
    st.subheader("Weighted vs Unweighted")
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

    line_plot = px.line(df_filtered, x="time", y=framework, color="brand",color_discrete_sequence=["blue", "green", "red", "purple", "orange"])
    for trace in line_plot.data:
        sub_fig.add_trace(trace,row=1,col=1)

    histogram = px.line(df_filtered_w,x="time",y=framework,color ="brand",color_discrete_sequence=["blue", "green", "red", "purple", "orange"])
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
        markets_available = ["germany","UK","italy"]
        market = st.selectbox('', markets_available)
        
        if market == "germany":
            slang = "MMM_DE_"
            res_weighted = None

        if market == "UK":
            slang ="MMM_UK_"
            res_weighted = "Yes"
            market_weighted = "uk_equity_age_weighted"

        if market =="italy":
            slang ="MMM_IT"
            res_weighted = None
    
        # getting our equity    
        if res_weighted == "Yes":
            filepath_equity,year_equity,month_equity,day_equity,hour_equity,minute_equity,second_equity = equity_info(data,market)
            filepath_equity_weighted,year_equity_w,month_equity_w,day_equity_w,hour_equity_w,minute_equity_w,second_equity_w = equity_info(data,market_weighted)
        else:
            filepath_equity,year_equity,month_equity,day_equity,hour_equity,minute_equity,second_equity = equity_info(data,market)

        #getting our mmm
        filepath_mmm,year_mmm,month_mmm,day_mmm,hour_mmm,minute_mmm,second_mmm = mmm_info(data,slang)

        if res_weighted == "Yes":
            st.write(f"""**Equity file version** {market} : {day_equity}/{month_equity}/{year_equity} - {hour_equity}: {minute_equity}: {second_equity} **Age weighted equity file version** {market_weighted}: {day_equity_w}/{month_equity_w}/{year_equity_w} - {hour_equity_w}: {minute_equity_w}: {second_equity_w} 
            **MMM data version** {market} : {day_mmm}/{month_mmm}/{year_mmm} - {hour_mmm}: {minute_mmm}: {second_mmm}""")
        else:
            st.write(f"""**Equity file version** {market}: {day_equity}/{month_equity}/{year_equity} - {hour_equity}: {minute_equity}: {second_equity} **MMM data version** {market} : {day_mmm}/{month_mmm}/{year_mmm}- {hour_mmm} - {minute_mmm}: {second_mmm}""")


        # reading the equity file
        if res_weighted == "Yes":
            df = reading_df(filepath_equity)
            df_weighted = reading_df(filepath_equity_weighted)
        else:
            df = reading_df(filepath_equity)

        # reading and processing the mmm file
        df_vol = processing_mmm(filepath_mmm)
        
        #creating the merged df 
        if res_weighted == "Yes":
            merged_df = merged_file(df,df_vol)
            merged_df_weighted = merged_file(df_weighted,df_vol)
        else:
            merged_df = merged_file(df,df_vol)

        #Equity options
        if res_weighted == "Yes":
            category_options,time_period_options,framework_options = equity_options(df)
            category_options_w,time_period_options_w,framework_options_w = equity_options(df_weighted)
        else:
            category_options,time_period_options,framework_options = equity_options(df)


        # Volume share options
        category_options_vol_share = df_vol["Category"].unique()

        #Merged options
        category_options_merged,time_period_options_merged,framework_options_merged,framework_options_value = merged_options(merged_df)

        # Significance options
        Brand_options = merged_df["brand"].unique()
        framework_options_sig = ["Volume_share","AF_Value_for_Money", "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity","Price_change"]
        lower,upper = calculate_confidence_intervals(merged_df["Framework_Awareness"])

        # Correlation options
        Brand_options = merged_df["brand"].unique()
        framework_options_corr = ["Volume_share","AF_Value_for_Money", "Framework_Awareness", "Framework_Saliency", "Framework_Affinity", "Total_Equity","Price_change"]


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

         # Comparing the weighted vs the unweighted
        if res_weighted == "Yes":
            fig_weigheted_vs_un = sub_plots_w(df,df_weighted,category_options,time_period_options,framework_options)
            st.plotly_chart(fig_weigheted_vs_un,use_container_width=True)
        
        #Market share Plot 
        fig_market_share = market_share_plot(df_vol,category_options_vol_share)
        st.plotly_chart(fig_market_share,use_container_width=True)

        #Buble plot
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

        # Significance Plot
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













