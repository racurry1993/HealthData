import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from garminconnect import Garmin
import datetime
from datetime import date, timedelta
import logging
import warnings
import google.generativeai as genai
from data_pre_processing import *

# Suppress the FutureWarning from sklearn
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="Garmin Data Dashboard")
st.title("Garmin AI Generated Analysis")
st.markdown("---")

# --- Function to call the Gemini API ---
# This function is removed as the clustering data it relies on has been removed.

# --- User Login and Data Fetching ---
with st.sidebar:
    st.header("Garmin Connect Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    st.warning("Please be aware that this application requires your Garmin Connect credentials.")
    
    # The function call is now INSIDE the button block.
    if st.button("Fetch Data"):
        if username and password:
            with st.spinner("Logging in and fetching data... This may take a few minutes for a full year of data."):
                try:
                    df = preprocessing_garmin_data(username, password)
                    st.session_state['df'] = df
                    st.success("Data fetched and pre-processed successfully!")
                except Exception as e:
                    st.error(f"Error fetching data: {e}. Please check your credentials.")
        else:
            st.error("Please enter both username and password.")

# --- Display Content After Data is Fetched ---
if 'df' in st.session_state:
    df_cleaned = st.session_state['df']
    
    st.header("Pre-processed Data Preview")
    st.dataframe(df_cleaned.head())
    st.info(f"The dataset contains data from {df_cleaned['Date'].min().date()} to {df_cleaned['Date'].max().date()}.")
    st.markdown("---")

    # --- Data Visualization Section ---
    st.header("Key Health & Activity Visualizations")
    
    col1, col2 = st.columns(2)
    
    # Define a variable for the RHR column to use
    rhr_col = 'restingHeartRate_x' if 'restingHeartRate_x' in df_cleaned.columns else None

    with col1:
        # Plot 1: Resting Heart Rate over time
        if rhr_col:
            fig_rhr = px.line(df_cleaned, x="Date", y=rhr_col, title="Resting Heart Rate Over Time")
            fig_rhr.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig_rhr, use_container_width=True)
        else:
            st.warning("Could not find resting heart rate data for plotting.")
            
    with col2:
        # Plot 2: Daily Steps over time
        if 'totalSteps' in df_cleaned.columns:
            fig_steps = px.line(df_cleaned, x="Date", y="totalSteps", title="Daily Steps Over Time")
            fig_steps.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig_steps, use_container_width=True)
        else:
            st.warning("Could not find daily steps data for plotting.")

    # --- Sleep Analysis Section ---
    st.subheader("Sleep Analysis")
    if all(col in df_cleaned.columns for col in ['sleepTimeHours', 'deepSleepHours', 'Date']):
        df_sleep = df_cleaned.copy()
        df_sleep['DayOfWeek'] = df_sleep['Date'].dt.day_name()
        
        # Calculate average sleep by day of week
        avg_sleep_by_day = df_sleep.groupby('DayOfWeek')[['sleepTimeHours', 'deepSleepHours']].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        # Plot 1: Average sleep by day of week
        fig_avg_sleep = px.bar(
            avg_sleep_by_day,
            x=avg_sleep_by_day.index,
            y=['sleepTimeHours', 'deepSleepHours'],
            barmode='group',
            labels={'value': 'Hours', 'DayOfWeek': 'Day of Week', 'variable': 'Sleep Metric'},
            title='Average Total Sleep and Deep Sleep by Day of Week'
        )
        st.plotly_chart(fig_avg_sleep, use_container_width=True)
        
        # Plot 2: Sleep Time Trend with Annotations
        fig_sleep_trend = px.line(df_sleep, x='Date', y='sleepTimeHours', title='Sleep Time Trend Over Time')
        
        max_sleep_row = df_sleep.loc[df_sleep['sleepTimeHours'].idxmax()]
        min_sleep_row = df_sleep.loc[df_sleep['sleepTimeHours'].idxmin()]
        
        fig_sleep_trend.add_annotation(x=max_sleep_row['Date'], y=max_sleep_row['sleepTimeHours'],
                                        text=f"Max: {max_sleep_row['sleepTimeHours']:.1f}h", showarrow=True, arrowhead=1)
        fig_sleep_trend.add_annotation(x=min_sleep_row['Date'], y=min_sleep_row['sleepTimeHours'],
                                        text=f"Min: {min_sleep_row['sleepTimeHours']:.1f}h", showarrow=True, arrowhead=1)
        st.plotly_chart(fig_sleep_trend, use_container_width=True)
        
        # Plot 3: Deep Sleep Percentage Trend with Annotations
        df_sleep['deepSleepPercentage'] = (df_sleep['deepSleepHours'] / df_sleep['sleepTimeHours']) * 100
        
        fig_deep_sleep_trend = px.line(df_sleep, x='Date', y='deepSleepPercentage', title='Deep Sleep Percentage Trend Over Time')

        max_ds_row = df_sleep.loc[df_sleep['deepSleepPercentage'].idxmax()]
        min_ds_row = df_sleep.loc[df_sleep['deepSleepPercentage'].idxmin()]
        
        fig_deep_sleep_trend.add_annotation(x=max_ds_row['Date'], y=max_ds_row['deepSleepPercentage'],
                                            text=f"Max: {max_ds_row['deepSleepPercentage']:.1f}%", showarrow=True, arrowhead=1)
        fig_deep_sleep_trend.add_annotation(x=min_ds_row['Date'], y=min_ds_row['deepSleepPercentage'],
                                            text=f"Min: {min_ds_row['deepSleepPercentage']:.1f}%", showarrow=True, arrowhead=1)
        st.plotly_chart(fig_deep_sleep_trend, use_container_width=True)
    else:
        st.warning("Could not find complete sleep data for plotting.")

    st.markdown("---")

    # --- LLM Integration ---
    # The LLM functionality is removed in this version of the code as it was tightly coupled with the k-means clustering output.
    # To re-add this, the prompt and data sent to the LLM would need to be re-designed to use the data from the new charts.
    st.header("LLM-Powered Data Insights")