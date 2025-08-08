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
from gspread_dataframe import set_with_dataframe
import gspread

# Suppress the FutureWarning from sklearn
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="Garmin Data Dashboard")
st.markdown("""
<style>
.stDeployButton {
    display:none;
}
</style>
""", unsafe_allow_html=True)
st.title("Garmin AI Generated Analysis")
st.markdown("---")

# --- Function to call the Gemini API ---
def get_llm_insight_with_gemini(data_summary, cluster_summary_df, cluster_label_map, api_key):
    """
    Generates a personalized insight using the Gemini API.
    """
    try:
        genai.configure(api_key=api_key)
        
        formatted_cluster_info = ""
        for cluster_id, label in cluster_label_map.items():
            if cluster_id in cluster_summary_df.index:
                summary_row = cluster_summary_df.loc[cluster_id]
                formatted_cluster_info += f"Segment '{label}': "
                
                # Check for column existence before accessing
                avg_steps = summary_row.get('totalSteps', 'N/A')
                avg_rhr = summary_row.get('restingHeartRate', 'N/A')
                avg_sleep = summary_row.get('sleepTimeHours', 'N/A')
                
                formatted_cluster_info += (
                    f"Avg Steps: {avg_steps:,.0f}, " if isinstance(avg_steps, (int, float)) else "Avg Steps: N/A, "
                )
                formatted_cluster_info += (
                    f"Avg RHR: {avg_rhr:.1f} bpm, " if isinstance(avg_rhr, (int, float)) else "Avg RHR: N/A, "
                )
                formatted_cluster_info += (
                    f"Avg Sleep: {avg_sleep:.1f} hrs.\n" if isinstance(avg_sleep, (int, float)) else "Avg Sleep: N/A.\n"
                )

        prompt = (
            f"Based on the following user health data summary and identified daily segments, "
            f"provide a comprehensive, yet easy-to-understand insight. "
            f"Highlight key achievements, explain what each daily segment (cluster) represents "
            f"based on its average characteristics, identify potential areas for improvement, "
            f"and suggest one actionable goal. The tone should be motivational and encouraging. "
            f"Use emojis to make it engaging. \n\n"
            f"Here is the overall data summary: {data_summary}\n\n"
            f"Here are the daily segments (clusters) identified:\n{formatted_cluster_info}"
        )
        
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while calling the Gemini API: {e}"
    
def get_google_sheet_client():
    """Authenticates and returns a gspread client."""
    try:
        # Use Streamlit's secrets management to get credentials
        creds = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(creds)
        return gc
    except Exception as e:
        st.error(f"Error authenticating with Google Sheets: {e}")
        return None

def save_to_google_sheet(df, spreadsheet_name, worksheet_name):
    """Saves a DataFrame to a specified worksheet in a Google Sheet."""
    gc = get_google_sheet_client()
    if gc is not None:
        try:
            sh = gc.open(spreadsheet_name)
            worksheet = sh.worksheet(worksheet_name)
            # Append the DataFrame, skipping the header for subsequent rows
            existing_data = worksheet.get_all_values()
            if not existing_data:
                # Write with header if sheet is empty
                set_with_dataframe(worksheet, df, row=1, col=1, include_column_header=True)
            else:
                # Append data without header
                set_with_dataframe(worksheet, df, row=len(existing_data) + 1, col=1, include_column_header=False)
            st.success("Data successfully saved to Google Sheets.")
        except gspread.exceptions.APIError as e:
            st.error(f"Google Sheets API Error: {e.args[0]['message']}")
        except Exception as e:
            st.error(f"An error occurred while saving to Google Sheets: {e}")

# --- User Login and Data Fetching ---
with st.sidebar:
    st.header("Garmin Connect Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    st.header("LLM Configuration")
    gemini_api_key = 'AIzaSyAiaswXxN3ngfEwMRXckBmEoZHO151jRv0'
    
    st.warning("Please be aware that this application requires your Garmin Connect credentials.")
    
    if st.button("Fetch Data"):
        if username and password:
            with st.spinner("Logging in and fetching data... This may take a few minutes to return and process all of your data."):
                try:
                    df = preprocessing_garmin_data(username, password)
                    save_to_google_sheet(df, spreadsheet_name="Garmin_User_Data", worksheet_name="Sheet1")
                    st.session_state['df'] = df
                    st.success("Data fetched and pre-processed successfully!")
                except Exception as e:
                    st.error(f"Error fetching data: {e}. Please check your credentials.")
        else:
            st.error("Please enter both username and password.")

# --- Display Content After Data is Fetched ---
if 'df' in st.session_state:
    df_cleaned = st.session_state['df']
    rhr_col = 'restingHeartRate' if 'restingHeartRate' in df_cleaned.columns else None
    st.header("Pre-processed Data Preview")
    st.dataframe(df_cleaned.head())
    st.info(f"The dataset contains data from {df_cleaned['Date'].min().date()} to {df_cleaned['Date'].max().date()}.")
    st.markdown("---")

    # --- K-Means Clustering Section (hidden from view) ---
    st.subheader("Background Analysis for AI Insights")
    st.info("The application is running AI and Machine Learning algorithms in the background to prepare data for the AI insights.")
    
    cluster_features = [
        'restingHeartRate', 'totalSteps', 'totalDistanceMeters', 
        'sleepTimeHours', 'deepSleepHours', 'remSleepHours', 
        'bodyBatteryHighestValue', 'daysSinceLastWorkout'
    ]
    
    # Filter for features that exist in the DataFrame
    present_features = [col for col in cluster_features if col in df_cleaned.columns]
    
    if len(present_features) > 1:
        cluster_data = df_cleaned[present_features].dropna()
        n_clusters = 4
        
        if not cluster_data.empty:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_cleaned.loc[cluster_data.index, 'cluster'] = kmeans.fit_predict(scaled_data)
            
            cluster_summary_df = df_cleaned.groupby('cluster')[present_features].mean()
            cluster_label_map = {i: f'Cluster {i}' for i in range(n_clusters)}
            
            st.session_state['cluster_info'] = (cluster_summary_df, cluster_label_map)
        else:
            st.warning("Not enough data to perform clustering for AI insights.")
            st.session_state['cluster_info'] = None
    else:
        st.warning("Not enough features available to perform clustering for AI insights.")
        st.session_state['cluster_info'] = None

    # --- LLM Integration ---
    st.header("LLM-Powered Data Insights")
    st.markdown("""
        This section uses a Large Language Model to analyze your data and provide a personalized summary.
    """)
    
    if st.button("Get AI Analysis"):
        if not gemini_api_key:
            st.error("Please enter a valid Gemini API Key in the sidebar.")
        elif st.session_state.get('cluster_info') is None:
            st.error("Please run the clustering analysis first before requesting AI insights.")
        else:
            with st.spinner("Generating personalized AI insights with the LLM..."):
                cluster_summary_df, cluster_label_map = st.session_state['cluster_info']

                final_summary_dict = {
                    'total_days_tracked': len(df_cleaned),
                    'average_daily_steps': df_cleaned['totalSteps'].mean() if 'totalSteps' in df_cleaned.columns else 0,
                    'average_resting_hr': df_cleaned[rhr_col].mean() if rhr_col in df_cleaned.columns else 0,
                    'average_sleep_hours': df_cleaned['sleepTimeHours'].mean() if 'sleepTimeHours' in df_cleaned.columns else 0,
                }
                
                llm_output = get_llm_insight_with_gemini(
                    final_summary_dict, 
                    cluster_summary_df, 
                    cluster_label_map,
                    gemini_api_key
                )
                
            st.markdown(llm_output)

    st.markdown("---")

    # --- KPI Cards Section ---
    st.subheader("RestingHR Insights")
    if 'ActivityPerformedToday' in df_cleaned.columns and 'totalSteps' in df_cleaned.columns and rhr_col:
        
        col1, col2 = st.columns(2)

        # RHR KPI Card
        with col1:
            avg_rhr_with_activity = df_cleaned[df_cleaned['ActivityPerformedToday']][rhr_col].mean()
            avg_rhr_without_activity = df_cleaned[~df_cleaned['ActivityPerformedToday']][rhr_col].mean()
            
            st.metric(label="Avg RHR (With Activity)", value=f"{avg_rhr_with_activity:.1f} bpm")
            st.metric(label="Avg RHR (Without Activity)", value=f"{avg_rhr_without_activity:.1f} bpm")

        # Steps KPI Card
        with col2:
            avg_steps_with_activity = df_cleaned[df_cleaned['ActivityPerformedToday']]['totalSteps'].mean()
            avg_steps_without_activity = df_cleaned[~df_cleaned['ActivityPerformedToday']]['totalSteps'].mean()

            st.metric(label="Avg Steps (With Activity)", value=f"{avg_steps_with_activity:,.0f}")
            st.metric(label="Avg Steps (Without Activity)", value=f"{avg_steps_without_activity:,.0f}")
    else:
        st.warning("Could not create KPI cards. 'ActivityPerformedToday', 'totalSteps', or 'restingHeartRate' columns are missing from the data.")

    st.markdown("---")

    # --- Data Visualization Section ---
    st.header("Key Health & Activity Visualizations")
    
    col1, col2 = st.columns(2)

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
            fig_steps = px.bar(df_cleaned, x="Date", y="totalSteps", color='ActivityPerformedToday', title="Daily Steps Over Time")
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

        # Calculate the variance score (absolute difference)
        avg_sleep_by_day['variance_score'] = (
            avg_sleep_by_day['sleepTimeHours'] - avg_sleep_by_day['deepSleepHours']
        ).abs()
        
        # Plot 1: Avg DOW Variance between total sleep and deep sleep
        fig_avg_sleep = px.bar(
            avg_sleep_by_day,
            x=avg_sleep_by_day.index,
            y='variance_score',
            color='variance_score',
            labels={'value': 'Score', 'DayOfWeek': 'Day of Week', 'variable': 'Variance Score'},
            title='Variance between Deep Sleep and Total Sleep (> variance = worse quality sleep)'
        )
        st.plotly_chart(fig_avg_sleep, use_container_width=True)
        
        # Plot 2: Sleep Time Trend with Annotations and Trendline
        fig_sleep_trend = px.scatter(df_sleep, x='Date', y='sleepTimeHours', title='Sleep Time Trend Over Time', trendline='ols', template='plotly_dark')
        
        max_sleep_row = df_sleep.loc[df_sleep['sleepTimeHours'].idxmax()]
        min_sleep_row = df_sleep.loc[df_sleep['sleepTimeHours'].idxmin()]
        
        fig_sleep_trend.add_annotation(x=max_sleep_row['Date'], y=max_sleep_row['sleepTimeHours'],
                                        text=f"Max: {max_sleep_row['sleepTimeHours']:.1f}h", showarrow=True, arrowhead=1)
        fig_sleep_trend.add_annotation(x=min_sleep_row['Date'], y=min_sleep_row['sleepTimeHours'],
                                        text=f"Min: {min_sleep_row['sleepTimeHours']:.1f}h", showarrow=True, arrowhead=1)
        st.plotly_chart(fig_sleep_trend, use_container_width=True)
        
        # Plot 3: Deep Sleep Percentage Trend with Annotations and Trendline
        df_sleep['deepSleepPercentage'] = (df_sleep['deepSleepHours'] / df_sleep['sleepTimeHours']) * 100
        
        fig_deep_sleep_trend = px.scatter(df_sleep, x='Date', y='deepSleepPercentage', title='Deep Sleep Percentage Trend Over Time', trendline='ols', template='plotly_dark')

        max_ds_row = df_sleep.loc[df_sleep['deepSleepPercentage'].idxmax()]
        min_ds_row = df_sleep.loc[df_sleep['deepSleepPercentage'].idxmin()]
        
        fig_deep_sleep_trend.add_annotation(x=max_ds_row['Date'], y=max_ds_row['deepSleepPercentage'],
                                            text=f"Max: {max_ds_row['deepSleepPercentage']:.1f}%", showarrow=True, arrowhead=1)
        fig_deep_sleep_trend.add_annotation(x=min_ds_row['Date'], y=min_ds_row['deepSleepPercentage'],
                                            text=f"Min: {min_ds_row['deepSleepPercentage']:.1f}%", showarrow=True, arrowhead=1)
        st.plotly_chart(fig_deep_sleep_trend, use_container_width=True)

    st.markdown("---")