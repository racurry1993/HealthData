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
st.title("Garmin Data Pre-Processing & Analysis")
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
                avg_rhr = summary_row.get('restingHeartRate_x', 'N/A')
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
        
        model = genai.GenerativeModel('gemini-2.5-pro') # Using a stable model
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while calling the Gemini API: {e}"

# --- User Login and Data Fetching ---
with st.sidebar:
    st.header("Garmin Connect Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    st.header("LLM Configuration")
    gemini_api_key = 'AIzaSyAiaswXxN3ngfEwMRXckBmEoZHO151jRv0'
    
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
    
    with col1:
        # Plot 1: Resting Heart Rate over time
        fig_rhr = px.line(df_cleaned, x="Date", y="restingHeartRate_x", title="Resting Heart Rate Over Time")
        fig_rhr.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_rhr, use_container_width=True)
        
    with col2:
        # Plot 2: Daily Steps over time
        fig_steps = px.line(df_cleaned, x="Date", y="totalSteps", title="Daily Steps Over Time")
        fig_steps.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_steps, use_container_width=True)

    # Plot 3: Sleep Metrics
    st.subheader("Sleep Data")
    sleep_df = df_cleaned[['Date', 'sleepTimeHours', 'deepSleepHours', 'remSleepHours', 'lightSleepHours']]
    sleep_df.set_index('Date', inplace=True)
    fig_sleep = px.area(sleep_df, y=['deepSleepHours', 'remSleepHours', 'lightSleepHours'], 
                         title="Sleep Stage Duration Over Time",
                         labels={'value':'Hours', 'variable':'Sleep Stage'})
    st.plotly_chart(fig_sleep, use_container_width=True)
    st.markdown("---")

    # --- K-Means Clustering Section ---
    st.header("Behavioral Clustering")
    
    cluster_features = st.multiselect(
        "Select features for clustering:",
        options=[
            'restingHeartRate_x', 'totalSteps', 'totalDistanceMeters', 
            'sleepTimeHours', 'deepSleepHours', 'remSleepHours', 
            'bodyBatteryHighestValue', 'daysSinceLastWorkout'
        ],
        default=['totalSteps', 'sleepTimeHours', 'restingHeartRate_x', 'bodyBatteryHighestValue']
    )
    
    if len(cluster_features) > 1:
        n_clusters = st.slider("Select number of clusters:", 2, 8, 4)
        
        # Prepare data for clustering
        cluster_data = df_cleaned[cluster_features].dropna()
        if not cluster_data.empty:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Apply KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_cleaned.loc[cluster_data.index, 'cluster'] = kmeans.fit_predict(scaled_data)
            
            # Create a summary of the clusters
            cluster_summary_df = df_cleaned.groupby('cluster')[cluster_features].mean()
            
            # Create descriptive labels for clusters (simplified)
            cluster_label_map = {i: f'Cluster {i}' for i in range(n_clusters)}
            
            # Plot the clusters
            st.subheader(f"Clusters based on selected features ({n_clusters} clusters)")
            
            # Choose a 3D or 2D plot based on the number of selected features
            if len(cluster_features) >= 3:
                fig_cluster = px.scatter_3d(
                    df_cleaned.dropna(subset=cluster_features), # Plot only rows with complete data
                    x=cluster_features[0],
                    y=cluster_features[1],
                    z=cluster_features[2],
                    color='cluster',
                    hover_data=['Date', 'activityType'],
                    title=f"K-Means Clusters ({cluster_features[0]}, {cluster_features[1]}, {cluster_features[2]})"
                )
            else:
                fig_cluster = px.scatter(
                    df_cleaned.dropna(subset=cluster_features),
                    x=cluster_features[0],
                    y=cluster_features[1],
                    color='cluster',
                    hover_data=['Date', 'activityType'],
                    title=f"K-Means Clusters ({cluster_features[0]} vs. {cluster_features[1]})"
                )
            st.plotly_chart(fig_cluster, use_container_width=True)
            st.session_state['cluster_info'] = (cluster_summary_df, cluster_label_map)
        else:
            st.warning("Not enough data to perform clustering with selected features.")
            st.session_state['cluster_info'] = None
    else:
        st.warning("Please select at least two features for clustering.")
        st.session_state['cluster_info'] = None
    st.markdown("---")

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
            with st.spinner("Generating personalized AI insights with Gemini..."):
                cluster_summary_df, cluster_label_map = st.session_state['cluster_info']

                # Create a simple overall data summary
                final_summary_dict = {
                    'total_days_tracked': len(df_cleaned),
                    'average_daily_steps': df_cleaned['totalSteps'].mean() if 'totalSteps' in df_cleaned.columns else 0,
                    'average_resting_hr': df_cleaned['restingHeartRate_x'].mean() if 'restingHeartRate_x' in df_cleaned.columns else 0,
                    'average_sleep_hours': df_cleaned['sleepTimeHours'].mean() if 'sleepTimeHours' in df_cleaned.columns else 0,
                }
                
                llm_output = get_llm_insight_with_gemini(
                    final_summary_dict, 
                    cluster_summary_df, 
                    cluster_label_map,
                    gemini_api_key
                )
                
            st.markdown(llm_output)