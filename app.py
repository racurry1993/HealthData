"""
Garmin Coach Dashboard - Streamlit app

Requirements (install via pip):
streamlit pandas numpy plotly gspread gspread_dataframe scikit-learn xgboost google-generative-ai garminconnect python-dateutil

Secrets expected in Streamlit secrets:
- gcp_service_account: (service account JSON dict)
- gemini_api_key: "<YOUR_GEMINI_API_KEY>"
- user_spreadsheet_url: "https://docs.google.com/..." (optional; otherwise a new sheet named Garmin_User_Data will be created)

Be careful: do NOT store any user passwords in plaintext permanently.
This app uses Google Sheets for light persistence (Users, Goals, ActivityData).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
from datetime import date, timedelta
import gspread
from gspread_dataframe import set_with_dataframe
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import uuid
import math
import warnings
from data_pre_processing import preprocessing_garmin_data
from garminconnect import Garmin
import json
import time
from dateutil.relativedelta import relativedelta
from pathlib import Path
import os
import google.generativeai as genai

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Garmin Coach Dashboard (Beta)")

# ==========================================================
# Google Sheets helpers (with caching to reduce API calls)
# ==========================================================

# Cache the Google Sheets client — resource-level (long-lived)
@st.cache_resource(show_spinner=False)
def get_gs_client():
    try:
        # Reconstruct the credentials dictionary from the flattened secrets
        creds = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(creds)
        return gc
    except Exception as e:
        raise RuntimeError(
            "Google Sheets auth failed. Check your 'gcp_service_account' secret "
            "(service account JSON) and that the target Sheets are shared with "
            "the service account email."
        ) from e

def ensure_sheet_and_tabs(spreadsheet_name="Garmin_User_Data"):
    """
    Create or open the spreadsheet and ensure all required tabs exist with headers.
    Not cached because it returns handles; we only call it on writes or explicit needs.
    """
    gc = get_gs_client()
    try:
        sh = gc.open_by_url(st.secrets.get("user_spreadsheet_url")) if st.secrets.get("user_spreadsheet_url") else gc.open(spreadsheet_name)
    except Exception:
        # create new spreadsheet
        sh = gc.create(spreadsheet_name)
    # required worksheets and headers
    required = {
        "Users": ["email", "role", "linked_coach_email", "certified_coach", "date_joined"],
        "Goals": ["goal_id", "user_email", "goal_type", "start_value", "target_value", "target_date", "status", "created_date", "progress_pct", "forecast_value", "forecast_success"],
        "ActivityData": ["user_email", "Date"] # ActivityData will be appended with many columns; header minimal
    }
    for ws_name, headers in required.items():
        try:
            ws = sh.worksheet(ws_name)
            # Ensure header exists (if missing or too short replace first row)
            cur_header = ws.row_values(1)
            if not cur_header or len(cur_header) < len(headers):
                try:
                    ws.delete_rows(1)
                except Exception:
                    pass
                ws.insert_row(headers, index=1)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=ws_name, rows=2000, cols=60)
            ws.insert_row(headers, index=1)
    return sh

# ------------------------------
# Cached reads (invalidate after writes)
# ------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def _read_users_df(spreadsheet_name="Garmin_User_Data"):
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("Users")
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=["email", "role", "linked_coach_email", "certified_coach", "date_joined"])
    return df

@st.cache_data(ttl=300, show_spinner=False)
def _read_goals_df(spreadsheet_name="Garmin_User_Data"):
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("Goals")
    records = ws.get_all_records()
    if not records:
        return pd.DataFrame(columns=["goal_id", "user_email", "goal_type", "start_value", "target_value", "target_date", "status", "created_date", "progress_pct", "forecast_value", "forecast_success"])
    return pd.DataFrame(records)

@st.cache_data(ttl=300, show_spinner=False)
def _read_activity_df(spreadsheet_name="Garmin_User_Data"):
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("ActivityData")
    records = ws.get_all_records()
    if not records:
        return pd.DataFrame(columns=["user_email", "Date"])
    df = pd.DataFrame(records)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def invalidate_all_cached_reads():
    # Clears all cached read_* dataframes so subsequent calls reread from Sheets after a write
    try:
        _read_users_df.clear()
        _read_goals_df.clear()
        _read_activity_df.clear()
    except Exception:
        # Fallback: nuke all cache_data
        st.cache_data.clear()

# ------------------------------
# Thin wrappers (preserve existing function names)
# ------------------------------
def read_users(spreadsheet_name="Garmin_User_Data"):
    return _read_users_df(spreadsheet_name)

def read_goals(spreadsheet_name="Garmin_User_Data"):
    return _read_goals_df(spreadsheet_name)

def read_activity_for_user(user_email, spreadsheet_name="Garmin_User_Data"):
    df = _read_activity_df(spreadsheet_name)
    if df.empty or "user_email" not in df.columns:
        return pd.DataFrame()
    df_user = df[df["user_email"] == user_email].copy()
    if "Date" in df_user.columns:
        df_user["Date"] = pd.to_datetime(df_user["Date"], errors="coerce")
    return df_user

def append_user_row(email, role="user", linked_coach_email="", certified_coach=False, spreadsheet_name="Garmin_User_Data"):
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("Users")
    row = [email, role, linked_coach_email, str(certified_coach).upper(), datetime.date.today().isoformat()]
    ws.append_row(row)
    invalidate_all_cached_reads()

def append_activity_data(df_activity: pd.DataFrame, user_email, spreadsheet_name="Garmin_User_Data"):
    if df_activity is None or df_activity.empty:
        return
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("ActivityData")
    df = df_activity.copy()
    df.insert(0, "user_email", user_email)
    existing = ws.get_all_values()
    if not existing or len(existing) < 1:
        set_with_dataframe(ws, df, row=1, include_column_header=True)
    else:
        set_with_dataframe(ws, df, row=len(existing) + 1, include_column_header=False)
    invalidate_all_cached_reads()

def append_goal_row(goal_row: dict, spreadsheet_name="Garmin_User_Data"):
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("Goals")
    headers = ws.row_values(1)
    row = [goal_row.get(h, "") for h in headers]
    ws.append_row(row)
    invalidate_all_cached_reads()

def update_goal_row(goal_id, updates: dict, spreadsheet_name="Garmin_User_Data"):
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("Goals")
    df = pd.DataFrame(ws.get_all_records())
    if df.empty or 'goal_id' not in df.columns:
        return False
    mask = df['goal_id'] == goal_id
    if not mask.any():
        return False
    rindex = df[mask].index[0] + 2 # +1 for header and +1 for 0-index
    headers = ws.row_values(1)
    for k, v in updates.items():
        if k in headers:
            col_idx = headers.index(k) + 1
            ws.update_cell(rindex, col_idx, str(v))
    invalidate_all_cached_reads()
    return True

# ==========================================================
# Forecasting utilities (unchanged)
# ==========================================================
def prepare_feature_df_for_metric(df_activity, metric_col, n_lags=7):
    if df_activity.empty or metric_col not in df_activity.columns:
        return pd.DataFrame()
    df = df_activity[['Date', metric_col]].dropna().copy()
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values('Date').reset_index(drop=True)
    df['day_idx'] = (df['Date'] - df['Date'].min()).dt.days
    df['rolling_mean_7'] = df[metric_col].rolling(window=n_lags, min_periods=1).mean()
    df['rolling_std_7'] = df[metric_col].rolling(window=n_lags, min_periods=1).std().fillna(0)
    df['day_of_week'] = df['Date'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    return df

def train_xgb_regressor(X, y):
    try:
        model = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.1, verbosity=0)
        model.fit(X, y)
        return model
    except Exception:
        lr = LinearRegression()
        lr.fit(X, y)
        return lr

def forecast_metric_on_date(df_activity, metric_col, target_date):
    """
    train model on historical and forecast value for target_date
    """
    if df_activity.empty or metric_col not in df_activity.columns:
        return None, None
    feat_df = prepare_feature_df_for_metric(df_activity, metric_col)
    if feat_df.empty or len(feat_df) < 5:
        # Fallback: if not enough data, return recent mean or last value
        recent_mean = df_activity[metric_col].dropna().tail(7).mean()
        if math.isnan(recent_mean):
             return None, "mean_fallback_nan" # Indicate failure to get a number
        return float(recent_mean), "mean_fallback"
    
    min_date = feat_df['Date'].min()
    target_day_idx = (pd.to_datetime(target_date) - min_date).days
    X = feat_df[['day_idx', 'rolling_mean_7', 'rolling_std_7', 'day_of_week', 'is_weekend']].fillna(0)
    y = feat_df[metric_col]
    model = train_xgb_regressor(X, y)
    last_row = feat_df.iloc[-1]
    
    X_target = pd.DataFrame({
        'day_idx': [target_day_idx],
        'rolling_mean_7': [last_row['rolling_mean_7']],
        'rolling_std_7': [last_row['rolling_std_7']],
        'day_of_week': [pd.to_datetime(target_date).weekday()],
        'is_weekend': [1 if pd.to_datetime(target_date).weekday() in (5,6) else 0]
    })
    try:
        pred = model.predict(X_target)[0]
    except Exception:
        # If prediction fails, fall back to the last known value
        pred = float(last_row[metric_col]) if not math.isnan(float(last_row[metric_col])) else None

    if pred is None: # If pred is still None, return None
        return None, model
    return float(pred), model

def forecast_rolling_average(df_activity, metric_col, forecast_days=30):
    """
    Forecasts a rolling 7-day average for the given metric for a specified number of days.
    """
    if df_activity.empty or metric_col not in df_activity.columns:
        return pd.DataFrame()

    feat_df = prepare_feature_df_for_metric(df_activity, metric_col)
    if feat_df.empty or len(feat_df) < 7:
        return pd.DataFrame()

    # Get the last 7 data points to seed the rolling average for the forecast period
    last_7_days = feat_df.tail(7)[metric_col].tolist()
    
    # Train the model on the full historical dataset
    X = feat_df[['day_idx', 'rolling_mean_7', 'rolling_std_7', 'day_of_week', 'is_weekend']].fillna(0)
    y = feat_df[metric_col]
    model = train_xgb_regressor(X, y)

    forecast_data = []
    last_known_date = feat_df['Date'].max()
    min_date = feat_df['Date'].min()

    for i in range(1, forecast_days + 1):
        future_date = last_known_date + timedelta(days=i)
        
        # Calculate the rolling mean of the most recent 7 values (historical + forecasted)
        current_rolling_mean = np.mean(last_7_days[-7:])
        current_rolling_std = np.std(last_7_days[-7:])
        
        target_day_idx = (future_date - min_date).days

        X_future = pd.DataFrame({
            'day_idx': [target_day_idx],
            'rolling_mean_7': [current_rolling_mean],
            'rolling_std_7': [current_rolling_std],
            'day_of_week': [future_date.weekday()],
            'is_weekend': [1 if future_date.weekday() in (5, 6) else 0]
        })
        
        try:
            future_value = model.predict(X_future)[0]
        except Exception:
            # Fallback to the last known value or a simple mean if prediction fails
            future_value = last_7_days[-1]
            
        forecast_data.append({
            'Date': future_date,
            metric_col: future_value,
            'rolling_mean_7': np.mean(last_7_days[-6:] + [future_value])
        })
        
        # Update the list of past values for the next iteration's rolling average calculation
        last_7_days.append(future_value)

    return pd.DataFrame(forecast_data)

# ==========================================================
# Utility: map goal type to column name from preprocessed data
# ==========================================================
def map_goal_type_to_column(goal_type_str):
    # normalize
    s = str(goal_type_str).strip().lower()
    mapping = {
        "vo2": "vO2MaxValue",
        "vo2 max": "vO2MaxValue",
        "vo2max": "vO2MaxValue",
        "resting hr": "restingHeartRate",
        "rhr": "restingHeartRate",
        "restingheart": "restingHeartRate",
        "steps": "totalSteps",
        "daily steps": "totalSteps",
        "sleep hours": "sleepTimeHours",
        "sleep": "sleepTimeHours",
        "deep sleep %": "deepSleepPercentage",
        "deep sleep percentage": "deepSleepPercentage",
        "body battery": "bodyBatteryMostRecentValue",
        "deep sleephours": "deepSleepHours",
    }
    return mapping.get(s, None)

# ==========================================================
# LLM wrapper (kept as separate function to edit prompt)
# ==========================================================
def generate_llm_insights(summary_dict, cluster_summary_text, goals_list, viewer_role="user"):
    """
    Keep this prompt function editable. It expects:
    - summary_dict : small dict of user metrics
    - cluster_summary_text : text describing clusters (optional)
    - goals_list : list of goal dicts (goal_type, start_value, target_value, target_date, progress_pct, forecast_value, forecast_success)
    - viewer_role : "user" or "coach"
    """
    try:
        # Use Streamlit secrets for API key
        gemini_api_key = 'AIzaSyAiaswXxN3ngfEwMRXckBmEoZHO151jRv0'
        if not gemini_api_key:
            return "LLM key not configured. Add 'gemini_api_key' to Streamlit secrets to enable insights."
        
        genai.configure(api_key=gemini_api_key)
        
        prompt_blocks = []
        prompt_blocks.append("You are a concise, data-driven coach. Provide a short titled response (one line title) and 4 short bullet points:")
        prompt_blocks.append("1) Short summary (1-2 sentences) of user's recent wearable trends.")
        prompt_blocks.append("2) For each active goal, state whether on-track and why (use progress_pct and forecast).")
        prompt_blocks.append("3) Give one prioritized action this week to improve goal likelihood.")
        prompt_blocks.append("4) If viewer is a coach, add a 1-sentence 'coach message' the coach can send to the athlete.")
        prompt_blocks.append(f"User summary: {summary_dict}")
        prompt_blocks.append(f"Daily segment summary: {cluster_summary_text}")
        
        if goals_list:
            prompt_blocks.append("Active goals (format: goal_type | start_value | target_value | target_date | progress_pct | forecast_value | forecast_success):")
            for g in goals_list:
                prompt_blocks.append(f"- {g['goal_type']} | {g.get('start_value','')} | {g.get('target_value','')} | {g.get('target_date','')} | {g.get('progress_pct',0)} | {g.get('forecast_value','')} | {g.get('forecast_success','')}")
        
        tone = "Use encouraging tone, emojis sparingly, and keep it short. Use numeric specifics where possible."
        if viewer_role == "coach":
            tone += " Also include a 1-line coach message for the athlete."

        prompt = "\n\n".join(prompt_blocks + [tone])
        
        # Use the gemini-2.5-flash-preview-05-20 model for grounded responses
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        resp = model.generate_content(prompt)
        
        return resp.text
    
    except Exception as e:
        return f"LLM call failed or not configured: {e}"

# ==========================================================
# UI: Core pages and helpers (defined BEFORE they are called)
# ==========================================================

def show_overview_page(user):
    st.title("My Dashboard")
    st.subheader("Overview")
    df = read_activity_for_user(user['email'])
    if df.empty:
        st.info("No synced activity data found. Use 'Fetch Data' in the left sidebar to pull from Garmin.")
        return

    # Convert relevant columns to numeric, coercing errors
    for col in ['totalSteps', 'restingHeartRate', 'deepSleepHours', 'sleepTimeHours', 'deepSleepPercentage', 'remSleepPercentage']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Convert `ActivityPerformedToday` to boolean
    if 'ActivityPerformedToday' in df.columns:
        df['ActivityPerformedToday'] = df['ActivityPerformedToday'].astype(bool)

    # KPIs
    kpi_cols = []
    if 'totalSteps' in df.columns and not df['totalSteps'].isnull().all():
        kpi_cols.append(("Avg Daily Steps", df['totalSteps'].mean()))
    if 'restingHeartRate' in df.columns and not df['restingHeartRate'].isnull().all():
        kpi_cols.append(("Avg Resting HR", df['restingHeartRate'].mean()))
    if 'sleepTimeHours' in df.columns and not df['sleepTimeHours'].isnull().all():
        kpi_cols.append(("Avg Sleep (hrs)", df['sleepTimeHours'].mean()))
    cols = st.columns(len(kpi_cols) if kpi_cols else 1)
    for i, (label, val) in enumerate(kpi_cols):
        with cols[i]:
            st.metric(label=label, value=f"{val:.1f}" if not math.isnan(val) else "N/A")

    st.markdown("---")
    st.subheader("Daily Activity Heatmap")
    heatmap_data_cols = ['totalSteps', 'restingHeartRate', 'deepSleepHours', 'sleepTimeHours', 'deepSleepPercentage']
    available_metrics = [col for col in heatmap_data_cols if col in df.columns and not df[col].isnull().all()]
    if available_metrics:
        df_heatmap = df.copy()
        df_heatmap['Date'] = pd.to_datetime(df_heatmap['Date'])
        
        metric_choice = st.selectbox(
            "Select metric for heatmap:",
            available_metrics
        )
        
        # Map selected metric to title
        title_map = {
            "totalSteps": "Total Steps",
            "restingHeartRate": "Resting Heart Rate",
            "deepSleepHours": "Deep Sleep Hours",
            "sleepTimeHours": "Total Sleep Hours",
            "deepSleepPercentage": "Deep Sleep %"
        }
        
        df_pivot = df_heatmap.pivot_table(
            index=df_heatmap['Date'].dt.isocalendar().week,
            columns=df_heatmap['Date'].dt.day_name(),
            values=metric_choice
        )
        # Reorder columns to be Mon-Sun
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_pivot = df_pivot.reindex(columns=days_order)
        
        fig = go.Figure(data=go.Heatmap(
            z=df_pivot.values,
            x=df_pivot.columns.tolist(),
            y=df_pivot.index.tolist(),
            colorscale='Viridis' if metric_choice in ["totalSteps", "deepSleepHours", "sleepTimeHours", "deepSleepPercentage"] else 'Inferno',
            showscale=True
        ))
        
        fig.update_layout(
            title=f'{title_map[metric_choice]} by Day of Week and Week of Year',
            xaxis_title="Day of Week",
            yaxis_title="Week of Year",
            yaxis_autorange='reversed'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to create the heatmap.")

    st.markdown("---")
    st.subheader("Sleep and Workout Impact Analysis")

    # --- Analysis 1: Impact of ActivityPerformedToday on Sleep ---
    df_cleaned = df.copy()
    if 'ActivityPerformedToday' in df_cleaned.columns and 'sleepTimeHours' in df_cleaned.columns:
        grouped_sleep_by_workout = df_cleaned.groupby('ActivityPerformedToday')['sleepTimeHours'].mean().reset_index()
        fig_sleep_by_workout = px.bar(
            grouped_sleep_by_workout,
            x='ActivityPerformedToday',
            y='sleepTimeHours',
            labels={'ActivityPerformedToday': 'Workout Performed Today', 'sleepTimeHours': 'Average Sleep Time (Hours)'},
            color='ActivityPerformedToday',
            color_discrete_map={True: '#4285F4', False: '#DB4437'}
        )
        fig_sleep_by_workout.update_xaxes(tickvals=[0, 1], ticktext=['No Workout', 'Workout'])
        fig_sleep_by_workout.update_layout(showlegend=False, title_x=0.5, yaxis_title='Average Sleep Time (Hours)', xaxis_title='Workout Performed Today')
        st.plotly_chart(fig_sleep_by_workout, use_container_width=True)
    else:
        st.info("Not enough data to analyze sleep impact by workout.")

    # Plot 2: Deep Sleep Percentage by Days Since Last Workout
    df_analyzed = df_cleaned.dropna(subset=['daysSinceLastWorkout']).copy()
    if not df_analyzed.empty and 'deepSleepPercentage' in df_analyzed.columns:
        df_analyzed['daysSinceLastWorkout_group'] = df_analyzed['daysSinceLastWorkout'].apply(
            lambda x: '0 (Workout Day)' if x == 0 else ('1' if x == 1 else ('2' if x == 2 else '3+'))
        )
        group_order = ['0 (Workout Day)', '1', '2', '3+']
        df_analyzed['daysSinceLastWorkout_group'] = pd.Categorical(df_analyzed['daysSinceLastWorkout_group'], categories=group_order, ordered=True)
        
        grouped_deep_sleep_by_days = df_analyzed.groupby('daysSinceLastWorkout_group')['deepSleepPercentage'].mean().reset_index()
        fig_deep_sleep_by_days = px.bar(
            grouped_deep_sleep_by_days,
            x='daysSinceLastWorkout_group',
            y='deepSleepPercentage',
            labels={'daysSinceLastWorkout_group': 'Days Since Last Workout', 'deepSleepPercentage': 'Average Deep Sleep Percentage (%)'},
            color='daysSinceLastWorkout_group'
        )
        fig_deep_sleep_by_layout = go.Layout(
          showlegend=False,
          title_x=0.5,
          yaxis_title='Average Deep Sleep Percentage (%)',
          xaxis_title='Days Since Last Workout'
        )
        fig_deep_sleep_by_days.update_layout(fig_deep_sleep_by_layout)
        st.plotly_chart(fig_deep_sleep_by_days, use_container_width=True)
    else:
        st.info("Not enough data to analyze deep sleep by days since last workout.")

    # Plot 3: Sleep Duration by Previous Day's Activity Type
    if 'ActivityPerformedToday' in df_cleaned.columns and 'activityType' in df_cleaned.columns and 'sleepTimeHours' in df_cleaned.columns:
        activity_cols_for_merge = ['Date', 'activityType']
        df_next_day_activity = df_cleaned[activity_cols_for_merge + ['ActivityPerformedToday']].copy()
        df_next_day_activity['Date_plus_1'] = df_next_day_activity['Date'] + pd.Timedelta(days=1)
        left_df = df_next_day_activity[df_next_day_activity['ActivityPerformedToday']].rename(columns={'Date': 'ActivityDate'})
        right_df = df_cleaned[['Date', 'sleepTimeHours']].rename(columns={'Date': 'NextDaySleepDate'})
        df_activity_impact = pd.merge(
            left_df,
            right_df,
            left_on='Date_plus_1',
            right_on='NextDaySleepDate',
            how='inner'
        )
        df_activity_impact_filtered = df_activity_impact[df_activity_impact['activityType'] != 'No Activity']
        if not df_activity_impact_filtered.empty:
            st.subheader("Average Sleep Duration by Previous Day's Activity Type")
            grouped_impact_by_type = df_activity_impact_filtered.groupby('activityType')['sleepTimeHours'].mean().reset_index()
            fig_impact_by_type = px.bar(
                grouped_impact_by_type,
                x='activityType',
                y='sleepTimeHours',
                labels={'activityType': 'Previous Day\'s Activity Type', 'sleepTimeHours': 'Average Sleep Time (Hours)'},
                color='activityType'
            )
            fig_impact_by_type.update_layout(showlegend=False, title_x=0.5, yaxis_title='Average Sleep Time (Hours)', xaxis_title='Previous Day\'s Activity Type')
            st.plotly_chart(fig_impact_by_type, use_container_width=True)
        else:
            st.info("No activity data for analysis on previous day's impact.")
    
    # Plot 4: Variance by Day of Week
    if 'Date' in df_cleaned.columns and 'sleepTimeHours' in df_cleaned.columns and 'deepSleepPercentage' in df_cleaned.columns:
        st.subheader("Variance between Total Sleep and Deep Sleep % by Day of Week")
        df_cleaned['day_of_week'] = pd.Categorical(df_cleaned['Date'].dt.day_name(), categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
        df_grouped_days_since = df_cleaned.groupby('day_of_week')[['sleepTimeHours', 'deepSleepPercentage']].mean()
        if not df_grouped_days_since.empty:
            df_grouped_days_since['variance_score'] = (df_grouped_days_since['sleepTimeHours'] - df_grouped_days_since['deepSleepPercentage'] / 10).abs() # Scale deep sleep % for comparison
            fig_variance = px.bar(
                df_grouped_days_since.reset_index(),
                x='day_of_week',
                y='variance_score',
                labels={'day_of_week': 'Day of Week', 'variance_score': 'Variance Score (lower is better)'},
                color='variance_score',
                color_continuous_scale='viridis'
            )
            fig_variance.update_layout(title_x=0.5, yaxis_title='Variance Score', xaxis_title='Day of Week')
            st.plotly_chart(fig_variance, use_container_width=True)
        else:
            st.info("Not enough data to calculate variance.")
    else:
        st.info("Missing required data to calculate variance.")


def show_insights_page(user):
    st.title("Insights & Forecasts")
    st.write("Get personalized insights and forecasts based on your data.")
    
    df = read_activity_for_user(user['email'])
    if df.empty:
        st.info("No synced activity data found.")
        return
        
    st.subheader("Garmin Coach AI Insights")
    with st.spinner("Generating AI insights..."):
        # Dummy data for the LLM call until we have actual data prep
        summary_dict = {}
        if 'totalSteps' in df.columns:
            summary_dict['steps_7d_avg'] = df['totalSteps'].tail(7).mean()
        if 'sleepTimeHours' in df.columns:
            summary_dict['sleep_7d_avg_hrs'] = df['sleepTimeHours'].tail(7).mean()
        
        # Pull goals for the LLM
        goals_df = read_goals(st.secrets.get("user_spreadsheet_url", "Garmin_User_Data"))
        user_goals_list = []
        if not goals_df.empty:
            active_goals = goals_df[
                (goals_df['user_email'] == user['email']) &
                (goals_df['status'] == 'active')
            ].to_dict('records')
            for g in active_goals:
                # Re-compute forecast for active goals
                goal_type = g.get('goal_type')
                if goal_type:
                    metric_col = map_goal_type_to_column(goal_type)
                    if metric_col and metric_col in df.columns:
                        cur_val = df[metric_col].iloc[-1] if not df.empty else None
                        try:
                            start_value = float(g.get('start_value')) if g.get('start_value') else None
                            target_value = float(g.get('target_value')) if g.get('target_value') else None
                            target_date = pd.to_datetime(g.get('target_date')) if g.get('target_date') else None
                        except (ValueError, TypeError):
                            start_value, target_value, target_date = None, None, None

                        if cur_val is not None and start_value is not None and target_value is not None and target_date is not None:
                            forecast_val, _ = forecast_metric_on_date(df, metric_col, target_date)
                            higher_is_better = target_value > start_value
                            if higher_is_better:
                                denom = (target_value - start_value) if (target_value - start_value) != 0 else 1e-6
                                progress_pct = max(0.0, min(100.0, (cur_val - start_value) / denom * 100.0))
                                forecast_success = bool(forecast_val >= target_value) if forecast_val is not None else "N/A"
                            else:
                                denom = (start_value - target_value) if (start_value - target_value) != 0 else 1e-6
                                progress_pct = max(0.0, min(100.0, (start_value - cur_val) / denom * 100.0))
                                forecast_success = bool(forecast_val <= target_value) if forecast_val is not None else "N/A"
                        
                            # write back updates
                            updates = {
                                "forecast_value": float(forecast_val) if forecast_val is not None else "",
                                "progress_pct": round(float(progress_pct), 2) if progress_pct != "" else "",
                                "forecast_success": str(forecast_success)
                            }
                            update_goal_row(g['goal_id'], updates)
                            # Update the local dictionary to be passed to LLM
                            g.update(updates)
                user_goals_list.append(g)

        llm_insights = generate_llm_insights(summary_dict, "Daily segments are not yet implemented.", user_goals_list, user['role'])
        st.info(llm_insights)

    st.markdown("---")
    st.subheader("Rolling Averages & 30-Day Forecast")
    
    df_ts = df.copy()
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
    df_ts = df_ts.sort_values('Date').reset_index(drop=True)
    
    # List of metrics to plot
    metrics_to_plot = ['totalSteps', 'sleepTimeHours']

    for metric in metrics_to_plot:
        if metric in df_ts.columns and not df_ts[metric].isnull().all():
            st.write(f"### {metric.replace('totalSteps', 'Total Steps').replace('sleepTimeHours', 'Total Sleep Hours')}")
            
            # Prepare data and calculate rolling average
            df_plot = df_ts[['Date', metric]].dropna().copy()
            df_plot['rolling_mean_7'] = df_plot[metric].rolling(window=7).mean().shift(-3)
            
            # Forecast the next 30 days of rolling average
            forecast_df = forecast_rolling_average(df_ts, metric, forecast_days=30)
            
            # Plot
            fig = go.Figure()

            # Add raw data points
            fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot[metric], mode='markers', name='Daily Value',
                                     marker=dict(color='lightgray', size=5)))
            
            # Add rolling average line
            fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['rolling_mean_7'], mode='lines', name='7-Day Rolling Average',
                                     line=dict(color='blue', width=2)))

            # Add forecast line
            if not forecast_df.empty:
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['rolling_mean_7'], mode='lines', name='30-Day Forecast',
                                         line=dict(color='red', width=2, dash='dash')))

            fig.update_layout(
                title=f"7-Day Rolling Average & 30-Day Forecast for {metric.replace('totalSteps', 'Total Steps').replace('sleepTimeHours', 'Total Sleep Hours')}",
                xaxis_title="Date",
                yaxis_title=metric,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Not enough data for the {metric.replace('totalSteps', 'Total Steps').replace('sleepTimeHours', 'Total Sleep Hours')} chart.")

    st.markdown("---")
    st.subheader("Goal Progress & Forecast")
    goals_df = read_goals(st.secrets.get("user_spreadsheet_url", "Garmin_User_Data"))
    if goals_df.empty:
        st.info("No goals set up yet.")
        return
        
    goals_df_user = goals_df[goals_df['user_email'] == user['email']]

    for _, g in goals_df_user.iterrows():
        st.markdown(f"**Goal:** {g['goal_type']} to {g['target_value']} by {g['target_date']}")
        
        progress_pct = g.get('progress_pct', 0)
        forecast_val = g.get('forecast_value', 'N/A')
        forecast_success = g.get('forecast_success', 'N/A')
        
        progress_bar_color = "green" if progress_pct >= 100 else "blue"
        st.progress(min(float(progress_pct) / 100, 1.0))
        st.write(f"Progress: **{float(progress_pct):.2f}%**")
        
        if forecast_val != "N/A" and forecast_val != "":
            forecast_status = "On Track" if forecast_success == 'True' else "Off Track"
            st.markdown(f"Forecasted Value on Target Date: **{float(forecast_val):.2f}** ({forecast_status})")
        else:
            st.markdown("Forecast Not Available (Not enough data yet)")


def show_goals_page(user):
    st.title("My Goals")
    st.write("This section is where you can view and set your goals.")
    goals_df = read_goals(st.secrets.get("user_spreadsheet_url", "Garmin_User_Data"))
    if not goals_df.empty:
        user_goals = goals_df[goals_df['user_email'] == user['email']]
        st.subheader("Current Goals")
        st.dataframe(user_goals)
    else:
        st.info("No goals found. Create one below.")
    
    st.subheader("Set a New Goal")
    with st.form("new_goal_form"):
        goal_type = st.selectbox("Goal Type", ["VO2 Max", "Resting HR", "Steps", "Sleep Hours"])
        col1, col2 = st.columns(2)
        with col1:
            start_value = st.number_input("Start Value (Optional)", step=0.1, format="%.2f")
        with col2:
            target_value = st.number_input("Target Value", step=0.1, format="%.2f")
        target_date = st.date_input("Target Date", min_value=datetime.date.today())
        
        submitted = st.form_submit_button("Create Goal")
        
        if submitted:
            goal_row = {
                "goal_id": str(uuid.uuid4()),
                "user_email": user['email'],
                "goal_type": goal_type,
                "start_value": start_value,
                "target_value": target_value,
                "target_date": target_date.isoformat(),
                "status": "active",
                "created_date": datetime.date.today().isoformat(),
                "progress_pct": 0,
                "forecast_value": "",
                "forecast_success": ""
            }
            append_goal_row(goal_row)
            st.success("Goal created successfully!")


def show_sync_page(user):
    st.title("Garmin Data Sync")
    st.write("Sync your data directly from Garmin Connect.")
    
    if st.secrets.get("garmin_username") and st.secrets.get("garmin_password"):
        sync_btn = st.button("Sync Data")
        if sync_btn:
            with st.spinner("Fetching data from Garmin..."):
                try:
                    # Add Garmin login logic here
                    client = Garmin(
                        st.secrets["garmin_username"],
                        st.secrets["garmin_password"]
                    )
                    client.login()
                    
                    # Fetch data for the last 90 days
                    today = datetime.date.today()
                    start_date = today - timedelta(days=90)
                    
                    activities = client.get_activities_by_date(start_date.isoformat(), today.isoformat())
                    
                    # Preprocess data using the provided function
                    df_garmin = preprocessing_garmin_data(activities, start_date)
                    
                    # Append to Google Sheets
                    append_activity_data(df_garmin, user['email'])
                    
                    st.success("Data synced successfully!")
                except Exception as e:
                    st.error(f"Failed to sync data from Garmin: {e}")
    else:
        st.warning("To sync data, you must provide your Garmin username and password in Streamlit secrets.")


# ==========================================================
# UI: Login / Registration
# ==========================================================
# Global state management
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None

# Sidebar login
st.sidebar.title("Sign in / Register")
email = st.sidebar.text_input("Email", value="", placeholder="your@email.com")
role_choice = st.sidebar.selectbox("Register as (if new)", ["user", "coach"])
btn = st.sidebar.button("Login / Register")

if btn:
    if not email:
        st.sidebar.error("Please enter an email")
    else:
        users = read_users()
        if not users.empty and (users['email'] == email).any():
            user_row = users[users['email'] == email].iloc[0].to_dict()
            st.session_state.logged_in = True
            st.session_state.user_profile = user_row
            st.sidebar.success(f"Welcome back, {email} ({user_row.get('role')})!")
        else:
            append_user_row(email, role=role_choice)
            st.session_state.logged_in = True
            st.session_state.user_profile = {"email": email, "role": role_choice, "linked_coach_email": "", "certified_coach": False}
            st.sidebar.success(f"Registered {email} as {role_choice}. You are now logged in.")
    # Rerun the app to update the main content based on the new session state
    st.experimental_rerun()


# Main content
if st.session_state.logged_in:
    # Tab navigation
    tab_overview, tab_goals, tab_insights, tab_sync = st.tabs(["Dashboard", "My Goals", "Insights", "Sync Garmin Data"])
    
    with tab_overview:
        show_overview_page(st.session_state.user_profile)
    
    with tab_goals:
        show_goals_page(st.session_state.user_profile)
    
    with tab_insights:
        show_insights_page(st.session_state.user_profile)
        
    with tab_sync:
        show_sync_page(st.session_state.user_profile)

else:
    st.title("Welcome to Garmin Coach Dashboard")
    st.info("Please log in or register using the sidebar to continue.")
