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

# -----------------
# FIX START
# -----------------
# This function was not passing the date arguments to preprocessing_garmin_data()
def sync_data_from_garmin(api, user_id):
    """
    Syncs new activity data from Garmin Connect to the Google Sheet.
    """
    st.info("Syncing data from Garmin Connect...")
    try:
        # Define the date range for data fetching
        end_date = date.today()
        start_date = end_date - timedelta(days=90)  # Fetch last 90 days

        # Fetch activities from Garmin API within the defined date range
        activities = api.get_activities_by_date(start_date.isoformat(), end_date.isoformat())

        # Pass the required arguments to the preprocessing function
        new_activity_data = preprocessing_garmin_data(activities, start_date, end_date)

        if not new_activity_data.empty:
            st.success(f"Successfully synced {len(new_activity_data)} activities from Garmin.")
            append_activity_data(new_activity_data, user_id)
        else:
            st.warning("No new activities found in the specified date range.")

    except Exception as e:
        st.error(f"Failed to sync data from Garmin: {e}")
        st.error("Please check your Garmin Connect credentials and try again.")
        # Logging the full traceback for debugging purposes
        import traceback
        st.error(traceback.format_exc())
# -----------------
# FIX END
# -----------------

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
    
    heatmap_data_cols = ['totalSteps', 'restingHeartRate', 'deepSleepHours', 'sleepTimeHours', 'totalDistanceMeters', 'totalCalories', 'vO2MaxValue', 'activeCalories', 'averageHeartRate', 'maxHeartRate']
    
    metrics = [col for col in heatmap_data_cols if col in df.columns and not df[col].isnull().all()]

    if not metrics:
        st.warning("Not enough data to create heatmaps.")
    else:
        selected_metric = st.selectbox("Select metric for heatmap:", metrics)
        
        # Resample to daily frequency
        daily_df = df.set_index('Date')[selected_metric].resample('D').sum().reset_index()
        daily_df['day_of_week'] = daily_df['Date'].dt.day_name()
        daily_df['week_of_year'] = daily_df['Date'].dt.isocalendar().week
        daily_df['year'] = daily_df['Date'].dt.year

        if not daily_df.empty:
            fig = go.Figure(data=go.Heatmap(
                    z=daily_df[selected_metric],
                    x=daily_df['week_of_year'],
                    y=daily_df['day_of_week'],
                    colorscale='Viridis',
                    hoverongaps=False))
            fig.update_layout(
                title=f"Weekly Heatmap for {selected_metric}",
                xaxis_nticks=36,
                yaxis={'categoryorder':'array', 'categoryarray':['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data to display in the heatmap.")
    
def show_goals_page(user):
    st.title("My Goals")
    
    st.markdown("---")
    st.subheader("Current Goals")
    
    goals_df = read_goals()
    if not goals_df.empty:
        # Filter goals for the current user
        user_goals_df = goals_df[goals_df['user_email'] == user['email']].copy()
        if not user_goals_df.empty:
            st.dataframe(user_goals_df)
            
            # Display progress charts for each goal
            for _, goal in user_goals_df.iterrows():
                goal_id = goal['goal_id']
                goal_type = goal['goal_type']
                
                st.subheader(f"Progress for: {goal_type}")
                
                try:
                    start_value = float(goal['start_value'])
                    target_value = float(goal['target_value'])
                    progress_pct = float(goal['progress_pct'])
                    forecast_value = float(goal['forecast_value']) if goal['forecast_value'] else None

                    # Create a progress bar
                    st.progress(progress_pct / 100)
                    st.markdown(f"**Progress:** {progress_pct:.2f}% (from {start_value} to {target_value})")
                    
                    # Create a plot showing progress vs forecast
                    if forecast_value is not None:
                        fig = go.Figure()
                        fig.add_trace(go.Indicator(
                            mode="number+gauge",
                            value=progress_pct,
                            domain={'x': [0.25, 1], 'y': [0.4, 0.9]},
                            gauge={'shape': "bullet",
                                'axis': {'range': [None, 100]},
                                'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': 100},
                                'steps': [{'range': [0, 100], 'color': "lightgray"}]},
                            title={'text': "Progress %"}
                        ))
                        fig.add_trace(go.Indicator(
                            mode="number",
                            value=forecast_value,
                            title={"text": "Forecasted Value"}
                        ))
                        st.plotly_chart(fig, use_container_width=True)

                except ValueError:
                    st.warning(f"Could not display chart for goal '{goal_type}' due to missing or invalid data.")

        else:
            st.info("You haven't added any goals yet.")

    st.markdown("---")
    st.subheader("Add a New Goal")
    with st.form("new_goal_form"):
        goal_type = st.selectbox("Goal Type", ["VO2 Max", "Resting HR", "Steps", "Sleep Hours", "Deep Sleep %", "Body Battery"])
        target_value = st.number_input("Target Value", min_value=0.0)
        target_date = st.date_input("Target Date")
        
        submitted = st.form_submit_button("Add Goal")
        if submitted:
            # Add goal to sheet
            user_activity_df = read_activity_for_user(user['email'])
            
            # Get the starting value
            metric_col = map_goal_type_to_column(goal_type)
            start_value = user_activity_df[metric_col].iloc[-1] if not user_activity_df.empty and metric_col in user_activity_df.columns and not user_activity_df[metric_col].isnull().all() else 0
            
            new_goal_row = {
                "goal_id": str(uuid.uuid4()),
                "user_email": user['email'],
                "goal_type": goal_type,
                "start_value": start_value,
                "target_value": target_value,
                "target_date": target_date.isoformat(),
                "status": "Active",
                "created_date": datetime.date.today().isoformat(),
                "progress_pct": 0,
                "forecast_value": "",
                "forecast_success": ""
            }
            
            append_goal_row(new_goal_row)
            st.success("Goal added successfully!")
            st.rerun()

def show_insights_page(user):
    st.title("Personalized Insights")
    st.markdown("---")

    # Get user's activity data
    df_activity = read_activity_for_user(user['email'])
    if df_activity.empty:
        st.info("No activity data to generate insights from. Please sync your data first.")
        return

    # Basic data summary
    user_summary = {
        "most_recent_steps": df_activity['totalSteps'].iloc[-1] if 'totalSteps' in df_activity.columns and not df_activity['totalSteps'].isnull().all() else "N/A",
        "avg_sleep_hours": df_activity['sleepTimeHours'].mean() if 'sleepTimeHours' in df_activity.columns and not df_activity['sleepTimeHours'].isnull().all() else "N/A",
        "avg_resting_hr": df_activity['restingHeartRate'].mean() if 'restingHeartRate' in df_activity.columns and not df_activity['restingHeartRate'].isnull().all() else "N/A"
    }

    # Dummy cluster summary for now
    cluster_summary_text = "The user has shown a consistent trend of activity on weekdays and lower activity on weekends."

    # Get active goals
    goals_df = read_goals()
    user_goals = goals_df[goals_df['user_email'] == user['email']].to_dict('records')

    # Generate insights from LLM
    with st.spinner("Generating personalized insights from your data..."):
        insights_text = generate_llm_insights(user_summary, cluster_summary_text, user_goals)
        st.markdown(insights_text)

# ==========================================================
# Main app flow
# ==========================================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None

# Sidebar for login/logout and main navigation
st.sidebar.title("Account")
if not st.session_state.logged_in:
    with st.sidebar.form("login_form"):
        st.subheader("Login / Register")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password") # NOTE: This is NOT secure.
        role_choice = st.selectbox("I am a:", ["user", "coach"])
        login_button = st.form_submit_button("Login / Register")

        if login_button:
            if not email:
                st.error("Please enter your email.")
            else:
                users_df = read_users()
                user_record = users_df[users_df['email'] == email]
                if user_record.empty:
                    # New user, register them
                    append_user_row(email, role=role_choice)
                    user_profile = {"email": email, "role": role_choice, "linked_coach_email": "", "certified_coach": False}
                    st.session_state.user_profile = user_profile
                    st.session_state.logged_in = True
                    st.success(f"Registered {email} as {role_choice}. You are now logged in.")
                else:
                    # Existing user, log them in
                    user_profile = user_record.iloc[0].to_dict()
                    st.session_state.user_profile = user_profile
                    st.session_state.logged_in = True
                    st.success(f"Welcome back, {email}! You are now logged in.")
                st.rerun()

else:
    st.sidebar.success(f"Logged in as {st.session_state.user_profile['email']}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_profile = None
        st.rerun()
        
    st.sidebar.markdown("---")
    st.sidebar.header("Data Management")
    garmin_username = st.sidebar.text_input("Garmin Username", key="garmin_username")
    garmin_password = st.sidebar.text_input("Garmin Password", type="password", key="garmin_password")
    
    if st.sidebar.button("Sync Data from Garmin"):
        if garmin_username and garmin_password:
            try:
                with st.spinner("Connecting to Garmin..."):
                    api = Garmin(garmin_username, garmin_password)
                    api.login()
                
                # Call the fixed function with the API object
                sync_data_from_garmin(api, st.session_state.user_profile['email'])
            
            except Exception as e:
                st.error(f"Failed to authenticate with Garmin. Please check your credentials. Error: {e}")
                
        else:
            st.sidebar.warning("Please enter both Garmin username and password.")


# Main content
if st.session_state.logged_in:
    # Tab navigation
    tab_overview, tab_goals, tab_insights = st.tabs(["Dashboard", "My Goals", "Insights"])
    
    with tab_overview:
        show_overview_page(st.session_state.user_profile)
    
    with tab_goals:
        show_goals_page(st.session_state.user_profile)
    
    with tab_insights:
        show_insights_page(st.session_state.user_profile)
