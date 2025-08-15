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
from data_pre_processing import preprocessing_garmin_data  # updated: now accepts start_date and returns DataFrame only

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Garmin Coach Dashboard (Beta)")

GARMIN_HEADERS = [
        "Date","distance","duration","elapsedDuration","movingDuration","elevationGain","elevationLoss",
        "averageSpeed","maxSpeed","calories","bmrCalories","averageHR","maxHR","maxRunningCadenceInStepsPerMinute",
        "ActivitySteps","aerobicTrainingEffect","anaerobicTrainingEffect","vO2MaxValue","maxDoubleCadence",
        "locationName","lapCount","waterEstimated","trainingEffectLabel","activityTrainingLoad",
        "minActivityLapDuration","aerobicTrainingEffectMessage","anaerobicTrainingEffectMessage",
        "moderateIntensityMinutes_x","vigorousIntensityMinutes_x","hrTimeInZone_1","hrTimeInZone_2","hrTimeInZone_3",
        "hrTimeInZone_4","hrTimeInZone_5","totalSets","activeSets","totalReps","typeKey_clean","awakeSleepSeconds",
        "awakeCount","ageGroup","sleepScoreInsight","sleepTimeSeconds","sleepScoreFeedback","avgSleepStress",
        "deepSleepSeconds","remSleepSeconds","averageRespirationValue","lightSleepSeconds","charged","drained",
        "steps_y","pushes","totalKilocalories","activeKilocalories","bmrKilocalories","wellnessKilocalories",
        "remainingKilocalories","totalSteps","netCalorieGoal","totalDistanceMeters","wellnessDistanceMeters",
        "wellnessActiveKilocalories","netRemainingKilocalories","dailyStepGoal","highlyActiveSeconds",
        "activeSeconds","sedentarySeconds","sleepingSeconds","includesWellnessData","includesActivityData",
        "includesCalorieConsumedData","moderateIntensityMinutes_y","vigorousIntensityMinutes_y",
        "floorsAscendedInMeters","floorsDescendedInMeters","floorsAscended","floorsDescended","intensityMinutesGoal",
        "userFloorsAscendedGoal","minHeartRate","maxHeartRate","restingHeartRate","lastSevenDaysAvgRestingHeartRate",
        "averageStressLevel","maxStressLevel","stressDuration","restStressDuration","activityStressDuration",
        "uncategorizedStressDuration","totalStressDuration","lowStressDuration","mediumStressDuration","highStressDuration",
        "stressPercentage","restStressPercentage","activityStressPercentage","uncategorizedStressPercentage",
        "lowStressPercentage","mediumStressPercentage","highStressPercentage","stressQualifier",
        "measurableAwakeDuration","measurableAsleepDuration","minAvgHeartRate","maxAvgHeartRate",
        "bodyBatteryChargedValue","bodyBatteryDrainedValue","bodyBatteryHighestValue","bodyBatteryLowestValue",
        "bodyBatteryMostRecentValue","bodyBatteryDuringSleep","averageMonitoringEnvironmentAltitude",
        "restingCaloriesFromActivity","avgWakingRespirationValue","highestRespirationValue","lowestRespirationValue",
        "ActivityStartHour","ActivityPerformedToday","activityType","workout_date","last_workout_date",
        "daysSinceLastWorkout","sleepTimeHours","deepSleepHours","remSleepHours","lightSleepHours",
        "awakeSleepHours","deepSleepPercentage","remSleepPercentage","lightSleepPercentage","awakeSleepPercentage",
        "day_of_week","is_weekend","month","day_of_year","user_email"
    ]

# ==========================================================
# Google Sheets helpers (with caching to reduce API calls)
# ==========================================================

# Cache the Google Sheets client — resource-level (long-lived)
@st.cache_resource(show_spinner=False)
def get_gs_client():
    # Helpful auth validations
    required_key = "gcp_service_account"
    if required_key not in st.secrets:
        raise RuntimeError(
            "Missing Streamlit secret 'gcp_service_account'. "
            "Add your Google Service Account JSON (as a dict) under that key."
        )
    try:
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
        "ActivityData": ["user_email", "Date"]  # ActivityData will be appended with many columns; header minimal
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
    rindex = df[mask].index[0] + 2  # +1 for header and +1 for 0-index
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
        recent_mean = df_activity[metric_col].dropna().tail(7).mean()
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
        pred = float(last_row[metric_col])
    return float(pred), model

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
        # add more mappings as needed
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
        import google.generativeai as genai
        gemini_api_key = st.secrets.get("gemini_api_key", None)
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
        model = genai.GenerativeModel('gemini-2.5-pro')
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"LLM call failed or not configured: {e}"

# ==========================================================
# UI: Login / Registration
# ==========================================================
def login_panel():
    st.sidebar.title("Sign in / Register")
    email = st.sidebar.text_input("Email", value="", placeholder="your@email.com")
    role_choice = st.sidebar.selectbox("Register as (if new)", ["user", "coach"])
    btn = st.sidebar.button("Login / Register")
    if btn:
        if not email:
            st.sidebar.error("Please enter an email")
            return None
        users = read_users()
        if not users.empty and (users['email'] == email).any():
            user_row = users[users['email'] == email].iloc[0].to_dict()
            st.sidebar.success(f"Welcome back: {email} ({user_row.get('role')})")
            return user_row
        else:
            append_user_row(email, role=role_choice)
            st.sidebar.success(f"Registered {email} as {role_choice}. Re-click login to load profile.")
            return {"email": email, "role": role_choice, "linked_coach_email": "", "certified_coach": False}
    return None
    
    # After login, the login form will disappear and the fetch data button will appear
    if st.session_state.logged_in:
        st.sidebar.button("Fetch Data")
        # Add your fetch data logic here

# ==========================================================
# UI: Core pages and helpers
# ==========================================================
def show_overview_page(user):
    st.title("My Dashboard")
    st.subheader("Overview")
    df = read_activity_for_user(user['email'])
    if df.empty:
        st.info("No synced activity data found. Use 'Fetch Data' in the left sidebar to pull from Garmin.")
        return
    # KPIs
    kpi_cols = []
    if 'totalSteps' in df.columns:
        kpi_cols.append(("Avg Daily Steps", df['totalSteps'].mean()))
    if 'restingHeartRate' in df.columns:
        kpi_cols.append(("Avg Resting HR", df['restingHeartRate'].mean()))
    if 'sleepTimeHours' in df.columns:
        kpi_cols.append(("Avg Sleep (hrs)", df['sleepTimeHours'].mean()))
    cols = st.columns(len(kpi_cols) if kpi_cols else 1)
    for i, (label, val) in enumerate(kpi_cols):
        with cols[i]:
            st.metric(label=label, value=f"{val:.1f}" if not math.isnan(val) else "N/A")
    st.markdown("### Recent data preview")
    st.dataframe(df.sort_values("Date", ascending=False).head(10))

def show_activity_trends_page(user):
    st.title("Activity Trends")
    df = read_activity_for_user(user['email'])
    if df.empty:
        st.info("No activity data available. Sync data first.")
        return
    col1, col2 = st.columns(2)
    with col1:
        if 'restingHeartRate' in df.columns:
            fig = px.line(df.sort_values("Date"), x="Date", y="restingHeartRate", title="Resting Heart Rate")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("restingHeartRate column missing")
    with col2:
        if 'totalSteps' in df.columns:
            fig = px.bar(df.sort_values("Date"), x="Date", y="totalSteps",
                         title="Daily Steps",
                         color='ActivityPerformedToday' if 'ActivityPerformedToday' in df.columns else None)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("totalSteps column missing")

def show_sleep_analysis_page(user):
    st.title("Sleep Analysis")
    df = read_activity_for_user(user['email'])
    if df.empty:
        st.info("No activity data available. Sync data first.")
        return
    if all(c in df.columns for c in ['sleepTimeHours', 'deepSleepHours', 'Date']):
        df_sleep = df[['Date','sleepTimeHours','deepSleepHours']].dropna().copy()
        df_sleep['deepPerc'] = (df_sleep['deepSleepHours'] / df_sleep['sleepTimeHours']).clip(0,1)*100
        avg_by_dow = df_sleep.copy()
        avg_by_dow['dow'] = avg_by_dow['Date'].dt.day_name()
        avg = (avg_by_dow
               .groupby('dow')[['sleepTimeHours','deepSleepHours']]
               .mean()
               .reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
               .fillna(0))
        avg['variance_score'] = (avg['sleepTimeHours'] - avg['deepSleepHours']).abs()
        fig = px.bar(avg.reset_index(), x='dow', y='variance_score', title="Sleep variance by weekday")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.scatter(df_sleep, x='Date', y='sleepTimeHours', trendline='ols', title="Sleep Time Trend")
        st.plotly_chart(fig2, use_container_width=True)
        fig3 = px.scatter(df_sleep, x='Date', y='deepPerc', trendline='ols', title="Deep Sleep % Trend")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Required sleep columns are missing")

def show_insights_page(user):
    st.title("Insights")
    df = read_activity_for_user(user['email'])
    goals_df = read_goals()
    user_goals = goals_df[goals_df['user_email'] == user['email']].copy() if not goals_df.empty else pd.DataFrame()
    # Goals summary table + progress bar chart
    st.subheader("Goals Summary")
    if user_goals.empty:
        st.info("No active goals. Use the Goals page to add one.")
    else:
        # cast types
        user_goals['progress_pct'] = pd.to_numeric(user_goals['progress_pct'], errors='coerce').fillna(0)
        user_goals['forecast_value'] = pd.to_numeric(user_goals['forecast_value'], errors='coerce').fillna(np.nan)
        table_show = user_goals[['goal_id','goal_type','start_value','target_value','target_date','status','progress_pct','forecast_value','forecast_success']].copy()
        st.dataframe(table_show)
        fig = px.bar(table_show.sort_values('progress_pct', ascending=False),
                     x='goal_type', y='progress_pct', color='forecast_success', title="Goal progress %")
        st.plotly_chart(fig, use_container_width=True)
    # Forecast vs target charts
    st.subheader("Forecast vs Target")
    if (not user_goals.empty) and (not df.empty):
        for _, g in user_goals.iterrows():
            metric_col = map_goal_type_to_column(g['goal_type'])
            st.markdown(f"**{g['goal_type']}** target: {g['target_value']} by {g['target_date']}")
            if metric_col and metric_col in df.columns:
                hist = df[['Date', metric_col]].dropna().sort_values('Date')
                if hist.empty:
                    st.info("No historical values for this metric.")
                    continue
                forecast_val, model = forecast_metric_on_date(hist, metric_col, g['target_date'])
                last_date = hist['Date'].max()
                last_val = hist[metric_col].iloc[-1]
                target_dt = pd.to_datetime(g['target_date'])
                proj_dates = pd.date_range(start=last_date, end=target_dt, freq='D')
                proj_vals = np.linspace(last_val, forecast_val, len(proj_dates))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist['Date'], y=hist[metric_col], mode='lines+markers', name='history'))
                fig.add_trace(go.Scatter(x=proj_dates, y=proj_vals, mode='lines', name='forecast'))
                fig.add_hline(y=float(g['target_value']), line_dash='dash', annotation_text='target', annotation_position='top left')
                fig.update_layout(title=f"Forecast vs Target for {g['goal_type']}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No data mapped for goal type {g['goal_type']}.")
    # Rolling trends
    st.subheader("Rolling 7-day trends")
    if not df.empty:
        to_plot = []
        if 'totalSteps' in df.columns: to_plot.append(('totalSteps','Daily steps'))
        if 'restingHeartRate' in df.columns: to_plot.append(('restingHeartRate','Resting HR'))
        if 'sleepTimeHours' in df.columns: to_plot.append(('sleepTimeHours','Sleep hours'))
        for col, title in to_plot:
            tmp = df[['Date', col]].dropna().sort_values('Date')
            tmp['rolling7'] = tmp[col].rolling(7, min_periods=1).mean()
            fig = px.line(tmp, x='Date', y='rolling7', title=f"{title} (7-day rolling avg)")
            st.plotly_chart(fig, use_container_width=True)
    # LLM summary
    st.subheader("AI Assistant Summary")
    summary_dict = {
        "days_tracked": len(df),
        "avg_steps": float(df['totalSteps'].mean()) if 'totalSteps' in df.columns and not df['totalSteps'].isnull().all() else None,
        "avg_resting_hr": float(df['restingHeartRate'].mean()) if 'restingHeartRate' in df.columns and not df['restingHeartRate'].isnull().all() else None,
        "avg_sleep_hours": float(df['sleepTimeHours'].mean()) if 'sleepTimeHours' in df.columns and not df['sleepTimeHours'].isnull().all() else None
    }
    # create cluster summary text placeholder (you could add kmeans clusters here like before)
    cluster_summary_text = "Daily segments not computed (or not available)."
    # prepare goals list for prompt
    goals_list = []
    if not user_goals.empty:
        for _, r in user_goals.iterrows():
            goals_list.append({
                "goal_type": r['goal_type'],
                "start_value": r.get('start_value'),
                "target_value": r.get('target_value'),
                "target_date": r.get('target_date'),
                "progress_pct": r.get('progress_pct'),
                "forecast_value": r.get('forecast_value'),
                "forecast_success": r.get('forecast_success')
            })
    llm_text = generate_llm_insights(summary_dict, cluster_summary_text, goals_list, viewer_role="user")
    st.markdown(llm_text)

def show_goals_page(user):
    st.title("Goals")
    st.subheader("Create a new goal")
    with st.form("create_goal"):
        goal_type = st.selectbox("Goal Type", ["Daily Steps", "VO2 Max", "Resting HR", "Sleep Hours", "Deep Sleep Percentage"])
        target_value = st.text_input("Target Value (numeric)", "")
        target_date = st.date_input("Target Date", value=(date.today() + timedelta(days=30)))
        description = st.text_area("Notes / Description (optional)", "")
        submitted = st.form_submit_button("Create Goal")
        if submitted:
            if not target_value or not target_value.replace('.','',1).isdigit():
                st.error("Please supply a numeric target value.")
            else:
                # get start value from most recent data
                metric_col = map_goal_type_to_column(goal_type)
                df = read_activity_for_user(user['email'])
                start_val = None
                if metric_col and (not df.empty) and metric_col in df.columns:
                    start_val = df[metric_col].dropna().iloc[-1]
                else:
                    start_val = ""
                # build goal row
                goal_id = str(uuid.uuid4())[:8]
                goal_row = {
                    "goal_id": goal_id,
                    "user_email": user['email'],
                    "goal_type": goal_type,
                    "start_value": start_val,
                    "target_value": float(target_value),
                    "target_date": target_date.isoformat(),
                    "status": "in-progress",
                    "created_date": date.today().isoformat(),
                    "progress_pct": 0,
                    "forecast_value": "",
                    "forecast_success": ""
                }
                append_goal_row(goal_row)
                st.success("Goal created. Forecast and progress will update after next data sync.")
    st.markdown("---")
    st.subheader("Your active goals")
    goals_df = read_goals()
    user_goals = goals_df[goals_df['user_email'] == user['email']].copy() if not goals_df.empty else pd.DataFrame()
    if user_goals.empty:
        st.info("No goals yet.")
    else:
        st.dataframe(user_goals.sort_values("created_date", ascending=False))

def show_coach_dashboard(coach_user):
    st.title("Coach Dashboard")
    st.subheader("Athlete Overview")
    users = read_users()
    athletes = users[users['linked_coach_email'] == coach_user['email']].copy() if not users.empty else pd.DataFrame()
    if athletes.empty:
        st.info("You have not invited any athletes. Use 'Invite Athlete' below.")
    else:
        # show athlete list
        st.dataframe(athletes[['email', 'date_joined']])
    st.markdown("---")
    st.subheader("Invite Athlete")
    with st.form("invite_athlete"):
        athlete_email = st.text_input("Athlete Email")
        invite_note = st.text_area("Optional note to athlete (not emailed by app currently)")
        submitted = st.form_submit_button("Invite")
        if submitted:
            if not athlete_email:
                st.error("Please enter athlete email")
            else:
                # create user row if not exists with linked_coach_email set
                users = read_users()
                if (not users.empty) and (users['email'] == athlete_email).any():
                    st.warning("This athlete already exists in Users tab. Updating linked_coach_email.")
                    append_user_row(athlete_email, role="user", linked_coach_email=coach_user['email'])
                else:
                    append_user_row(athlete_email, role="user", linked_coach_email=coach_user['email'])
                st.success(f"Invited {athlete_email}. They will need to sign in and sync Garmin data.")

# ==========================================================
# Sidebar: Garmin Fetch & Login
# ==========================================================
st.sidebar.header("Garmin & Account")

# Initialize session state for user profile and Garmin credentials
if 'user_profile' not in st.session_state:
    st.session_state['user_profile'] = None
if 'garmin_uname' not in st.session_state:
    st.session_state['garmin_uname'] = ''
if 'garmin_pwd' not in st.session_state:
    st.session_state['garmin_pwd'] = ''

# --- Login Panel ---
if not st.session_state['user_profile']:
    st.session_state['user_profile'] = login_panel()

user_profile = st.session_state['user_profile']
def _compute_fetch_start_date_for_user(user_email: str) -> date:
    """
    If user has ActivityData rows:
        start_date = last Date + 1
    Else:
        start_date = 2024-01-01
    """
    try:
        df_user = read_activity_for_user(user_email)
        if df_user.empty or 'Date' not in df_user.columns or df_user['Date'].dropna().empty:
            return pd.to_datetime("2024-01-01").date()
        last_dt = pd.to_datetime(df_user['Date'].max()).date()
        return (last_dt + timedelta(days=1))
    except Exception:
        return pd.to_datetime("2024-01-01").date()

# ============================
# Garmin fetch (if logged in)
# ============================
if user_profile:
    st.sidebar.markdown(f"Signed in as: **{user_profile['email']}** ({user_profile['role']})")
    st.sidebar.markdown("Sync your Garmin data (writes to ActivityData sheet)")
    # Initialize session state
    if 'garmin_uname' not in st.session_state:
        st.session_state['garmin_uname'] = ''
    if 'garmin_pwd' not in st.session_state:
        st.session_state['garmin_pwd'] = ''
    
    # Username input
    st.session_state['garmin_uname'] = st.sidebar.text_input(
        "Garmin Username (email)", 
        value=st.session_state['garmin_uname']
    )
    
    # Password input — do NOT use value= to avoid clearing on focus
    pwd_input = st.sidebar.text_input(
        "Garmin Password", 
        type="password"
    )
    if pwd_input:  # only update session_state if user typed something
        st.session_state['garmin_pwd'] = pwd_input
    
    # Use stored credentials
    uname = st.session_state['garmin_uname']
    pwd = st.session_state['garmin_pwd']


    # Show computed fetch range for transparency
    intended_start_date = _compute_fetch_start_date_for_user(user_profile['email'])
    today_date = date.today()
    st.sidebar.caption(f"Planned fetch range: {intended_start_date.isoformat()} → {today_date.isoformat()}")

    if st.sidebar.button("Fetch Data from Garmin"):
        if not uname or not pwd:
            st.sidebar.error("Enter Garmin credentials to fetch.")
        else:
            # Guard: if we're already up to date, skip
            if intended_start_date > today_date:
                st.sidebar.info("Already up to date — no new days to fetch.")
            else:
                with st.spinner("Fetching and preprocessing Garmin data (this can take a minute)..."):
                    try:
                        # Pass start_date to preprocessing; it returns a prepped DataFrame only
                        df = preprocessing_garmin_data(uname, pwd, start_date=intended_start_date, end_date=date.today(),
                                                      headers=GARMIN_HEADERS)
                        if df is None or df.empty:
                            st.warning("No new Garmin data returned for the requested window.")
                        else:
                            # attach user email and write to ActivityData
                            append_activity_data(df, user_profile['email'])
                            st.success(f"Fetched {df.shape[0]} rows and appended to ActivityData.")
                    except Exception as e:
                        st.error(f"Error fetching Garmin data: {e}")

# If not logged in, stop further UI
if not user_profile:
    st.title("Garmin Coach Dashboard")
    st.markdown("Please sign in or register via the left sidebar.")
    st.stop()

# ==========================================================
# Main Navigation
# ==========================================================
menu_options = ["My Dashboard", "Insights", "Activity Trends", "Sleep Analysis", "Goals", "Settings"]
if user_profile.get('role') == 'coach':
    menu_options.insert(1, "Coach Dashboard")

choice = st.sidebar.selectbox("Go to", menu_options)

# Render pages
if choice == "My Dashboard":
    show_overview_page(user_profile)
elif choice == "Insights":
    show_insights_page(user_profile)
elif choice == "Activity Trends":
    show_activity_trends_page(user_profile)
elif choice == "Sleep Analysis":
    show_sleep_analysis_page(user_profile)
elif choice == "Goals":
    show_goals_page(user_profile)
elif choice == "Coach Dashboard":
    if user_profile.get('role') == 'coach':
        show_coach_dashboard(user_profile)
    else:
        st.error("Coach Dashboard accessible only to users with role='coach'")
elif choice == "Settings":
    st.title("Settings")
    st.write("Account email:", user_profile['email'])
    st.write("Role:", user_profile['role'])
    st.write("Linked coach:", user_profile.get('linked_coach_email', 'None'))
    st.info("To change roles or certified status edit the Users sheet directly or extend this page to support admin actions.")

# ==========================================================
# Background: update forecasts for goals if ActivityData exists
# ==========================================================
def recompute_all_goal_forecasts():
    goals_df = read_goals()
    if goals_df.empty:
        return
    for idx, g in goals_df.iterrows():
        try:
            # skip if already completed
            if str(g.get('status','')).lower() in ("completed","done"):
                continue
            metric_col = map_goal_type_to_column(g['goal_type'])
            if not metric_col:
                continue
            user_email = g['user_email']
            df_user = read_activity_for_user(user_email)
            if df_user.empty or metric_col not in df_user.columns:
                continue
            # forecast
            forecast_val, model = forecast_metric_on_date(
                df_user[['Date', metric_col]].rename(columns={metric_col: metric_col}),
                metric_col,
                g['target_date']
            )
            # compute progress: if start_value exists, use it. else pick earliest value available
            try:
                start_value = float(g['start_value']) if g.get('start_value') not in (None, "", "nan") else float(df_user[metric_col].dropna().iloc[0])
            except Exception:
                start_value = float(df_user[metric_col].dropna().iloc[0]) if not df_user[metric_col].dropna().empty else None
            try:
                target_value = float(g['target_value'])
            except Exception:
                target_value = None
            progress_pct = ""
            forecast_success = ""
            if target_value is not None and start_value is not None:
                # direction: for some metrics higher is better (vo2, steps); for RHR lower is better
                higher_is_better = True
                if str(g['goal_type']).lower() in ("resting hr", "rhr"):
                    higher_is_better = False
                # compute current progress relative to start->target
                cur_val = float(df_user[metric_col].dropna().iloc[-1]) if not df_user[metric_col].dropna().empty else start_value
                if higher_is_better:
                    denom = (target_value - start_value) if (target_value - start_value) != 0 else 1e-6
                    progress_pct = max(0.0, min(100.0, (cur_val - start_value) / denom * 100.0))
                    forecast_success = bool(forecast_val >= target_value)
                else:
                    denom = (start_value - target_value) if (start_value - target_value) != 0 else 1e-6
                    progress_pct = max(0.0, min(100.0, (start_value - cur_val) / denom * 100.0))
                    forecast_success = bool(forecast_val <= target_value)
            # write back updates
            updates = {
                "forecast_value": float(forecast_val) if forecast_val is not None else "",
                "progress_pct": round(float(progress_pct), 2) if progress_pct != "" else "",
                "forecast_success": str(forecast_success)
            }
            update_goal_row(g['goal_id'], updates)
        except Exception:
            # skip problematic goals
            continue

# recompute forecasts for all goals on app load (lightweight)
try:
    recompute_all_goal_forecasts()
except Exception:
    pass

# End of app.py
