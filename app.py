# app.py
"""
Streamlit Garmin Dashboard (role-based with coach invites, Goals, ML forecasting, LLM insights)
Requirements: streamlit, pandas, numpy, plotly, gspread, gspread_dataframe, scikit-learn, xgboost, google-generative-ai,
              garminconnect, python-dateutil
Make sure to put sensitive keys in Streamlit secrets:
- gcp_service_account: { ... }  (service account JSON as dict)
- gemini_api_key: "YOUR_GEMINI_KEY"
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
from google.api_core.exceptions import NotFound
from data_pre_processing import preprocessing_garmin_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
import uuid
import io
import math

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Garmin Coach Dashboard (Beta)")

#
# ---------------------------
# Helpers: Google Sheets utils
# ---------------------------
#
def get_gs_client():
    try:
        creds = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(creds)
        return gc
    except Exception as e:
        st.error("Google Sheets auth failed. Add service account JSON to Streamlit secrets as 'gcp_service_account'.")
        st.stop()

def ensure_sheet_and_tabs(spreadsheet_name="Garmin_User_Data"):
    """Open spreadsheet and create required worksheets if they do not exist."""
    gc = get_gs_client()
    try:
        sh = gc.open(spreadsheet_name)
    except Exception:
        # Create new spreadsheet
        sh = gc.create(spreadsheet_name)
    # Ensure worksheets exist
    required = {
        "Users": ["email", "role", "linked_coach_email", "certified_coach", "date_joined"],
        "Goals": ["goal_id", "user_email", "goal_type", "start_value", "target_value", "target_date", "status", "created_date", "progress_pct", "forecast_value", "forecast_success"],
        "ActivityData": ["user_email", "Date"]  # ActivityData will be appended with many columns; header minimal
    }
    for ws_name, headers in required.items():
        try:
            ws = sh.worksheet(ws_name)
            # ensure headers exist -- if first row is empty write headers
            vals = ws.row_values(1)
            if not vals or len(vals) < len(headers):
                ws.delete_rows(1)
                ws.insert_row(headers, index=1)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=ws_name, rows=1000, cols=20)
            ws.insert_row(headers, index=1)
    return sh

def read_users(spreadsheet_name="Garmin_User_Data"):
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("Users")
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    # If empty, return empty df with columns
    if df.empty:
        df = pd.DataFrame(columns=["email", "role", "linked_coach_email", "certified_coach", "date_joined"])
    return df

def append_user_row(email, role="user", linked_coach_email="", certified_coach=False, spreadsheet_name="Garmin_User_Data"):
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("Users")
    row = [email, role, linked_coach_email, str(certified_coach).upper(), datetime.date.today().isoformat()]
    ws.append_row(row)

def append_activity_data(df_activity: pd.DataFrame, user_email, spreadsheet_name="Garmin_User_Data"):
    # Write activity data to ActivityData sheet; include user_email column
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("ActivityData")
    df = df_activity.copy()
    df.insert(0, "user_email", user_email)
    # get existing to determine where to place
    existing = ws.get_all_values()
    if not existing or len(existing) < 1:
        set_with_dataframe(ws, df, row=1, include_column_header=True)
    else:
        set_with_dataframe(ws, df, row=len(existing) + 1, include_column_header=False)

def read_activity_for_user(user_email, spreadsheet_name="Garmin_User_Data"):
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("ActivityData")
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    df = df[df['user_email'] == user_email].copy()
    # ensure Date exists and parse
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

def read_goals(spreadsheet_name="Garmin_User_Data"):
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("Goals")
    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        df = pd.DataFrame(columns=["goal_id", "user_email", "goal_type", "start_value", "target_value", "target_date", "status", "created_date", "progress_pct", "forecast_value", "forecast_success"])
    return df

def append_goal_row(goal_row: dict, spreadsheet_name="Garmin_User_Data"):
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("Goals")
    # append in the order of headers present in first row
    headers = ws.row_values(1)
    row = [goal_row.get(h, "") for h in headers]
    ws.append_row(row)

def update_goal_row(goal_id, updates: dict, spreadsheet_name="Garmin_User_Data"):
    sh = ensure_sheet_and_tabs(spreadsheet_name)
    ws = sh.worksheet("Goals")
    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        return False
    mask = df['goal_id'] == goal_id
    if not mask.any():
        return False
    rindex = df[mask].index[0] + 2  # gspread 1-based + header
    # update each column
    headers = ws.row_values(1)
    for k, v in updates.items():
        if k in headers:
            col_idx = headers.index(k) + 1
            ws.update_cell(rindex, col_idx, str(v))
    return True

#
# ---------------------------
# Authentication (Email-based)
# ---------------------------
#
def login_ui():
    st.sidebar.header("Sign in / Register")
    email = st.sidebar.text_input("Email", value="", placeholder="you@example.com")
    role_choice = st.sidebar.selectbox("If new, register as", ["user", "coach"])
    if st.sidebar.button("Login / Register"):
        if not email:
            st.sidebar.error("Please enter an email.")
            return None
        users_df = read_users()
        if (users_df['email'] == email).any():
            user_row = users_df[users_df['email'] == email].iloc[0].to_dict()
            st.sidebar.success(f"Welcome back, {email} ({user_row['role']})")
            return user_row
        else:
            # register new user
            append_user_row(email, role=role_choice, linked_coach_email="", certified_coach=False)
            st.sidebar.success(f"Registered {email} as {role_choice}. Re-click login to load profile.")
            return {"email": email, "role": role_choice, "linked_coach_email": "", "certified_coach": False}
    return None

#
# ---------------------------
# Forecasting utilities
# ---------------------------
#
def prepare_feature_df_for_metric(df_activity, metric_col, n_lags=7):
    """
    Given activity dataframe (Date indexed) and metric_col string,
    create a feature table: day_index, value, rolling_mean_7, rolling_std_7, day_of_week, is_weekend
    """
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

@st.cache_data(ttl=3600)
def train_xgb_regressor(X, y):
    try:
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
        model.fit(X, y)
        return model
    except Exception as e:
        # fallback to simple linear regression
        lr = LinearRegression()
        lr.fit(X, y)
        return lr

def forecast_metric_on_date(df_activity, metric_col, target_date):
    """
    Train a model on historical values and return forecasted value on the target_date.
    Returns (forecast_value, model_used)
    """
    if df_activity.empty or metric_col not in df_activity.columns:
        return None, None
    feat_df = prepare_feature_df_for_metric(df_activity, metric_col)
    if feat_df.empty or len(feat_df) < 5:
        # Not enough data — use simple linear extrapolation of last 7-day mean
        recent_mean = df_activity[metric_col].dropna().tail(7).mean()
        return float(recent_mean), "mean_fallback"
    # Target day index to predict
    min_date = feat_df['Date'].min()
    target_day_idx = (pd.to_datetime(target_date) - min_date).days
    # train features
    X = feat_df[['day_idx', 'rolling_mean_7', 'rolling_std_7', 'day_of_week', 'is_weekend']].fillna(0)
    y = feat_df[metric_col]
    model = train_xgb_regressor(X, y)
    # Build input row for target
    last_row = feat_df.iloc[-1]
    delta_days = target_day_idx - last_row['day_idx']
    # naive approach: compute rolling mean shift; for features use last rolling_mean and set day_idx accordingly
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

#
# ---------------------------
# LLM integration: Gemini (google.generativeai)
# ---------------------------
#
def generate_llm_insight(summary_dict, cluster_summary_text, goals_list, viewer_role="user"):
    try:
        import google.generativeai as genai
        gemini_api_key = st.secrets.get("gemini_api_key", None)
        if not gemini_api_key:
            return "LLM not configured. Set gemini_api_key in Streamlit secrets to enable AI insights."
        genai.configure(api_key=gemini_api_key)
        prompt_parts = []
        prompt_parts.append("You are a friendly, concise data-driven coach. Analyze the user's wearable data and goals and provide:")
        prompt_parts.append("1) Short summary (2-3 lines). 2) Which goals are on track vs off-track. 3) One prioritized action for this week. 4) If viewer is a coach, include a short note coaches can use to message the athlete.")
        prompt_parts.append(f"User summary stats: {summary_dict}")
        prompt_parts.append(f"Cluster summary (daily segments): {cluster_summary_text}")
        prompt_parts.append("Active goals and forecast (format: goal_type | start_value | target_value | target_date | progress_pct | forecast_value | forecast_success):")
        for g in goals_list:
            prompt_parts.append(str(g))
        tone = "Motivational and succinct. Use emojis and a short title."
        if viewer_role == "coach":
            tone += " Provide 'coach tips' (2 short bullet points) that the coach can use to advise the athlete."
        prompt = "\n\n".join([*prompt_parts, tone])
        model = genai.GenerativeModel('gemini-2.5-pro')
        res = model.generate_content(prompt)
        return res.text
    except Exception as e:
        return f"LLM error or not configured: {e}"

#
# ---------------------------
# UI Pages & Components
# ---------------------------
#
def show_overview(user):
    st.header("Overview")
    st.subheader("Quick Stats")
    df = read_activity_for_user(user['email'])
    if df.empty:
        st.info("No ActivityData found for this user. Go to 'Fetch Data' to sync from Garmin.")
        return
    # Key metrics
    metrics = {}
    for col_key, label in [
        ("totalSteps", "Avg Daily Steps"),
        ("restingHeartRate", "Avg Resting HR (bpm)"),
        ("sleepTimeHours", "Avg Sleep (hrs)")
    ]:
        if col_key in df.columns:
            metrics[label] = df[col_key].mean()
    cols = st.columns(len(metrics) or 1)
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(label=label, value=f"{value:.1f}" if not math.isnan(value) else "N/A")
    st.markdown("#### Recent data preview")
    st.dataframe(df.sort_values("Date", ascending=False).head(10))

def show_activity_trends(user):
    st.header("Activity Trends")
    df = read_activity_for_user(user['email'])
    if df.empty:
        st.info("No ActivityData found for this user. Sync with Garmin to see trends.")
        return
    col1, col2 = st.columns(2)
    with col1:
        if 'restingHeartRate' in df.columns:
            fig = px.line(df.sort_values("Date"), x="Date", y="restingHeartRate", title="Resting Heart Rate Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("restingHeartRate not available.")
    with col2:
        if 'totalSteps' in df.columns:
            fig = px.bar(df.sort_values("Date"), x="Date", y="totalSteps", title="Daily Steps (colored by activity presence)", color='ActivityPerformedToday' if 'ActivityPerformedToday' in df.columns else None)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("totalSteps not available.")

def show_sleep_analysis(user):
    st.header("Sleep Analysis")
    df = read_activity_for_user(user['email'])
    if df.empty:
        st.info("No ActivityData found for this user. Sync with Garmin to see sleep insights.")
        return
    if all(c in df.columns for c in ['sleepTimeHours', 'deepSleepHours', 'Date']):
        df_sleep = df[['Date', 'sleepTimeHours', 'deepSleepHours']].dropna().copy()
        df_sleep['deepPerc'] = (df_sleep['deepSleepHours'] / df_sleep['sleepTimeHours']).clip(0,1)*100
        avg_by_dow = df_sleep.copy()
        avg_by_dow['dow'] = avg_by_dow['Date'].dt.day_name()
        avg = avg_by_dow.groupby('dow')[['sleepTimeHours','deepSleepHours']].mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).fillna(0)
        avg['variance_score'] = (avg['sleepTimeHours'] - avg['deepSleepHours']).abs()
        fig = px.bar(avg.reset_index(), x='dow', y='variance_score', title="Variance between Total Sleep and Deep Sleep (by weekday)")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.scatter(df_sleep, x='Date', y='sleepTimeHours', trendline='ols', title="Sleep Time Trend")
        st.plotly_chart(fig2, use_container_width=True)
        fig3 = px.scatter(df_sleep, x='Date', y='deepPerc', trendline='ols', title="Deep Sleep % Trend")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("sleepTimeHours or deepSleepHours columns are not available.")

def show_insights(user):
    st.header("Insights")
    df = read_activity_for_user(user['email'])
    goals_df = read_goals()
    user_goals = goals_df[goals_df['user_email'] == user['email']] if not goals_df.empty else pd.DataFrame()
    # Section: Goals summary
    st.subheader("Goals Summary")
    if user_goals.empty:
        st.info("No active goals. Go to the Goals page to add one.")
    else:
        display = user_goals[['goal_id','goal_type','start_value','target_value','target_date','status','progress_pct','forecast_value','forecast_success']].copy()
        st.dataframe(display)
        # Bar chart of progress
        fig = px.bar(display.sort_values("progress_pct", ascending=False), x='goal_type', y='progress_pct', color='forecast_success', title="Goal progress %")
        st.plotly_chart(fig, use_container_width=True)
    # Section: Forecast vs Target charts for each goal
    st.subheader("Forecast vs Target")
    if not user_goals.empty and not df.empty:
        for _, g in user_goals.iterrows():
            try:
                metric = map_goal_type_to_column(g['goal_type'])
                if metric and metric in df.columns:
                    st.markdown(f"**{g['goal_type']}** — target: {g['target_value']} by {g['target_date']}")
                    historical = df[['Date', metric]].dropna().sort_values("Date")
                    if historical.empty:
                        st.info("No historical values for this metric.")
                        continue
                    forecast_val, _ = forecast_metric_on_date(historical, metric, g['target_date'])
                    # line chart with history, forecast line to target date, and horizontal target line
                    hist_trace = go.Scatter(x=historical['Date'], y=historical[metric], mode='lines+markers', name='history')
                    last_date = historical['Date'].max()
                    last_val = historical[metric].iloc[-1]
                    target_dt = pd.to_datetime(g['target_date'])
                    # create simple projection line
                    projection_dates = pd.date_range(start=last_date, end=target_dt, freq='D')
                    if len(projection_dates) > 1:
                        proj_vals = np.linspace(last_val, forecast_val, len(projection_dates))
                        proj_trace = go.Scatter(x=projection_dates, y=proj_vals, mode='lines', name='forecast')
                        target_trace = go.Scatter(x=[historical['Date'].min(), historical['Date'].max(), target_dt], y=[g['target_value'], g['target_value'], g['target_value']], mode='lines', name='target', line=dict(dash='dash'))
                        fig = go.Figure([hist_trace, proj_trace, target_trace])
                        fig.update_layout(title=f"Forecast vs Target for {g['goal_type']}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No mapped metric for goal type {g['goal_type']} or data not present.")
            except Exception as e:
                st.error(f"Error plotting goal forecast: {e}")
    # Section: Rolling trends
    st.subheader("Rolling Trends (7-day)")
    if not df.empty:
        metrics_to_plot = []
        if 'totalSteps' in df.columns:
            metrics_to_plot.append(('totalSteps', 'Daily Steps'))
        if 'restingHeartRate' in df.columns:
            metrics_to_plot.append(('restingHeartRate', 'Resting HR'))
        if 'sleepTimeHours' in df.columns:
            metrics_to_plot.append(('sleepTimeHours', 'Sleep Hours'))
        for col, title in metrics_to_plot:
            tmp = df[['Date', col]].dropna().sort_values("Date")
            if tmp.empty:
                continue
            tmp['rolling7'] = tmp[col].rolling(7, min_periods=1).mean()
            fig = px.line(tmp, x='Date', y='rolling7', title=f"{title} — 7-day rolling average")
            st.plotly_chart(fig, use_container_width=True)
    # Section: AI LLM summary
    st.subheader("AI Assistant Summary")
    summary_dict = {
        "days_tracked": len(df),
        "avg_steps": float(df['totalSteps'].mean()) if 'total
