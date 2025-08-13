def preprocessing_garmin_data(username, password, last_login_date=None):
    from garminconnect import Garmin
    import datetime
    from datetime import date, timedelta
    import pandas as pd
    import logging
    import seaborn as sns
    import numpy as np

    # Logging setup
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"garmin_data_script_{timestamp_str}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    # Login to Garmin
    try:
        api = Garmin(username, password)
        logger.info("Attempting to log in to Garmin Connect...")
        api.login()
        logger.info("Login successful.")
    except Exception as e:
        logger.critical(f"A critical error occurred during script execution: {e}", exc_info=True)
        return pd.DataFrame()

    # Determine date range
    today = date.today()
    if last_login_date:
        start_date_obj = pd.to_datetime(last_login_date).date() + timedelta(days=1)
    else:
        start_date_obj = date(2024, 1, 1)  # First-time user case

    st_dt = start_date_obj.isoformat()
    end_dt = today.isoformat()

    logger.info(f"Fetching data from {st_dt} to {end_dt}")

    def get_iso_date_range(start_date_str, end_date_str):
        start_date = date.fromisoformat(start_date_str)
        end_date = date.fromisoformat(end_date_str)
        if start_date > end_date:
            return []
        return [(start_date + timedelta(days=i)).isoformat()
                for i in range((end_date - start_date).days + 1)]

    date_range = get_iso_date_range(st_dt, end_dt)

    if not date_range:
        logger.info("No new dates to process.")
        return pd.DataFrame()

    # === Existing data pull & cleaning logic ===
    # Keeping your full original cleaning code here, no changes except using the computed date_range
    # -----------------------------
    # (Paste your existing processing code here unchanged except for start/end dates)
    # -----------------------------

    # For demonstration, let's pretend total_df is your final combined dataframe
    total_df = pd.DataFrame({"Date": date_range})  # Replace with your processed df

    # === Align columns with ActivityData worksheet ===
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials

        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name("config.json", scope)
        client = gspread.authorize(creds)

        ws = client.open("HealthData").worksheet("ActivityData")
        activity_columns = ws.row_values(1)

        for col in activity_columns:
            if col not in total_df.columns:
                total_df[col] = np.nan

        total_df = total_df[activity_columns]

    except Exception as e:
        logger.error(f"Error aligning columns with ActivityData worksheet: {e}")

    return total_df
