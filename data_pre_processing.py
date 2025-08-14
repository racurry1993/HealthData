def preprocessing_garmin_data(username, password, start_date, end_date, headers):
    from garminconnect import Garmin
    import pandas as pd
    import numpy as np
    import datetime
    import logging
    from datetime import date, timedelta

    # Logging setup
    now = datetime.datetime.now()
    log_filename = f"garmin_data_script_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # Garmin login
    try:
        api = Garmin(username, password)
        logger.info("Attempting to log in to Garmin Connect...")
        api.login()
        logger.info("Login successful.")
    except Exception as e:
        logger.critical(f"Login failed: {e}", exc_info=True)
        raise

    # Date range
    def get_iso_date_range(start_date, end_date):
        start_date = date.fromisoformat(str(start_date))
        end_date = date.fromisoformat(str(end_date))
        if start_date > end_date:
            raise ValueError("Start date cannot be after end date.")
        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date.isoformat())
            current_date += timedelta(days=1)
        return date_list

    date_range = get_iso_date_range(start_date, end_date)

    # Initialize storage
    total_sleep_df = pd.DataFrame()
    total_bb_df = pd.DataFrame()
    total_rhr_df = pd.DataFrame()
    total_steps_df = pd.DataFrame()
    total_user_df = pd.DataFrame()
    total_activity_df = pd.DataFrame()

    # Sleep Data
    for dt in date_range:
        try:
            sleep_data = api.get_sleep_data(dt)
            sleep_df = pd.json_normalize(sleep_data)
            sleep_df["Date"] = dt
            total_sleep_df = pd.concat([total_sleep_df, sleep_df], ignore_index=True)
        except Exception as e:
            logger.warning(f"No sleep data for {dt}: {e}")

    # Body Battery, RHR, Steps, User Summary
    for dt in date_range:
        try:
            bb_df = pd.DataFrame(api.get_body_battery(dt))
            if not bb_df.empty:
                bb_df.rename(columns={"date": "Date"}, inplace=True)
                total_bb_df = pd.concat([total_bb_df, bb_df], ignore_index=True)
        except:
            pass
        try:
            rhr_df = pd.DataFrame(api.get_rhr_day(dt))
            if not rhr_df.empty:
                rhr_df.rename(columns={"statisticsStartDate": "Date"}, inplace=True)
                total_rhr_df = pd.concat([total_rhr_df, rhr_df], ignore_index=True)
        except:
            pass
        try:
            steps_df = pd.DataFrame(api.get_steps_data(dt))
            if not steps_df.empty:
                steps_df["Date"] = pd.to_datetime(steps_df["startGMT"]).dt.strftime("%Y-%m-%d")
                steps_df = steps_df.groupby("Date")[["steps"]].sum().reset_index()
                total_steps_df = pd.concat([total_steps_df, steps_df], ignore_index=True)
        except:
            pass
        try:
            user_df = pd.DataFrame(api.get_user_summary(dt))
            if not user_df.empty:
                user_df["Date"] = dt
                total_user_df = pd.concat([total_user_df, user_df], ignore_index=True)
        except:
            pass

    # Activities
    try:
        activities = api.get_activities_by_date(start_date.isoformat(), end_date.isoformat())
        activity_df = pd.DataFrame(activities)
        if not activity_df.empty:
            activity_df["Date"] = pd.to_datetime(activity_df["startTimeLocal"]).dt.strftime("%Y-%m-%d")
            total_activity_df = pd.concat([total_activity_df, activity_df], ignore_index=True)
    except Exception as e:
        logger.error(f"Error fetching activities: {e}")

    # Merge datasets
    df_final = pd.DataFrame({"Date": date_range})
    for df in [total_activity_df, total_sleep_df, total_bb_df, total_rhr_df, total_steps_df, total_user_df]:
        if not df.empty:
            df_final = df_final.merge(df, on="Date", how="left")

    # Add username column
    df_final["username"] = username

    # Align columns to match Google Sheets headers
    for col in headers:
        if col not in df_final.columns:
            df_final[col] = np.nan
    df_final = df_final[headers]

    return df_final
