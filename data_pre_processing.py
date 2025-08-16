def preprocessing_garmin_data(username, password, start_date, end_date):
    from garminconnect import Garmin
    import datetime
    from datetime import date, timedelta
    import pandas as pd
    import logging
    import seaborn as sns
    import numpy as np

    # --- Logging Setup ---
    now = datetime.datetime.now()
    log_filename = f"garmin_data_script_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    # --- Login ---
    try:
        api = Garmin(username, password)
        logger.info("Attempting to log in to Garmin Connect...")
        api.login()
        logger.info("Login successful.")
    except Exception as e:
        logger.critical(f"A critical error occurred during login: {e}", exc_info=True)
        raise

    # --- Date range ---
    def get_iso_date_range(start_date_str: str, end_date_str: str) -> list[str]:
        date_list = []
        start_date_obj = date.fromisoformat(start_date_str)
        end_date_obj = date.fromisoformat(end_date_str)

        if start_date_obj > end_date_obj:
            raise ValueError("Start date cannot be after end date.")

        current_date = start_date_obj
        while current_date <= end_date_obj:
            date_list.append(current_date.isoformat())
            current_date += timedelta(days=1)
        return date_list

    def get_rhr_value(metric_list):
        if isinstance(metric_list, list) and len(metric_list) > 0:
            if isinstance(metric_list[0], dict):
                return metric_list[0].get('value')
        return None

    date_range = get_iso_date_range(start_date, end_date)
    logger.info(f"Fetching data from {start_date} to {end_date} ({len(date_range)} days).")

    total_sleep_df = pd.DataFrame()
    new_total_sleep_df = pd.DataFrame()
    missed_dts = []
    for dt in date_range:
        logger.info(f"Fetching data for period: {dt}")
        try:
            ##Sleep DF
            sleep_df = pd.DataFrame(api.get_sleep_data(dt))
            staging_sleep_df = sleep_df.iloc[:, [0]].T
            staging_sleep_df.reset_index(inplace=True)
            staging_sleep_df.drop(['index'], axis=1, inplace=True)
            staging_sleep_df['Date'] = dt
            additional_sleep_df = sleep_df.reset_index().iloc[:,2:].drop_duplicates()
            additional_sleep_df['Date'] = dt
            additional_sleep_df
            sleep_df = staging_sleep_df.merge(additional_sleep_df,  left_on='Date', right_on='Date', how='inner')
            first_record_sleep_scores = sleep_df['sleepScores'].iloc[0]
            first_record_other_data = sleep_df.iloc[0].drop('sleepScores').to_dict()
            # Flatten the sleepScores dictionary
            normalized_scores = pd.json_normalize(first_record_sleep_scores, sep='_')
            # Create a DataFrame for other simple columns
            other_data_df = pd.DataFrame([first_record_other_data])
            # Combine them. Use the same index to ensure they line up for concat.
            sleep_df = pd.concat([other_data_df, normalized_scores], axis=1)
            sleep_df.drop_duplicates(inplace=True)
            total_sleep_df = pd.concat([total_sleep_df, sleep_df], axis=0)
        except Exception as e:
            missed_dts.append(dt)
    for missed_date in missed_dts:
        sleep_df = pd.DataFrame([api.get_sleep_data(missed_date)])
        record_data = sleep_df.iloc[0, 0]
        normalized_df = pd.json_normalize(record_data, sep='_')
        normalized_df['Date'] = missed_date
        new_total_sleep_df = pd.concat([new_total_sleep_df, normalized_df], axis=0)

    t1 = set(new_total_sleep_df.columns)
    t2 = set(total_sleep_df.columns)
    keep_cols = list(t1.intersection(t2))

    total_sleep_df = total_sleep_df[keep_cols]
    new_total_sleep_df = new_total_sleep_df[keep_cols]
    total_sleep_df = pd.concat([total_sleep_df, new_total_sleep_df], axis=0)


    def missing_percentage_cols(df):
        for x in df.columns[df.isnull().sum() > 0]:
            missing_percentage = (df[x].isnull().sum()/len(df) * 100)
            print(f'Percentage of Missing Values Column - {x}: {missing_percentage:.2f}%')
        return

    missing_percentage_cols(total_sleep_df)
    cols_to_delete = ['sleepQualityTypePK','autoSleepEndTimestampGMT','sleepResultTypePK',
                        'autoSleepStartTimestampGMT','sleepScorePersonalizedInsight',
                        'highestRespirationValue','lowestRespirationValue','sleepStartTimestampLocal',
                        'sleepStartTimestampGMT','sleepVersion','retro','deviceRemCapable','sleepEndTimestampLocal',
                        'sleepEndTimestampGMT','sleepFromDevice','napTimeSeconds',
                        'userProfilePK','sleepWindowConfirmationType','calendarDate','id','unmeasurableSleepSeconds'
                        ]
    for x in cols_to_delete:
        try:
            total_sleep_df.drop(x, axis=1, inplace=True)
        except:
            print(f'Column {x} does not exist')
    print('=====Dropped 100% Null Columns======')
    missing_percentage_cols(total_sleep_df)

    total_bb_df = pd.DataFrame()
    total_rhr_df = pd.DataFrame()
    total_steps_df = pd.DataFrame()
    total_user_df = pd.DataFrame()
    for dt in date_range:
        logger.info(f"Fetching data for period: {dt}")
        try:

            ##BodyBattery DF
            bb_df = pd.DataFrame(api.get_body_battery(dt))
            bb_df = bb_df[['date','charged','drained']]
            bb_df.rename({'date':'Date'}, axis=1, inplace=True)
            total_bb_df = pd.concat([total_bb_df, bb_df[['Date','charged','drained']]], axis=0)

            ##RestingHR DF
            rhr_df = pd.DataFrame(api.get_rhr_day(dt))
            #rhr_df['allMetricsClean'] = rhr_df['allMetrics'].apply(
            #    lambda x: x.get('WELLNESS_RESTING_HEART_RATE') if isinstance(x, dict) else None
            #)
            # Apply the function to the 'allMetricsClean' column
            #rhr_df['resting_heart_rate_value'] = rhr_df['allMetricsClean'].apply(get_rhr_value)
            rhr_df.rename({'statisticsStartDate':'Date'}, axis=1, inplace=True)
            total_rhr_df = pd.concat([total_rhr_df, rhr_df], axis=0)


            ##Steps DF
            steps_df = pd.DataFrame(api.get_steps_data(dt))
            steps_df['Date'] = pd.to_datetime(steps_df['startGMT']).dt.strftime('%Y-%m-%d')
            steps_df = steps_df.groupby(['Date'])[['steps','pushes']].sum().head()
            steps_df.reset_index(inplace=True)
            total_steps_df = pd.concat([total_steps_df, steps_df], axis=0)

            ##User Summary Data
            user = pd.DataFrame(api.get_user_summary(dt))
            user.reset_index(inplace=True)
            user = user.loc[user['index'] == 'typeId'].drop('index', axis=1)
            total_user_df = pd.concat([total_user_df, user], axis=0)
        except Exception as e:
            logger.error(f"Unexpected error during data processing: {e}", exc_info=True)

    total_user_df.rename({'calendarDate':'Date'}, axis=1, inplace=True)

    mn = min(date_range)
    mx = max(date_range)
    total_activity_df = pd.DataFrame()
    logger.info("Getting activities data...")
    try:
        activities = api.get_activities_by_date(mn, mx)
        activity_df = pd.DataFrame(activities)
        total_activity_df = pd.concat([total_activity_df, activity_df], axis=0)

    except Exception as e:
        logger.error(f"Error during activities data pull: {e}", exc_info=True)

    total_activity_df['typeKey_clean'] = total_activity_df['activityType'].apply(
        lambda x: x.get('typeKey') if isinstance(x, dict) else None
    )
    total_activity_df.drop(['activityType','eventType'], axis=1, inplace=True)
    total_activity_df['Date'] = pd.to_datetime(total_activity_df['startTimeLocal']).dt.strftime('%Y-%m-%d')
    cols_mapping = total_activity_df.columns.drop('Date').to_list()
    cols_mapping = ['Date'] + cols_mapping
    total_activity_df = total_activity_df[cols_mapping]

    dataframes = [total_activity_df, total_sleep_df, total_bb_df, total_rhr_df, total_steps_df, total_user_df]
    for missing_data in dataframes:
        print(f"Missing Data for DF")
        missing_percentage_cols(missing_data)

    def drop_missing_cols_by_threshold(df, threshold):
        missing_percentages = df.isnull().mean() * 100
        cols_to_drop = missing_percentages[missing_percentages >= threshold].index
        
        if not cols_to_drop.empty:
            print(f"Dropped columns with missing values >= {threshold}%: {cols_to_drop.tolist()}")
        else:
            print(f"No columns with missing values >= {threshold}% were found.")
            
        df_cleaned = df.drop(columns=cols_to_drop)
        return df_cleaned

    threshold_value = 90
    for df in dataframes:
        cleaned_df = drop_missing_cols_by_threshold(df, threshold_value)

    ## Total DFs Listed Below
    #1.) total_activity_df ##Activities data
    #2.) total_sleep_df ##Sleep data
    #3.) total_bb_df ##Body Battery data
    #4.) total_rhr_df #Resting HR data
    #5.) total_steps_df #Steps data
    #6.) total_user_df #User summary data

    df_list = [total_activity_df, total_sleep_df, total_bb_df, total_rhr_df, total_steps_df, total_user_df]

    try:
        logger.info("Starting combination of dataframes...")
        total_df = pd.DataFrame(date_range)
        total_df.rename({0:'Date'}, axis=1, inplace=True)
        for stage_df in df_list:
            total_df = total_df.merge(stage_df, left_on='Date', right_on='Date', how='left')
    except Exception as e:
        logger.error(f"Error during combination process of all datasets {e}", exc_info=True)


    total_df['ActivityStartHour'] = pd.to_datetime(total_df['startTimeLocal']).dt.strftime('%H')
    total_df.rename({'steps_x':'ActivitySteps'}, inplace=True, axis=1)
    total_df = total_df.drop_duplicates(subset='Date')
    columns_to_check = total_df.columns.drop('Date')
    rows_to_keep = ~((total_df[columns_to_check].isnull() | (total_df[columns_to_check] == 0)).all(axis=1))
    total_df = total_df.loc[rows_to_keep]

    cols_to_drop = ['latestRespirationValue','lowestRespirationValue_y','highestRespirationValue_y','latestRespirationTimeGMT',
                    'respirationAlgorithmVersion','latestSpo2ReadingTimeLocal','latestSpo2ReadingTimeGmt','startTimeGMT',
                    'activityId','ownerId','ownerProfileImageUrlSmall','hasImages','ownerProfileImageUrlMedium','ownerProfileImageUrlLarge',
                    'privacy','hasVideo','timeZoneId','deviceId','endTimeGMT','favorite','decoDive','workoutId','calendarDate','id',
                    'deviceRemCapable','retro','sleepStartTimestampGMT','userProfilePK','skinTempDataExists','userProfileId_x','statisticsEndDate',
                    'userProfileId_y','userDailySummaryId','rule','uuid','wellnessStartTimeGmt','wellnessEndTimeGmt','privacyProtected',
                    'source','lastSyncTimestampGMT','bodyBatteryVersion','startTimeLocal','hasPolyline','userRoles','userPro',
                    'beginTimestamp','sportTypeId','summarizedDiveInfo','manufacturer','hasSplits','qualifyingDive','hasHeatMap','pr',
                    'purposeful','parent','manualActivity','autoCalcCalories','elevationCorrected','atpActivity','summarizedExerciseSets',
                    'sleepEndTimestampGMT','sleepEndTimestampLocal','sleepQualityTypePK','sleepResultTypePK','sleepStartTimestampLocal',
                    'sleepVersion','sleepWindowConfirmationType','sleepLevels','sleepScorePersonalizedInsight','groupedMetrics',
                    'wellnessStartTimeLocal','wellnessEndTimeLocal','durationInMilliseconds','wellnessDescription','averageSpo2',
                    'lowestSpo2','latestSpo2','splitSummaries','allMetrics','allMetricsClean','sleepFromDevice','startLatitude','startLongitude',
                    'endLatitude','endLongitude','ownerDisplayName','autoSleepStartTimestampGMT','autoSleepEndTimestampGMT','totalDuration_optimalStart',
                    'totalDuration_optimalEnd','optimalStart','stress_optimalEnd','remPercentage_optimalStart','remPercentage_optimalEnd',
                    'remPercentage_idealStartInSeconds','remPercentage_idealEndInSeconds','restlessness_optimalStart','restlessness_optimalEnd',
                    'lightPercentage_optimalStart','lightPercentage_optimalEnd','lightPercentage_idealStartInSeconds','lightPercentage_idealEndInSeconds',
                    'deepPercentage_optimalStart','deepPercentage_optimalEnd','deepPercentage_idealStartInSeconds','deepPercentage_idealEndInSeconds']
    for col in cols_to_drop:
        try:
            total_df.drop(col, axis=1, inplace=True)
        except:
            print('Column does not exist')

    logger.info("Pre-Processing finished.")
    total_df['ActivityPerformedToday'] = total_df['activityName'].isnull().apply(lambda x: 0 if x == True else 1)
    username = total_df['ownerFullName'].unique()
    cols_na = list(total_df.columns[total_df.isnull().sum() > 0])
    for col in cols_na:
        print(f"{col}: % missing - {len(total_df.loc[(total_df[col].isnull() == True) & (total_df['ActivityPerformedToday'] == 0)]) / len(total_df)}")


    logger.info("Handling Missing Values")

    cols_to_drop_na = ['sleepWindowConfirmed','abnormalHeartRateAlertsCount','burnedKilocalories',
                    'consumedKilocalories','bodyBatteryAtWakeTime','maxVerticalSpeed','maxElevation','minElevation',
                    'maxTemperature','minTemperature','averageRunningCadenceInStepsPerMinute',
                    'avgStrideLength']
    print('-------Dropping Columns--------')
    for col in cols_to_drop_na:
        try:
            total_df.drop(col, axis=1, inplace=True)
        except Exception as e:
            logger.error(f"Error during filling 0 (NA) of column {e}", exc_info=True)

    cols_to_fill_0 = ['totalSets','activeSets','totalReps','hrTimeInZone_1','hrTimeInZone_2','hrTimeInZone_3','hrTimeInZone_4','hrTimeInZone_5',
                    'vO2MaxValue',
                    'maxDoubleCadence','moderateIntensityMinutes_x','vigorousIntensityMinutes_x','elevationGain','elevationLoss',
                    'maxSpeed','minActivityLapDuration','ActivitySteps','steps_y','pushes',
                    'totalDistanceMeters','wellnessDistanceMeters','wellnessActiveKilocalories','netRemainingKilocalories',
                    'highlyActiveSeconds','activeSeconds','sedentarySeconds','intensityMinutesGoal','userFloorsAscendedGoal',
                    'moderateIntensityMinutes_y','vigorousIntensityMinutes_y','stressQualifier','measurableAwakeDuration','measurableAsleepDuration',
                    ]

    print('-------Filling Columns w/0--------')
    for col in cols_to_fill_0:
        try:
            total_df[col].fillna(0, inplace=True)
        except Exception as e:
            logger.error(f"Error during filling 0 (NA) of column {e}", exc_info=True)

    cols_to_fill_mean = ['minAvgHeartRate','maxAvgHeartRate','avgWakingRespirationValue','restingCaloriesFromActivity','averageMonitoringEnvironmentAltitude',
                        'ageGroup','movingDuration', 'bmrCalories','averageHR','minTemperature','maxTemperature','minElevation',
                        'maxElevation','maxVerticalSpeed','avgSleepStress','awakeCount','awakeSleepSeconds','deepSleepSeconds',
                        'highestRespirationValue_x','lightSleepSeconds','lowestRespirationValue_x','napTimeSeconds','remSleepSeconds',
                        'sleepTimeSeconds','remSleepData','bodyBatteryChange','restingHeartRate_x','overall_value','remPercentage_value',
                        'lightPercentage_value','deepPercentage_value','charged','drained','totalKilocalories','activeKilocalories',
                        'bmrKilocalories','wellnessKilocalories','remainingKilocalories','totalSteps','netCalorieGoal','dailyStepGoal',
                        'sleepingSeconds','floorsAscendedInMeters','floorsDescendedInMeters','floorsAscended','floorsDescended',
                        'minHeartRate','maxHeartRate','restingHeartRate_y','lastSevenDaysAvgRestingHeartRate','averageStressLevel',
                        'maxStressLevel','stressDuration','restStressDuration','activityStressDuration','uncategorizedStressDuration',
                        'totalStressDuration','lowStressDuration','mediumStressDuration','highStressDuration','stressPercentage',
                        'restStressPercentage','activityStressPercentage','uncategorizedStressPercentage','lowStressPercentage',
                        'mediumStressPercentage','highStressPercentage','bodyBatteryChargedValue','bodyBatteryDrainedValue',
                        'bodyBatteryHighestValue','bodyBatteryLowestValue','bodyBatteryMostRecentValue','bodyBatteryDuringSleep',
                        ]

    print('-------Filling Columns w/mean--------')
    cols_error = []
    for col in cols_to_fill_0:
        try:
            mean_value = total_df[col].mean()
            total_df[col].fillna(mean_value, inplace=True)
        except Exception as e:
            print(f"{col} has an exception of {e} and wasn't converted")
            cols_error.append(col)

    total_df['stressQualifier'].fillna('UNKNOWN', inplace=True)
    # --- Part 1: Data Loading and Initial Preprocessing ---
    print("1. Loading Data and Initial Preprocessing...")
    df_cleaned = total_df.copy()

    # Convert 'Date' column to datetime objects
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])

    # Sort by Date to prepare for time-series operations
    df_cleaned = df_cleaned.sort_values(by='Date').reset_index(drop=True)

    # Identify and drop columns that are almost entirely null or not useful
    # These were identified in previous steps
    columns_to_drop_initial = [
        'sleepMovement', 'burnedKilocalories', 'consumedKilocalories',
        'abnormalHeartRateAlertsCount', 'ownerFullName'
    ]

    for col in columns_to_drop_initial:
        if col in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=[col])
            print(f"Dropped column: {col}")

    # Impute 0 for metrics where NaN indicates absence of measurement
    # These are typically activity-specific metrics that might not apply to all activity types
    columns_to_fill_0 = [
        'movingDuration', 'elevationGain', 'elevationLoss', 'minElevation', 'maxElevation',
        'maxSpeed', 'maxVerticalSpeed', 'ActivitySteps', 'minTemperature', 'maxTemperature',
        'averageRunningCadenceInStepsPerMinute', 'maxRunningCadenceInStepsPerMinute',
        'avgStrideLength', 'vO2MaxValue', 'maxDoubleCadence', 'totalSets', 'activeSets',
        'totalReps', 'hrTimeInZone_1', 'hrTimeInZone_2', 'hrTimeInZone_3', 'hrTimeInZone_4',
        'hrTimeInZone_5', 'distance', 'duration', 'elapsedDuration', 'calories', 'averageSpeed',
        'maxHR', 'averageHR', 'aerobicTrainingEffect', 'anaerobicTrainingEffect', 'lapCount',
        'waterEstimated', 'activityTrainingLoad', 'minActivityLapDuration',
        'moderateIntensityMinutes_x', 'vigorousIntensityMinutes_x', 'steps_y', 'pushes',
        'totalKilocalories', 'activeKilocalories', 'bmrKilocalories', 'wellnessKilocalories',
        'remainingKilocalories', 'totalSteps', 'netCalorieGoal', 'totalDistanceMeters',
        'wellnessDistanceMeters', 'wellnessActiveKilocalories', 'netRemainingKilocalories',
        'dailyStepGoal', 'highlyActiveSeconds', 'activeSeconds', 'sedentarySeconds',
        'sleepingSeconds', 'moderateIntensityMinutes_y', 'vigorousIntensityMinutes_y',
        'floorsAscendedInMeters', 'floorsDescendedInMeters', 'floorsAscended', 'floorsDescended',
        'intensityMinutesGoal', 'userFloorsAscendedGoal', 'minHeartRate', 'maxHeartRate',
        'restingHeartRate_y', 'lastSevenDaysAvgRestingHeartRate', 'averageStressLevel',
        'maxStressLevel', 'stressDuration', 'restStressDuration', 'activityStressDuration',
        'uncategorizedStressDuration', 'totalStressDuration', 'lowStressDuration',
        'mediumStressDuration', 'highStressDuration', 'stressPercentage', 'restStressPercentage',
        'activityStressPercentage', 'uncategorizedStressPercentage', 'lowStressPercentage',
        'mediumStressPercentage', 'highStressPercentage', 'measurableAwakeDuration',
        'measurableAsleepDuration', 'minAvgHeartRate', 'maxAvgHeartRate', 'bodyBatteryChargedValue',
        'bodyBatteryDrainedValue', 'bodyBatteryHighestValue', 'bodyBatteryLowestValue',
        'bodyBatteryMostRecentValue', 'averageMonitoringEnvironmentAltitude',
        'restingCaloriesFromActivity', 'avgWakingRespirationValue'
    ]

    for col in columns_to_fill_0:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].fillna(0)
            # print(f"Filled NaN in {col} with 0.")

    # Impute remaining numerical columns with their median
    numerical_cols = df_cleaned.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        if df_cleaned[col].isnull().any():
            median_val = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_val)
            # print(f"Filled NaN in numerical column {col} with median ({median_val}).")

    cols_to_fill_cat_0 = ['activityName', 'locationName', 'trainingEffectLabel','aerobicTrainingEffectMessage',
                        'anaerobicTrainingEffectMessage']

    for col in cols_to_fill_cat_0:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].fillna('None')

    # Impute remaining categorical columns with their mode
    categorical_cols = df_cleaned.select_dtypes(include='object').columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().any():
            # Check if the column is entirely null before trying to get mode
            if not df_cleaned[col].isnull().all():
                mode_val = df_cleaned[col].mode()[0]
                df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                # print(f"Filled NaN in categorical column {col} with mode ({mode_val}).")
            else:
                # If still entirely null, fill with 'Unknown'
                df_cleaned[col] = df_cleaned[col].fillna('Unknown')
                # print(f"Filled entirely NaN categorical column {col} with 'Unknown'.")

    # Verify no more nulls
    final_nulls = df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0]
    if final_nulls.empty:
        print("All null values have been successfully handled.")
    else:
        print("\nRemaining null values after all imputations:")
        print(final_nulls)

    def activitystart_cleanup(row):
        if row['ActivityPerformedToday'] == 1:
            return row['ActivityStartHour']
        else:
            return 0
    df_cleaned['ActivityStartHour'] = df_cleaned.apply(activitystart_cleanup, axis=1)


    # --- 1. Create 'activityType' Column ---
    def categorize_activity(activity_name):
        activity_name_lower = str(activity_name).lower()
        if 'walking' in activity_name_lower:
            return 'Walking'
        elif 'running' in activity_name_lower or '5k' in activity_name_lower:
            return 'Running'
        elif 'golf' in activity_name_lower:
            return 'Golf'
        elif 'strength' in activity_name_lower or 'worko' in activity_name_lower:
            return 'Strength Training'
        elif 'cycling' in activity_name_lower:
            return 'Cycling'
        elif 'hiking' in activity_name_lower:
            return 'Hiking'
        elif 'multisport' in activity_name_lower:
            return 'Multisport' # Could be broken down further if 'multisport' has sub-types
        elif activity_name_lower == 'none':
            return 'No Activity'
        else:
            # Catch-all for any uncategorized activities. Review these after initial run.
            print(f"Warning: Uncategorized activity: {activity_name}")
            return 'Other'

    df_cleaned['activityType'] = df_cleaned['activityName'].apply(categorize_activity)
    df_cleaned.drop('activityName', axis=1, inplace=True)

    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    df_cleaned = df_cleaned.sort_values(by='Date').reset_index(drop=True)
    df_cleaned['ActivityPerformedToday'] = df_cleaned['ActivityPerformedToday'].astype(bool)

    # Create a column for workout dates, fill non-workout days with NaT (Not a Time)
    df_cleaned['workout_date'] = df_cleaned['Date'].where(df_cleaned['ActivityPerformedToday'])

    # Fill forward the last workout date. `.ffill()` will propagate the last valid workout_date downwards.
    df_cleaned['last_workout_date'] = df_cleaned['workout_date'].ffill()

    # Calculate days since last workout. For days before any workout, this will be NaT.
    df_cleaned['daysSinceLastWorkout'] = (df_cleaned['Date'] - df_cleaned['last_workout_date']).dt.days

    # --- Convert sleep seconds to hours for better readability ---
    sleep_cols_seconds = ['sleepTimeSeconds', 'deepSleepSeconds', 'remSleepSeconds', 'lightSleepSeconds', 'awakeSleepSeconds']
    for col in sleep_cols_seconds:
        df_cleaned[col.replace('Seconds', 'Hours')] = df_cleaned[col] / 3600

    df_cleaned['deepSleepPercentage'] = (df_cleaned['deepSleepSeconds'] / df_cleaned['sleepTimeSeconds']) * 100
    df_cleaned['remSleepPercentage'] = (df_cleaned['remSleepSeconds'] / df_cleaned['sleepTimeSeconds']) * 100
    df_cleaned['lightSleepPercentage'] = (df_cleaned['lightSleepSeconds'] / df_cleaned['sleepTimeSeconds']) * 100
    df_cleaned['awakeSleepPercentage'] = (df_cleaned['awakeSleepSeconds'] / df_cleaned['sleepTimeSeconds']) * 100 # Might be high if sleepTimeSeconds includes awake time

    # Time-based features
    df_cleaned['day_of_week'] = df_cleaned['Date'].dt.day_name()
    df_cleaned['is_weekend'] = df_cleaned['Date'].dt.weekday.isin([5, 6]).astype(int)
    df_cleaned['month'] = df_cleaned['Date'].dt.month
    df_cleaned['day_of_year'] = df_cleaned['Date'].dt.dayofyear
    print("Created time-based features (day_of_week, is_weekend, month, day_of_year).")
    df_cleaned = df_cleaned.drop_duplicates(subset='Date')
    numeric_columns = ['totalSteps', 'ActivitySteps', 'calories', 'bmrCalories',
                   'averageHR', 'maxHR', 'movingDuration', 'elapsedDuration']

    for col in numeric_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0)

    return df_cleaned


#df = preprocessing_garmin_data("racurry93@gmail.com", "Bravesr1")
