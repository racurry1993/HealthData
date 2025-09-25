import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery
import os

# Assume this file exists in the same directory.
from data_pre_processing import preprocessing_garmin_data

# --- BIGQUERY CLIENT (REAL IMPLEMENTATION) ---
# Ensure you have your Google Cloud credentials configured.
# You can set the GOOGLE_APPLICATION_CREDENTIALS environment variable,
# or provide the path to your service account key file here.
# For example: os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/your-key.json'

class BigQueryClient:
    """
    A class to handle real BigQuery operations.
    """
    def __init__(self):
        # Initialize the BigQuery client using the service account credentials
        self.client = bigquery.Client.from_service_account_info(
            st.secrets["gcp_bigquery_service_account"]
        )
        self.table_id = "garminuserdata.garminuserdata.garmin_activity_data"

    def get_user_most_recent_date(self, username: str) -> datetime | None:
        """Queries the table for the most recent date for a specific user."""
        query = f"""
        SELECT MAX(Date) AS most_recent_date
        FROM `{self.table_id}`
        WHERE username = '{username}'
        """
        query_job = self.client.query(query)
        try:
            result = query_job.result()
            row = next(result)
            if row.most_recent_date:
                # BigQuery returns a datetime object, we want just the date part
                return row.most_recent_date.date()
            return None
        except StopIteration:
            return None

    def add_data(self, username: str, df: pd.DataFrame):
        """Inserts a DataFrame into the BigQuery table."""
        # Add the username to the DataFrame before logging
        df['username'] = username
        
        job_config = bigquery.LoadJobConfig(write_disposition='WRITE_APPEND')
        
        # Load the DataFrame to the BigQuery table
        job = self.client.load_table_from_dataframe(df, self.table_id, job_config=job_config)
        
        try:
            job.result()  # Wait for the load job to complete
            st.success(f"Successfully appended {job.output_rows} rows to BigQuery table {self.table_id}.")
        except Exception as e:
            st.error(f"Failed to load data to BigQuery: {e}")
            st.error(f"Job errors: {job.errors}")
            
    def get_all_user_data(self, username: str) -> pd.DataFrame:
        """Pulls all data for a specific user from BigQuery."""
        query = f"""
        SELECT *
        FROM `{self.table_id}`
        WHERE username = '{username}'
        ORDER BY Date ASC
        """
        query_job = self.client.query(query)
        
        try:
            df = query_job.result().to_dataframe()
            st.success(f"Successfully retrieved {len(df)} records for user '{username}'.")
            return df
        except Exception as e:
            st.error(f"Failed to retrieve data from BigQuery: {e}")
            return pd.DataFrame()

# --- MOCK AUTH CLIENT (RETAINED FOR DEMO) ---
# We keep this mock class as it simulates the part of your application
# that retrieves new data from an external source (e.g., Garmin API).
class MockAuthClient:
    """
    A mock class to simulate user authentication and data retrieval from a source.
    """
    def authenticate(self, username, password) -> bool:
        """Simulates a login attempt."""
        # Replace with your real authentication logic
        return bool(username and password)

    def get_user_data_since(self, username: str, start_date: datetime) -> pd.DataFrame:
        """
        Simulates getting new data for a user since a specific date.
        In a real app, this would be a call to your data source API.
        """
        st.info(f"Simulating a data pull for user '{username}' from {start_date} to today.")
        # This is dummy data.
        data = {
            'Date': pd.date_range(start=start_date, end=datetime.now(), periods=5),
            'value': range(len(pd.date_range(start=start_date, end=datetime.now(), periods=5))),
            'some_column': ['A', 'b', 'C', 'd', 'E'],
            # The following columns are placeholders to match your cleaning function
            'locationName': ['Home', 'Park', 'Gym', 'Street', 'Trail'],
            'trainingEffectLabel': ['Low', 'High', 'Low', 'High', 'Medium'],
            'aerobicTrainingEffectMessage': ['Normal', 'Normal', 'Normal', 'Normal', 'Normal'],
            'anaerobicTrainingEffectMessage': ['Normal', 'Normal', 'Normal', 'Normal', 'Normal'],
            'typeKey_clean': ['run', 'walk', 'cycle', 'run', 'walk'],
            'startTimeCat': ['morning', 'afternoon', 'evening', 'morning', 'afternoon'],
            'ageGroup': ['20-29', '30-39', '40-49', '50-59', '60+'],
            'sleepScoreInsight': ['Good', 'Good', 'Good', 'Good', 'Good'],
            'sleepScoreFeedback': ['Good', 'Good', 'Good', 'Good', 'Good'],
            'stressQualifier': ['High', 'High', 'High', 'High', 'High'],
            'ActivityStartHour': [8, 14, 20, 9, 15],
            'activityType': ['running', 'walking', 'cycling', 'running', 'walking'],
            'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        }
        return pd.DataFrame(data)

def get_health_recommendations(user_data_df: pd.DataFrame) -> str:
    """
    This function simulates an ML/AI/LLM-powered recommendation engine.
    In a real app, you would pass the user's data to a trained model or an LLM API.
    """
    st.subheader("Generating Personalized Recommendations...")
    
    # Simulate a prompt to a powerful LLM
    prompt = f"""
    Analyze the following user health data to provide personalized daily recommendations.
    
    Data summary:
    - Total data points: {len(user_data_df)}
    - Average value: {user_data_df['value'].mean():.2f}
    - Highest value: {user_data_df['value'].max()}
    - Lowest value: {user_data_df['value'].min()}
    
    Provide actionable advice on a daily basis. Use a professional, encouraging tone.
    """
    
    # Simulate the LLM response
    mock_response = f"""
Based on your recent activity and tracked values, here are some actionable recommendations to help you improve your health.
These insights are derived from a comprehensive analysis of your personal data.

### Day 1: Focus on Consistency
* **Actionable Tip:** Your average tracked value is **{user_data_df['value'].mean():.2f}**. Aim to maintain consistency by tracking a similar value for the next 24 hours.
* **Why it matters:** Consistent data logging helps the model identify patterns and provide more accurate long-term recommendations.

### Day 2: Challenge Yourself
* **Actionable Tip:** Your peak value was **{user_data_df['value'].max()}**. Today, try to reach a new personal best or maintain a high level of performance.
* **Why it matters:** Pushing your limits safely is key to making progress and achieving higher health goals.

### Daily Goal Setting
* **Set a goal:** Aim to increase your daily tracked value by **10%** over the next week.
* **Monitor your progress:** Use the provided data tables to track your progress and see the impact of these recommendations in real-time.

Remember, every effort counts. Consistency and small daily improvements lead to significant long-term results.
"""
    return mock_response

# --- APPLICATION LOGIC ---

# Initialize clients
bq_client = BigQueryClient()
auth_client = MockAuthClient()

# --- STREAMLIT UI ---

st.set_page_config(page_title="Personal Health Dashboard", layout="wide")

st.title('Health and Wellness Dashboard')
st.markdown("""
<div style="text-align: center; font-size: 18px; color: gray;">
    Your personal data, processed and transformed into actionable health recommendations.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# User login form
with st.form("login_form"):
    st.subheader("Log in to Access Your Personalized Dashboard")
    st.info("For this demo, any non-empty username/password will work.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Log In & Process Data")

if submitted:
    if auth_client.authenticate(username, password):
        st.success("Login successful! Processing your data...")
        
        # 1. Check for the most recent date for this specific user
        most_recent_date = bq_client.get_user_most_recent_date(username)
        
        if most_recent_date:
            st.info(f"User '{username}' found. Last data entry was on: **{most_recent_date}**.")
            # Pull data from the day after the last entry
            start_date_filter = most_recent_date + timedelta(days=1)
            st.write(f"Fetching data from **{start_date_filter}** to the current date.")
        else:
            st.warning(f"New user '{username}'. Pulling all available data from **2024-01-01**.")
            # For a new user, pull data from 2024 forward
            start_date_filter = datetime(2024, 1, 1).date()

        # 2. Get data from the source (mocked as a function call)
        raw_df = auth_client.get_user_data_since(username, start_date_filter)

        if not raw_df.empty:
            with st.expander("View Raw and Processed Data"):
                st.subheader("Raw Data from Source")
                st.dataframe(raw_df)

                # 3. Use the pre-processing function
                try:
                    processed_df = clean_data(raw_df.copy())
                    st.subheader("Cleaned Data (from data_pre_processing.py)")
                    st.dataframe(processed_df)

                    # 4. Log data to BigQuery
                    st.subheader("Logging New Data...")
                    bq_client.add_data(username, processed_df)

                except Exception as e:
                    st.error(f"An error occurred during data processing: {e}")
            
            # 5. Get all of the user's data for the ML/AI model
            full_user_data = bq_client.get_all_user_data(username)
            if not full_user_data.empty:
                st.markdown("---")
                st.header("Daily Health Recommendations")
                
                # 6. Integrate ML/AI/LLM for health recommendations
                recommendations = get_health_recommendations(full_user_data)
                
                # Display the recommendations in a professional format
                st.markdown(recommendations)
            
        else:
            st.info("No new data to process since the last entry.")

    else:
        st.error("Invalid username or password.")
