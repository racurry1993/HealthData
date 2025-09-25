import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery
import os
import json

# Assume this file exists in the same directory.
from data_pre_processing import preprocessing_garmin_data

# --- BIGQUERY CLIENT (REAL IMPLEMENTATION) ---
# NOTE: This code has been temporarily modified to embed the credentials directly
# to bypass the persistent file-loading issue with Streamlit secrets.
class BigQueryClient:
    """
    A class to handle real BigQuery operations.
    """
    def __init__(self):
        # Service account credentials embedded directly as a dictionary.
        # This bypasses the need for the secrets.toml file entirely for BigQuery.
        service_account_info = {
            "type": "service_account",
            "project_id": "garminuserdata",
            "private_key_id": "48134d3b1bf58a88741f204515597a840c613fd2",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDNGwy5lYUcEXYI\nEk4i027RS6SZVDNOorkReZgm+21RPfJmv/rejXsId6c6K0yYEMejNZlbc6kKAr/b\nthe74Mu2XrA834djZ1BtdfJwwyAg1WOq/eF5QUlJK5BgdcLPrZvTJRMqFSOjcJos\nkN3ZhXvgj5nqU3cfDNesh616uSD+r46HlUM3RIWDwR1DM24jpbrfyzurtwNiwTlu\nS0z/naj4/BQSbCuBInHn/rbQeVIkhV4X61L0zZkyR6cn04/FAsB02QH65stEhUe1\nrooHDlgWf/hjsS2NV+P+d3XSvU7xbayeeINukPTnhdQd77MnZ3NjbifYmXq/kiDz\nrpUA0iNDAgMBAAECggEAB5s8YmRz9neECPr1iTmlM6JjJ08XRL6or6UhOMMOGFhl\nI8XRmuzwGq3Uk7pQoDq3E+iy3hZfWdeZiJ9ltEuMizdS198Vk3gRHoJ16Vnr4+vU\niN431ILn81JFF3AmlI/V8gN5VmGOfuJgyynEjwKsMJg87pLuwHR97O0/6017k62b\nRDEvgbXqQF7VWDeqG5CdEvzoVkUHoZtPjADn0jyyBrzk7vxs8NdB4Fmp4XAufsv9\nY6d7LitPLO1uB5r5EAgbvQ47wZ/uG6F0IYd6dYK0IMgCNui8RqBgelo0S0rpjFqS\nQlJ9AWAR1fAlKX/Dq0Sj4dOefO4lvtUV+oAdewIoOQKBgQD5f1kf3fcgVomAnvZX\nHC0gA8gFyc2grVx9rbO+ybyWNabpH5Kb8dRxSYEmNbJLoTVFGGmSjdC6h4lWZJaa\nMQ0QUbab28fKoVHu9L+4ODbQniv4pQyBgFzVi1MZahVzvYv3aLGLZLtTPXLLCFpm\n15LnV5dSTxGqGxYD+5aUJfwlZwKBgQDSc4TH0X863H8pOSBldWPHKzQbW+Ix/RTe\nEP9VbCpf23ihr/mszc1RnvhHYUBG5o8wnrP2rK8SNylJmLQh1dSZrNKqKRzUCv80\nZb0w8P1yoXC50G4ezaCw/jiGnOzw2n3gZaSRbEgQlfBtxPwVInU3kfoYnOlO6B1s\nFckkTU/txQKBgAXxU5Ufu1go14OZxbJTeHuvu17v7JbsKizQK1za/0PwqTYaS2qt\neurr3kijtMh6YYNwzmrwN82JlurY4IFxs6b0202hEYQxDXuMlthzdlLHwbJddAvN\nm+h2Nhd/4FzuYdwVwUzZrGCSMR7G5yhV8CjUfEU4nuoXVRHpKv4CaQr3AoGBAMz+\nAx+UPEc8koy3/Yt4cPOHbNkddkZVC+eHTP+LPfdjU6zDOgON7+oKXDNDUpX9bQrh\n+9BSwrGOk6QBn6y5mb4bLpTbOR5+m7oRQ+kRRP9Mq/4DPdC5YUYmSy8sWkv9t9FF\nkLiqbcPiGXEDCL0ZdG4tvhwNc+ENjeNDkUrQGAQ1AoGADJVtUtIKFCjEKHpAISNy\n2jwdcAGeFfe/wFGfOhntZQzTErOFOf+dEJpLW/Zyg/8GbrFPgQCrbk0BNsHM8fTQ\nIbWIRxHYgfOIObDgO89KJKkt/09ztvph5/DR3AtQhQwD6IQgU3gyMreYzj+g5Ebf\nE7Xv3IcmI9bLBFiODki7KVQ=\n-----END PRIVATE KEY-----\n",
            "client_email": "garminuserdata@garminuserdata.iam.gserviceaccount.com",
            "client_id": "106768300223415646530",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/garminuserdata%40garminuserdata.iam.gserviceaccount.com",
            "universe_domain": "googleapis.com"
        }
        
        try:
            self.client = bigquery.Client.from_service_account_info(service_account_info)
            self.table_id = "garminuserdata.garminuserdata.garmin_activity_data"
            st.success("BigQuery client initialized successfully.")
        except Exception as e:
            st.error(f"Failed to initialize BigQuery client: {e}")
            self.client = None
            self.table_id = None

    def get_user_most_recent_date(self, username: str) -> datetime | None:
        """Queries the table for the most recent date for a specific user."""
        if not self.client: return None
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
        if not self.client: return
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
        if not self.client: return pd.DataFrame()
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
                    processed_df = preprocessing_garmin_data(raw_df.copy())
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
