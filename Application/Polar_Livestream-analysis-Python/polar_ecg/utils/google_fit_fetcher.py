import os
import json
from datetime import datetime, timedelta

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/fitness.activity.read',
    'https://www.googleapis.com/auth/fitness.body.read',
    'https://www.googleapis.com/auth/fitness.heart_rate.read',
    'https://www.googleapis.com/auth/fitness.body_temperature.read',
    'https://www.googleapis.com/auth/fitness.sleep.read'
]

class GoogleFitFetcher:
    """Manages OAuth Desktop Flow and fetches historical datasets from Google Fit API."""
    
    def __init__(self, token_path="token.json", client_secret_path="client_secret.json"):
        self.token_path = token_path
        self.client_secret_path = client_secret_path
        self.creds = None

    def authenticate(self):
        """Authenticates the user and builds the creds obj."""
        # The file token.json stores the user's access and refresh tokens
        if os.path.exists(self.token_path):
            self.creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secret_path, SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_path, 'w') as token:
                token.write(self.creds.to_json())
                
        return True

    def fetch_historical_summary(self, timeframe="7_days"):
        """
        Retrieves aggregate step, heart, calories, and sleep data within timeframe.
        timeframe: "7_days" or "1_month"
        """
        if not self.creds:
            raise Exception("Authentication required before fetching.")
            
        service = build('fitness', 'v1', credentials=self.creds)

        # Calculate Timestamps in Milliseconds
        now = datetime.now()
        if timeframe == "1_month":
            start = now - timedelta(days=30)
        else: # default 7_days
            start = now - timedelta(days=7)
            
        end_time_ms = int(now.timestamp() * 1000)
        start_time_ms = int(start.timestamp() * 1000)

        # Build Aggregation Request
        req_body = {
            "aggregateBy": [
                {
                    "dataTypeName": "com.google.step_count.delta"
                },
                {
                    "dataTypeName": "com.google.calories.expended"
                },
                {
                    "dataTypeName": "com.google.heart_minutes.summary"
                },
                {
                    "dataTypeName": "com.google.heart_rate.bpm"
                },
                {
                    "dataTypeName": "com.google.body.temperature"
                },
                {
                    "dataTypeName": "com.google.sleep.segment",
                }
            ],
            # Group into 1 day buckets
            "bucketByTime": { "durationMillis": 86400000 },
            "startTimeMillis": start_time_ms,
            "endTimeMillis": end_time_ms
        }

        response = service.users().dataset().aggregate(userId="me", body=req_body).execute()
        
        # Parse output into clean dict keyed by date string
        summary = {
            "timeframe": timeframe,
            "days": []
        }
        
        for bucket in response.get('bucket', []):
            start_ms = int(bucket.get('startTimeMillis', 0))
            day_str = datetime.fromtimestamp(start_ms / 1000.0).strftime('%Y-%m-%d')
            
            day_data = {
                "date": day_str,
                "steps": 0,
                "calories": 0.0,
                "heart_points": 0.0,
                "avg_bpm": None,
                "body_temp": None,
                "sleep_hours": 0.0
            }
            
            # Loop through datasets (matches order of aggregateBy array)
            for ds in bucket.get('dataset', []):
                points = ds.get('point', [])
                if not points: continue
                val = points[0].get('value', [{}])[0]
                
                dt_name = ds.get('dataSourceId', '')
                dt_type = ds.get('dataTypeName', '')
                
                if 'step_count' in dt_name or 'step_count' in dt_type:
                    day_data["steps"] = val.get("intVal", 0)
                elif 'calories' in dt_name or 'calories' in dt_type:
                    day_data["calories"] = round(val.get("fpVal", 0.0), 1)
                elif 'heart_minutes' in dt_name or 'heart_minutes' in dt_type:
                    day_data["heart_points"] = val.get("fpVal", 0.0)
                elif 'heart_rate' in dt_name or 'heart_rate' in dt_type:
                    day_data["avg_bpm"] = round(val.get("fpVal", 0.0), 1)
                elif 'temperature' in dt_name or 'temperature' in dt_type:
                    day_data["body_temp"] = round(val.get("fpVal", 0.0), 1)
                elif 'sleep' in dt_name or 'sleep' in dt_type or len(dt_name) == 0:
                    sleep_ms = sum([p['endTimeNanos'] - p['startTimeNanos'] for p in points]) / 1000000.0
                    day_data["sleep_hours"] = round((sleep_ms / 1000.0) / 3600.0, 2)
                    
            summary["days"].append(day_data)
            
        return summary
