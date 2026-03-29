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
        from googleapiclient.errors import HttpError

        # Calculate Timestamps in Milliseconds
        now = datetime.now()
        if timeframe == "1_month":
            start = now - timedelta(days=30)
            num_days = 30
        else: # default 7_days
            start = now - timedelta(days=7)
            num_days = 7
            
        end_time_ms = int(now.timestamp() * 1000)
        start_time_ms = int(start.timestamp() * 1000)

        summary = {
            "timeframe": timeframe,
            "days": []
        }
        
        # Pre-allocate day dictionaries
        for i in range(num_days + 1):
            d = start + timedelta(days=i)
            summary["days"].append({
                "date": d.strftime('%Y-%m-%d'),
                "steps": 0,
                "calories": 0.0,
                "heart_points": 0.0,
                "avg_bpm": None,
                "body_temp": None,
                "sleep_hours": 0.0
            })

        # Separate REST requests let us gracefully catch "no default datasource"
        # errors if the test patient doesn't have a smartwatch or track temperature.
        metrics = [
            ("steps", "com.google.step_count.delta"),
            ("calories", "com.google.calories.expended"),
            ("heart_points", "com.google.heart_minutes"),
            ("avg_bpm", "com.google.heart_rate.bpm"),
            ("body_temp", "com.google.body.temperature")
        ]
        
        for key, type_name in metrics:
            req_body = {
                "aggregateBy": [{"dataTypeName": type_name}],
                "bucketByTime": { "durationMillis": 86400000 },
                "startTimeMillis": start_time_ms,
                "endTimeMillis": end_time_ms
            }
            
            try:
                res = service.users().dataset().aggregate(userId="me", body=req_body).execute()
                
                for bucket in res.get('bucket', []):
                    b_start = int(bucket.get('startTimeMillis', 0))
                    day_str = datetime.fromtimestamp(b_start / 1000.0).strftime('%Y-%m-%d')
                    
                    day_data = next((d for d in summary['days'] if d['date'] == day_str), None)
                    if not day_data: continue
                    
                    for ds in bucket.get('dataset', []):
                        points = ds.get('point', [])
                        if not points: continue
                        
                        val = points[0].get('value', [{}])[0]
                        if key == "steps":
                            day_data["steps"] += val.get("intVal", 0)
                        elif key == "calories":
                            day_data["calories"] += round(val.get("fpVal", 0.0), 1)
                        elif key == "heart_points":
                            day_data["heart_points"] += round(val.get("fpVal", 0.0), 1)
                        elif key == "avg_bpm":
                            day_data["avg_bpm"] = round(val.get("fpVal", 0.0), 1)
                        elif key == "body_temp":
                            day_data["body_temp"] = round(val.get("fpVal", 0.0), 1)
                            
            except HttpError as e:
                # Silently ignore sensors the user does not possess
                pass
                
        # --- Fetch Sleep Data via Sessions API ---
        # Google Fit aggregate drops sleep sessions that cross midnight.
        # It is highly recommended to use the Sessions endpoint (activityType=72).
        try:
            start_iso = start.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            end_iso = now.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            
            sessions_res = service.users().sessions().list(
                userId="me", 
                startTime=start_iso,
                endTime=end_iso,
                activityType=72
            ).execute()
            
            for session in sessions_res.get('session', []):
                s_start = int(session.get('startTimeMillis', 0))
                s_end = int(session.get('endTimeMillis', 0))
                if s_start == 0 or s_end == 0: continue
                
                # Assign the sleep session to the date it started on
                day_str = datetime.fromtimestamp(s_start / 1000.0).strftime('%Y-%m-%d')
                day_data = next((d for d in summary['days'] if d['date'] == day_str), None)
                if day_data:
                    sleep_hours = (s_end - s_start) / 3600000.0  # ms to hr
                    day_data["sleep_hours"] += round(sleep_hours, 2)
        except Exception as e:
            print(f"Failed to fetch sleep sessions: {e}")
            
        return summary
