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
        if os.path.exists(self.token_path):
            self.creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secret_path, SCOPES)
                self.creds = flow.run_local_server(port=0)
            
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

        now = datetime.now()
        if timeframe == "1_month":
            start = now - timedelta(days=30)
            num_days = 30
        else:
            start = now - timedelta(days=7)
            num_days = 7
            
        end_time_ms = int(now.timestamp() * 1000)
        start_time_ms = int(start.timestamp() * 1000)

        summary = {
            "timeframe": timeframe,
            "days": []
        }
        
        for i in range(num_days + 1):
            d = start + timedelta(days=i)
            day_str = d.strftime('%Y-%m-%d')
            b_start_ms = int(datetime.strptime(day_str, '%Y-%m-%d').timestamp() * 1000)
            summary["days"].append({
                "date": day_str,
                "steps": 0,
                "calories": 0.0,
                "heart_points": 0.0,
                "avg_bpm": None,
                "hr_array": {"start_ts": b_start_ms, "interval_ms": 900000, "values": [None]*96},
                "body_temp": None,
                "temp_array": {"start_ts": b_start_ms, "interval_ms": 900000, "values": [None]*96},
                "sleep_hours": 0.0,
                "sleep_stages": {
                    "light": 0.0,
                    "deep": 0.0,
                    "rem": 0.0,
                    "awake": 0.0
                }
            })

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
                pass
                
        array_metrics = [
            ("hr_array", "com.google.heart_rate.bpm"),
            ("temp_array", "com.google.body.temperature")
        ]
        
        for key, type_name in array_metrics:
            req_body = {
                "aggregateBy": [{"dataTypeName": type_name}],
                "bucketByTime": { "durationMillis": 900000 },
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
                        reading = round(val.get("fpVal", 0.0), 1)
                        if reading > 0:
                            idx = int((b_start - day_data[key]["start_ts"]) / 900000)
                            if 0 <= idx < 96:
                                day_data[key]["values"][idx] = reading
            except HttpError:
                pass

        try:
            sleep_req = {
                "aggregateBy": [{"dataTypeName": "com.google.sleep.segment"}],
                "bucketByTime": { "durationMillis": 86400000 },
                "startTimeMillis": start_time_ms,
                "endTimeMillis": end_time_ms
            }
            sleep_res = service.users().dataset().aggregate(userId="me", body=sleep_req).execute()
            
            for bucket in sleep_res.get('bucket', []):
                b_start = int(bucket.get('startTimeMillis', 0))
                day_str = datetime.fromtimestamp(b_start / 1000.0).strftime('%Y-%m-%d')
                
                day_data = next((d for d in summary['days'] if d['date'] == day_str), None)
                if not day_data: continue
                
                for ds in bucket.get('dataset', []):
                    for point in ds.get('point', []):
                        val_struct = point.get('value', [{}])[0]
                        stage_enum = val_struct.get('intVal', 1)
                        
                        p_start = int(point.get('startTimeNanos', 0))
                        p_end = int(point.get('endTimeNanos', 0))
                        dur_hrs = (p_end - p_start) / 1e9 / 3600.0
                        if dur_hrs < 0: continue
                        
                        if stage_enum == 4:
                            day_data["sleep_stages"]["deep"] += dur_hrs
                        elif stage_enum == 3:
                            day_data["sleep_stages"]["light"] += dur_hrs
                        elif stage_enum == 5:
                            day_data["sleep_stages"]["rem"] += dur_hrs
                        elif stage_enum in (2, 6):
                            day_data["sleep_stages"]["awake"] += dur_hrs
                            
        except Exception as e:
            print(f"Failed to fetch sleep stages: {e}")

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
                
                day_str = datetime.fromtimestamp(s_start / 1000.0).strftime('%Y-%m-%d')
                day_data = next((d for d in summary['days'] if d['date'] == day_str), None)
                if day_data:
                    sleep_hours = (s_end - s_start) / 3600000.0
                    day_data["sleep_hours"] += sleep_hours
                    
        except Exception as e:
            print(f"Failed to fetch sleep sessions: {e}")

        for d in summary['days']:
            d["sleep_hours"] = round(d["sleep_hours"], 2)
            d["sleep_stages"]["deep"] = round(d["sleep_stages"]["deep"], 2)
            d["sleep_stages"]["light"] = round(d["sleep_stages"]["light"], 2)
            d["sleep_stages"]["rem"] = round(d["sleep_stages"]["rem"], 2)
            d["sleep_stages"]["awake"] = round(d["sleep_stages"]["awake"], 2)
            
        return summary
