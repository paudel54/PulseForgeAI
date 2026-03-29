import json
import sys
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

def test_fit():
    try:
        creds = Credentials.from_authorized_user_file('token.json')
        service = build('fitness', 'v1', credentials=creds)
        
        now = datetime.now()
        start = now - timedelta(days=2) # Last 2 days to capture everything
        
        end_time_ms = int(now.timestamp() * 1000)
        start_time_ms = int(start.timestamp() * 1000)

        metrics = [
            # ("steps", "com.google.step_count.delta"),
            # ("calories", "com.google.calories.expended"),
            ("heart_points", "com.google.heart_minutes.summary"),
            ("avg_bpm", "com.google.heart_rate.bpm"),
            ("body_temp", "com.google.body.temperature"),
            ("sleep", "com.google.sleep.segment")
        ]
        
        results = {}
        for key, type_name in metrics:
            req_body = {
                "aggregateBy": [{"dataTypeName": type_name}],
                "bucketByTime": { "durationMillis": 86400000 },
                "startTimeMillis": start_time_ms,
                "endTimeMillis": end_time_ms
            }
            try:
                res = service.users().dataset().aggregate(userId="me", body=req_body).execute()
                results[key] = res
            except Exception as e:
                results[key] = {"error": str(e)}
                
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print("Fatal error:", e)

if __name__ == "__main__":
    test_fit()
