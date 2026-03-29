import json
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

def test_fetch():
    creds = Credentials.from_authorized_user_file('token.json')
    service = build('fitness', 'v1', credentials=creds)
    now = datetime.now()
    start = now - timedelta(days=5)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    req_sleep = {
        "aggregateBy": [{"dataTypeName": "com.google.sleep.segment"}],
        "bucketByTime": {"durationMillis": 86400000},
        "startTimeMillis": start_ms,
        "endTimeMillis": end_ms
    }
    
    try:
        sleep_res = service.users().dataset().aggregate(userId="me", body=req_sleep).execute()
        with open("sleep_out.json", "w") as f:
            json.dump(sleep_res, f, indent=2)
        
        total_segs = 0
        for b in sleep_res.get('bucket', []):
            for ds in b.get('dataset', []):
                total_segs += len(ds.get('point', []))
        print(f"Wrote sleep_out.json with {total_segs} segments mapped.")
    except Exception as e:
        print("error", e)
        
if __name__ == "__main__": test_fetch()
