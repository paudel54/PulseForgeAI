import json
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

def test_fetch():
    creds = Credentials.from_authorized_user_file('token.json')
    service = build('fitness', 'v1', credentials=creds)
    
    now = datetime.now()
    start = now - timedelta(days=2) # Test on 2 days
    
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)
    
    # 1. Test Sleep Stages using aggregate with session
    req_sleep = {
        "aggregateBy": [{"dataTypeName": "com.google.sleep.segment"}],
        "startTimeMillis": start_ms,
        "endTimeMillis": end_ms
    }
    
    try:
        sleep_res = service.users().dataset().aggregate(userId="me", body=req_sleep).execute()
        with open("sleep_out.json", "w") as f:
            json.dump(sleep_res, f, indent=2)
        print("Wrote sleep_out.json")
    except Exception as e:
        print("Sleep Error:", e)

    # 2. Test raw derived HR reading (Using the REST API 'users.dataSources.datasets.get')
    # First we need the data source. Derived HR is usually:
    hr_ds = "derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm"
    temp_ds = "derived:com.google.body.temperature:com.google.android.gms:merge_temperature"
    dataset_id = f"{start_ms}000000-{end_ms}000000"
    
    try:
        hr_raw = service.users().dataSources().datasets().get(
            userId="me", 
            dataSourceId=hr_ds, 
            datasetId=dataset_id
        ).execute()
        with open("hr_raw.json", "w") as f:
            json.dump(hr_raw, f, indent=2)
        print(f"Wrote hr_raw.json ({len(hr_raw.get('point', []))} points)")
    except Exception as e:
        print("HR Raw Error:", e)

if __name__ == "__main__":
    test_fetch()
