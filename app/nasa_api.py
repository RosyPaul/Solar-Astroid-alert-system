import requests
import os
from datetime import date, timedelta
from dotenv import load_dotenv
from model import predict_flare

load_dotenv()
NASA_API_KEY = os.getenv("NASA_API_KEY")

def fetch_todays_flares():
    today = str(date.today())
    week_ago = str(date.today() - timedelta(days=7))
    
    url = "https://api.nasa.gov/DONKI/FLR"
    params = {
        "startDate": week_ago,
        "endDate": today,
        "api_key": NASA_API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if not data:
        print("No flares detected in last 7 days!")
        return []
    
    flares = []
    for flare in data:
        try:
            class_type = flare.get('classType', 'C1.0')
            flare_class = class_type[0]
            flare_intensity = float(class_type[1:])
            begin_time = flare.get('beginTime', '')
            
            flares.append({
                'flrID': flare.get('flrID'),
                'flare_class': flare_class,
                'flare_intensity': flare_intensity,
                'duration_minutes': 21.0,
                'activeRegionNum': float(flare.get('activeRegionNum') or 0),
                'has_linked_events': 1 if flare.get('linkedEvents') else 0,
                'hour': int(begin_time[11:13]) if begin_time else 0,
                'month': int(begin_time[5:7]) if begin_time else 1,
                'dayofweek': date.today().weekday(),
                'beginTime': begin_time,
                'sourceLocation': flare.get('sourceLocation', 'Unknown')
            })
        except Exception as e:
            print(f"Skipping flare due to error: {e}")
            continue
    
    return flares

if __name__ == "__main__":
    flares = fetch_todays_flares()
    print(f"Flares in last 7 days: {len(flares)}\n")
    
    for flare in flares:
        result = predict_flare(flare.copy())
        print(f"{flare['beginTime']} | {flare['flare_class']}{flare['flare_intensity']} | {result['label']} | {result['probability']*100:.1f}%")