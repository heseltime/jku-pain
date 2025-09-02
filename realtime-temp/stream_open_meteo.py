import time, requests, datetime as dt, os
from dotenv import load_dotenv

# Load .env (for optional LAT/LON overrides)
load_dotenv()

# Defaults: Vienna; override by setting LAT, LON in .env
LAT = float(os.getenv("LAT", "48.2082"))
LON = float(os.getenv("LON", "16.3738"))

URL = "https://api.open-meteo.com/v1/forecast"

def get_current_temp():
    r = requests.get(URL, params={
        "latitude": LAT,
        "longitude": LON,
        "current_weather": True
    }, timeout=10)
    r.raise_for_status()
    cw = r.json()["current_weather"]
    return {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "temp_c": cw["temperature"],
        "windspeed": cw["windspeed"],
        "winddir": cw["winddirection"]
    }

if __name__ == "__main__":
    print(f"Streaming temperature for lat={LAT}, lon={LON} (Ctrl+C to stop)")
    while True:
        try:
            print(get_current_temp())
        except Exception as e:
            print("error:", e)
        time.sleep(60)
