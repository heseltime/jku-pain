# stream_extreme_temp.py
import asyncio, time, math
import httpx
from collections import defaultdict
from datetime import datetime, timezone

# Example watchpoints (lat, lon, label). Add more or generate a grid.
WATCHPOINTS = [
    (48.2082, 16.3738, "Vienna"),
    (51.5074, -0.1278, "London"),
    (40.7128, -74.0060, "New York"),
    (35.6895, 139.6917, "Tokyo"),
    (-33.8688, 151.2093, "Sydney"),
    (28.6139, 77.2090, "Delhi"),
    (-23.5505, -46.6333, "São Paulo"),
    (30.0444, 31.2357, "Cairo"),
]

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
POLL_SECONDS = 60 
SPIKE_THRESHOLD = 5.0  # °C per ~hour (approx since we poll often)
ABS_HOT = 40.0
ABS_COLD = -25.0

# store last seen temps by key
last_temp = {}
last_time = {}

def ts():
    return datetime.now(timezone.utc).isoformat()

async def fetch_point(client, lat, lon, label):
    r = await client.get(OPEN_METEO_URL, params={
        "latitude": lat,
        "longitude": lon,
        "current_weather": True
    }, timeout=15.0)
    r.raise_for_status()
    j = r.json()["current_weather"]
    return {
        "label": label,
        "lat": lat,
        "lon": lon,
        "temp_c": j["temperature"],
        "windspeed": j["windspeed"],
        "winddir": j["winddirection"],
        "provider_time": j.get("time"),  # ISO8601
        "received_time": ts(),
    }

def check_extreme_movement(key, curr):
    alerts = []
    t = curr["temp_c"]
    # absolute extremes
    if t >= ABS_HOT:
        alerts.append(f"🔥 Absolute hot extreme: {t:.1f} °C at {curr['label']}")
    if t <= ABS_COLD:
        alerts.append(f"🧊 Absolute cold extreme: {t:.1f} °C at {curr['label']}")

    # spikes vs previous reading (~last 10 minutes); scale to per-hour equivalent
    if key in last_temp and key in last_time:
        dt_minutes = (datetime.fromisoformat(curr["received_time"]) -
                      datetime.fromisoformat(last_time[key])).total_seconds() / 60.0
        if dt_minutes > 0:
            per_hour = (t - last_temp[key]) * (60.0/dt_minutes)
            if abs(per_hour) >= SPIKE_THRESHOLD:
                arrow = "⬆️" if per_hour > 0 else "⬇️"
                alerts.append(f"{arrow} Rapid change: {per_hour:+.1f} °C/hr at {curr['label']} (now {t:.1f} °C)")
    # update state
    last_temp[key] = t
    last_time[key] = curr["received_time"]
    return alerts

async def poll_loop():
    async with httpx.AsyncClient() as client:
        print(f"{ts()} Starting stream for {len(WATCHPOINTS)} locations (Ctrl+C to stop)")
        while True:
            try:
                tasks = [fetch_point(client, lat, lon, label) for lat, lon, label in WATCHPOINTS]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for wp, res in zip(WATCHPOINTS, results):
                    lat, lon, label = wp
                    key = f"{lat:.2f},{lon:.2f}"
                    if isinstance(res, Exception):
                        print(f"{ts()} error {label}: {res}")
                        continue
                    alerts = check_extreme_movement(key, res)
                    # Always print the sample; alerts only when triggered
                    print(f"{ts()} {label}: {res['temp_c']:.1f} °C (ws {res['windspeed']:.0f} km/h)")
                    for a in alerts:
                        # Here you could push to Kafka, Redis Streams, Slack, etc.
                        print(f"{ts()} ALERT: {a}")
            except Exception as e:
                print(f"{ts()} loop error: {e}")
            await asyncio.sleep(POLL_SECONDS)

if __name__ == "__main__":
    asyncio.run(poll_loop())
