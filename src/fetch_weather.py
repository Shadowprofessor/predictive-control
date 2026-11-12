#!/usr/bin/env python3
"""
Pull Open-Meteo ARCHIVE (reanalysis) weather + irradiance + aerosol for the
Neemrana site, covering the full span of the light-event dataset.

Writes into data/weather/.

    pip install requests pandas
    python src/fetch_weather.py

Outputs:
    weather_hourly.csv   - hourly weather + irradiance
    aerosol_hourly.csv   - hourly PM2.5 / aerosol optical depth
Both are free for non-commercial use and need no API key.
"""
import requests, pandas as pd, sys, time
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "data" / "weather"
OUT.mkdir(parents=True, exist_ok=True)

LAT, LON, TZ = 27.9614, 76.4018, "Asia/Kolkata"   # NIIT University, Neemrana
START, END = "2025-01-01", "2026-11-11"           # 2025 included for the solar sim

WEATHER_VARS = ",".join([
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
    "direct_radiation", "precipitation", "rain", "weather_code",
    "wind_speed_10m", "surface_pressure", "visibility",
])

def get(url, params, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  attempt {i+1} failed: {e}", file=sys.stderr)
            time.sleep(5 * (i + 1))
    raise SystemExit("could not reach Open-Meteo - check your connection")

print(f"Fetching weather archive {START} -> {END} for {LAT},{LON} ...")
j = get("https://archive-api.open-meteo.com/v1/archive", {
    "latitude": LAT, "longitude": LON, "start_date": START, "end_date": END,
    "hourly": WEATHER_VARS, "timezone": TZ,
})
w = pd.DataFrame(j["hourly"])
w["time"] = pd.to_datetime(w["time"])
w.to_csv(OUT / "weather_hourly.csv", index=False)
print(f"  wrote weather_hourly.csv  rows={len(w)}  cols={len(w.columns)}")

# Daily sunrise/sunset straight from the API (cross-check against pvlib)
print("Fetching daily sunrise/sunset ...")
jd = get("https://archive-api.open-meteo.com/v1/archive", {
    "latitude": LAT, "longitude": LON, "start_date": START, "end_date": END,
    "daily": "sunrise,sunset,daylight_duration", "timezone": TZ,
})
pd.DataFrame(jd["daily"]).to_csv(OUT / "sun_daily.csv", index=False)
print("  wrote sun_daily.csv")

# Aerosol / PM2.5 - the monsoon+smog feature. Archive starts 2022-07.
print("Fetching air quality (aerosol) ...")
try:
    ja = get("https://air-quality-api.open-meteo.com/v1/air-quality", {
        "latitude": LAT, "longitude": LON, "start_date": START, "end_date": END,
        "hourly": "pm2_5,pm10,aerosol_optical_depth,dust", "timezone": TZ,
    })
    a = pd.DataFrame(ja["hourly"]); a["time"] = pd.to_datetime(a["time"])
    a.to_csv(OUT / "aerosol_hourly.csv", index=False)
    print(f"  wrote aerosol_hourly.csv  rows={len(a)}")
except SystemExit:
    print("  aerosol fetch failed - continue without it, it is optional")

print(f"\nDone. Files written to {OUT}")
