#!/usr/bin/env python3
"""
Tier-1 daily predictor - run once a day (~14:30 IST, before dusk).

  1. pulls the Open-Meteo FORECAST (weather + air quality) for the NIIT site
  2. builds exactly the training features: weather averaged over
     [sunset-180, sunset-60] min for tonight's dusk and
     [sunrise-60, sunrise+60] min for tomorrow's dawn, plus day-of-year harmonics
  3. predicts both offsets with the Random Forests, adds the safety margin
  4. writes the plan to the ThingSpeak CONTROL channel the ESP32 polls

If anything fails nothing is written, and the ESP32 stays on its on-chip
seasonal clock (tier 2). That is the intended graceful degradation.

    export TS_CTRL_WRITE_KEY=XXXXXXXX
    python server/daily_predictor.py            # predict + publish
    python server/daily_predictor.py --dry-run  # predict only

cron (IST):  30 14 * * *  cd /path/predictive-control && python server/daily_predictor.py
"""
import argparse, json, os, sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd, requests, joblib, pvlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import LAT, LON, ALT, TZ, WEATHER_VARS, ASTRO, doy_harmonics  # noqa

MODEL_VERSION = 1
M = ROOT / "server" / "models"
meta = json.load(open(M / "light_model_meta.json"))

def forecast() -> pd.DataFrame:
    common = dict(latitude=LAT, longitude=LON, timezone=TZ, forecast_days=3)
    w = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params={**common,
        "hourly": ",".join(v for v in WEATHER_VARS if v not in ("pm2_5", "aerosol_optical_depth", "dust"))}).json()
    a = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=60, params={**common,
        "hourly": "pm2_5,aerosol_optical_depth,dust"}).json()
    df = pd.DataFrame(w["hourly"]).merge(pd.DataFrame(a["hourly"]), on="time", how="left")
    df["time"] = pd.to_datetime(df["time"]); df = df.set_index("time").tz_localize(TZ)
    return df[WEATHER_VARS].resample("1min").interpolate("linear")

def features(hi, ref, kind, day):
    a, b = meta["windows_min"][kind]
    w = hi.loc[ref + pd.Timedelta(minutes=a): ref + pd.Timedelta(minutes=b)].mean()
    x = doy_harmonics([pd.Timestamp(day)]).iloc[0].to_dict()
    med = meta[kind]["weather_medians"]
    for v in WEATHER_VARS:
        x[v + "_w"] = w[v] if pd.notna(w[v]) else med[v + "_w"]
    return pd.DataFrame([x])[meta["features"]]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--date", help="plan date YYYY-MM-DD (default today)"); args = ap.parse_args()
    day = date.fromisoformat(args.date) if args.date else pd.Timestamp.now(tz=TZ).date()
    site = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)
    sun = site.get_sun_rise_set_transit(pd.DatetimeIndex([day, day + timedelta(days=1)]).tz_localize(TZ), method="spa")
    sunset, sunrise_next = sun["sunset"].iloc[0], sun["sunrise"].iloc[1]
    hi = forecast()
    plan = {}
    for kind, ref, d in [("dusk", sunset, day), ("dawn", sunrise_next, day + timedelta(days=1))]:
        rf = joblib.load(M / f"light_rf_{kind}.joblib")
        raw = float(rf.predict(features(hi, ref, kind, d))[0])
        m = meta[kind]["rf_margin_min"]
        plan[kind] = raw - m if kind == "dusk" else raw + m       # shift toward the safe side
        print(f"{kind}: ref {ref:%H:%M}  model {raw:+.1f} min  margin {m:.0f}  ->  plan {plan[kind]:+.1f} min "
              f"(switch at {(ref + pd.Timedelta(minutes=plan[kind])):%H:%M})")
    if args.dry_run:
        return
    key = os.environ.get("TS_CTRL_WRITE_KEY")
    if not key:
        sys.exit("TS_CTRL_WRITE_KEY not set - nothing published (ESP32 stays on tier 2)")
    r = requests.get("https://api.thingspeak.com/update", timeout=30, params=dict(
        api_key=key, field1=day.strftime("%Y%m%d"), field2=f"{plan['dusk']:.1f}",
        field3=f"{plan['dawn']:.1f}", field4=MODEL_VERSION, field5=0))
    print("published, entry id", r.text)

if __name__ == "__main__":
    main()
