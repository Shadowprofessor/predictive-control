"""Shared paths, site constants and helpers."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
WX_DIR = ROOT / "data" / "weather"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
for p in (PROC, FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

# NIIT University, Neemrana, Rajasthan
LAT, LON, ALT, TZ = 27.9614, 76.4018, 320, "Asia/Kolkata"

WEATHER_VARS = [
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance",
    "precipitation", "relative_humidity_2m", "temperature_2m",
    "pm2_5", "aerosol_optical_depth", "dust",
]
ASTRO = ["sin1", "cos1", "sin2", "cos2"]
SEED = 42


def load_weather() -> pd.DataFrame:
    """Hourly Open-Meteo reanalysis + air quality, tz-aware index."""
    wx = pd.read_csv(WX_DIR / "weather_hourly.csv", parse_dates=["time"])
    ae = pd.read_csv(WX_DIR / "aerosol_hourly.csv", parse_dates=["time"])
    wx = wx.merge(ae, on="time", how="left").set_index("time").sort_index()
    wx.index = wx.index.tz_localize(TZ)
    return wx


def doy_harmonics(dates) -> pd.DataFrame:
    d = pd.DatetimeIndex(pd.to_datetime(dates)).dayofyear.values
    return pd.DataFrame({
        "sin1": np.sin(2 * np.pi * d / 365), "cos1": np.cos(2 * np.pi * d / 365),
        "sin2": np.sin(4 * np.pi * d / 365), "cos2": np.cos(4 * np.pi * d / 365),
    })


def leave_one_month_out(X, y, groups, make_model):
    """Blocked CV: each calendar month is held out in turn (no adjacent-day leakage)."""
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        m = make_model()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def mae(y, p):
    return float(np.nanmean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.nanmean((y - p) ** 2)))
