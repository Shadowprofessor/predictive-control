#!/usr/bin/env python3
"""
Step 1 - Extract dusk/dawn switching events from the raw ThingSpeak light log.

Cleaning protocol (reported in the paper, Sec. IV):
  1. keep binary LDR rows only (drops 10-Mar schema-test rows)
  2. invert April polarity (LDR==relay agreement is 0.000 in Apr, 1.000 from May)
  3. debounce: a state must persist >= 20 min to count
  4. drop events next to a logging gap > 120 s, and manual-override periods (mode 2/3)
  5. keep |offset| < 120 min from astronomical sunset/sunrise
  6. keep May onward (April was a test/commissioning month)

Weather features are anchored to the ASTRONOMICAL reference time, which is known
in advance - never to the observed switch instant (that would leak the target).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, pvlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
warnings.filterwarnings("ignore")

DEBOUNCE_MIN, MAX_GAP_S, MAX_OFFSET = 20, 120, 120
WINDOWS = {"dusk": (-180, -60), "dawn": (-60, 60)}   # minutes relative to sunset / sunrise

log = {}
df = pd.read_csv(RAW / "light_feeds.csv", low_memory=False)
log["raw rows"] = len(df)
df["t"] = pd.to_datetime(df["created_at"], format="ISO8601")
df["ldr"] = pd.to_numeric(df["field1"], errors="coerce")
df["mode"] = pd.to_numeric(df["field4"], errors="coerce")
df = df[df["ldr"].isin([0, 1])].sort_values("t").reset_index(drop=True)
log["binary rows"] = len(df)

apr = (df["t"] >= "2026-04-01") & (df["t"] < "2026-05-01")
df.loc[apr, "ldr"] = 1 - df.loc[apr, "ldr"]            # 1 == DARK everywhere
df["gap_prev_s"] = df["t"].diff().dt.total_seconds()
df["mode"] = df["mode"].ffill()

df["run"] = (df["ldr"].diff().fillna(0) != 0).cumsum()
runs = df.groupby("run").agg(start=("t", "first"), end=("t", "last"), state=("ldr", "first"),
                             gap_before=("gap_prev_s", "first"), mode=("mode", "first"))
runs["dur_min"] = (runs["end"] - runs["start"]).dt.total_seconds() / 60
log["raw state runs"] = len(runs)
st = runs[runs["dur_min"] >= DEBOUNCE_MIN].copy()
st["prev"] = st["state"].shift()
ev = st[st["prev"].notna() & (st["state"] != st["prev"])].copy()
ev["kind"] = np.where(ev["state"] == 1, "dusk", "dawn")
ev["date"] = pd.to_datetime(ev["start"].dt.date)
log["debounced transitions"] = len(ev)

site = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)
days = pd.DatetimeIndex(sorted(ev["date"].unique())).tz_localize(TZ)
sun = site.get_sun_rise_set_transit(days, method="spa")
sun.index = sun.index.tz_localize(None).normalize()
ev["sunrise"] = ev["date"].map(sun["sunrise"])
ev["sunset"] = ev["date"].map(sun["sunset"])
ev["ref"] = np.where(ev["kind"] == "dusk", ev["sunset"], ev["sunrise"])
ev["ref"] = pd.to_datetime(ev["ref"], utc=True).dt.tz_convert(TZ)
ev["offset_min"] = (ev["start"] - ev["ref"]).dt.total_seconds() / 60
pos = pvlib.solarposition.get_solarposition(pd.DatetimeIndex(ev["start"]), LAT, LON, altitude=ALT)
ev["sun_elev_at_switch"] = pos["apparent_elevation"].values

ev = ev[ev["offset_min"].abs() < MAX_OFFSET];                       log["within +-120 min"] = len(ev)
ev = ev[(ev["gap_before"] <= MAX_GAP_S) & ~ev["mode"].isin([2, 3])]; log["no gap / no manual"] = len(ev)
ev = ev[ev["date"] >= "2026-05-01"];                                  log["May onward"] = len(ev)
ev = ev[ev["date"].dt.month != 9];                                    log["drop Sept (4 events, outage)"] = len(ev)

wx = load_weather()
hi = wx[WEATHER_VARS].resample("1min").interpolate("linear")
rows = []
for _, r in ev.iterrows():
    a, b = WINDOWS[r["kind"]]
    w = hi.loc[r["ref"] + pd.Timedelta(minutes=a): r["ref"] + pd.Timedelta(minutes=b)].mean()
    rows.append(w.add_suffix("_w"))
ev = pd.concat([ev.reset_index(drop=True), pd.DataFrame(rows).reset_index(drop=True)], axis=1)
ev = pd.concat([ev, doy_harmonics(ev["date"])], axis=1)
ev["month"] = ev["date"].dt.month
ev["doy"] = ev["date"].dt.dayofyear

keep = ["date", "month", "doy", "kind", "start", "ref", "sunrise", "sunset", "offset_min",
        "sun_elev_at_switch"] + ASTRO + [v + "_w" for v in WEATHER_VARS]
ev[keep].to_csv(PROC / "light_events.csv", index=False)
pd.Series(log).to_csv(TAB / "light_cleaning_funnel.csv", header=["count"])

print("Cleaning funnel:")
for k, v in log.items():
    print(f"  {k:32s} {v:>9,}")
print(f"\nfinal: dusk={sum(ev.kind=='dusk')}  dawn={sum(ev.kind=='dawn')}  "
      f"{ev.date.min().date()} -> {ev.date.max().date()}")
print(ev.groupby("kind")[["offset_min", "sun_elev_at_switch"]].agg(["mean", "std"]).round(2))
