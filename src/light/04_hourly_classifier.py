#!/usr/bin/env python3
"""
Step 4 - Design comparison (ii) vs (iii).
(ii) per-timestep classifier: every 10 min predict ON/OFF from hour-of-day,
     day-of-year and the weather at that time.
(iii) once-a-day event regressor (Step 2): predict the two switch instants.
Switch times are read off the classifier's predicted state sequence and scored
on the same events, same leave-one-month-out folds.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
warnings.filterwarnings("ignore")

ev = pd.read_csv(PROC / "light_events.csv", parse_dates=["date", "start", "ref"])
ev["start"] = pd.to_datetime(ev["start"], utc=True).dt.tz_convert(TZ)
ev["ref"] = pd.to_datetime(ev["ref"], utc=True).dt.tz_convert(TZ)
df = pd.read_csv(RAW / "light_feeds.csv", low_memory=False, usecols=["created_at", "field1"])
df["t"] = pd.to_datetime(df["created_at"], format="ISO8601")
df["ldr"] = pd.to_numeric(df["field1"], errors="coerce")
df = df[df.ldr.isin([0, 1])].set_index("t").sort_index()
days = set(ev.date.dt.date)
st = df["ldr"].resample("10min").mean().dropna()
st = st[np.isin(st.index.date, list(days))]
X = pd.DataFrame({"y": (st > 0.5).astype(int)}, index=st.index)
hod = X.index.hour + X.index.minute / 60
X["h_sin"], X["h_cos"] = np.sin(2*np.pi*hod/24), np.cos(2*np.pi*hod/24)
X = pd.concat([X, doy_harmonics(X.index.tz_localize(None).normalize()).set_index(X.index)], axis=1)
wx = load_weather()[WEATHER_VARS].resample("10min").interpolate("linear")
X = X.join(wx, how="left")
X["month"] = X.index.month
feats = ["h_sin", "h_cos"] + ASTRO + WEATHER_VARS
X[feats] = X[feats].fillna(X[feats].median())
g = X.month.values
pred = np.full(len(X), np.nan)
for m in np.unique(g):
    tr, te = g != m, g == m
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.4,
                                 class_weight="balanced", random_state=SEED, n_jobs=-1)
    clf.fit(X.loc[tr, feats], X.loc[tr, "y"]); pred[te] = clf.predict(X.loc[te, feats])
X["p"] = pred
acc = (X.p == X.y).mean()

out = []
for _, r in ev.iterrows():
    win = X.loc[r.ref - pd.Timedelta(minutes=150): r.ref + pd.Timedelta(minutes=150), "p"]
    want = 1 if r.kind == "dusk" else 0
    flips = win[(win == want) & (win.shift() == 1 - want)]
    t_hat = flips.index[0] if len(flips) else (r.ref + pd.Timedelta(minutes=150) if r.kind == "dusk" else r.ref - pd.Timedelta(minutes=150))
    out.append(dict(kind=r.kind, true=r.offset_min, pred=(t_hat - r.ref).total_seconds() / 60,
                    found=len(flips) > 0))
O = pd.DataFrame(out)
evp = pd.read_csv(PROC / "light_predictions_loso.csv")
rows = []
for k in ["dusk", "dawn"]:
    o = O[O.kind == k]; e = evp[evp.kind == k]
    rows.append(dict(event=k, design="(ii) per-10-min classifier",
                     MAE_min=mae(o.true.values, o.pred.values), switch_found_pct=100*o.found.mean()))
    rows.append(dict(event=k, design="(iii) once-a-day RF regressor",
                     MAE_min=mae(e.offset_min.values, e["M2 Random Forest"].values), switch_found_pct=100.0))
R = pd.DataFrame(rows); R.to_csv(TAB / "light_design_comparison.csv", index=False)
print(f"10-min bins: {len(X):,}  dark fraction {X.y.mean():.2f}  "
      f"switch bins per day ~2 of 144  ->  classifier state accuracy {acc:.3f}")
print(R.round(2).to_string(index=False))
