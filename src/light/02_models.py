#!/usr/bin/env python3
"""
Step 2 - Virtual-photocell models: predict tonight's switch offset (min from
astronomical sunset/sunrise).

Baselines: B1 fixed astronomical clock, B2 calibrated constant offset,
           B3 seasonal clock (day-of-year harmonics, Ridge)
Models   : M1 Ridge, M2 Random Forest, M3 XGBoost - all on astro + weather
Protocols: leave-one-month-out CV (primary), forward split May-Jul -> Aug+Oct,
           paired bootstrap CI vs the seasonal clock, forecast-noise robustness.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
warnings.filterwarnings("ignore")

WF = [v + "_w" for v in WEATHER_VARS]
MODELS = {
    "M1 Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "M2 Random Forest": lambda: RandomForestRegressor(
        n_estimators=500, min_samples_leaf=4, max_features=0.4, random_state=SEED, n_jobs=-1),
    "M3 XGBoost": lambda: XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8,
        colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED, n_jobs=2),
}
SEASONAL = lambda: Ridge(alpha=1.0)

ev = pd.read_csv(PROC / "light_events.csv", parse_dates=["date"])
rows, preds, sig = [], [], []
for kind in ["dusk", "dawn"]:
    s = ev[ev.kind == kind].reset_index(drop=True)
    y, g = s.offset_min.values, s.month.values
    Xa = s[ASTRO].values
    Xw = s[WF].fillna(s[WF].median()).values
    Xaw = np.column_stack([Xa, Xw])

    P = {"B1 Fixed astronomical clock": np.zeros_like(y),
         "B2 Calibrated clock": np.array([np.median(y[g != m]) for m in g]),
         "B3 Seasonal clock": leave_one_month_out(Xa, y, g, SEASONAL)}
    for n, mk in MODELS.items():
        P[n] = leave_one_month_out(Xaw, y, g, mk)

    # forward split: train May-Jul, test Aug + Oct
    tr = np.isin(g, [5, 6, 7]); te = ~tr
    fwd = {"B3 Seasonal clock": SEASONAL().fit(Xa[tr], y[tr]).predict(Xa[te])}
    for n, mk in MODELS.items():
        fwd[n] = mk().fit(Xaw[tr], y[tr]).predict(Xaw[te])

    base = mae(y, P["B3 Seasonal clock"])
    for n, p in P.items():
        rows.append(dict(event=kind, model=n, MAE_min=mae(y, p), RMSE_min=rmse(y, p),
                         gain_vs_fixed_pct=100 * (1 - mae(y, p) / mae(y, P["B1 Fixed astronomical clock"])),
                         gain_vs_seasonal_pct=100 * (1 - mae(y, p) / base),
                         fwd_MAE_min=mae(y[te], fwd[n]) if n in fwd else np.nan))
    # paired bootstrap vs seasonal clock
    rs = np.random.RandomState(SEED)
    for n in MODELS:
        d = np.abs(y - P["B3 Seasonal clock"]) - np.abs(y - P[n])
        bs = [d[rs.randint(0, len(d), len(d))].mean() for _ in range(5000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig.append(dict(event=kind, model=n, mean_gain_min=d.mean(), ci_lo=lo, ci_hi=hi,
                        significant=lo > 0, win_rate=(d > 0).mean()))
    # forecast-noise robustness for each model
    for noise in [0.15, 0.30]:
        rsn = np.random.RandomState(0)
        Xn = np.column_stack([Xa, Xw * (1 + noise * rsn.randn(*Xw.shape))])
        for n, mk in MODELS.items():
            p = leave_one_month_out(Xn, y, g, mk)
            rows.append(dict(event=kind, model=f"{n} [forecast noise {int(noise*100)}%]",
                             MAE_min=mae(y, p), RMSE_min=rmse(y, p),
                             gain_vs_fixed_pct=100 * (1 - mae(y, p) / mae(y, P["B1 Fixed astronomical clock"])),
                             gain_vs_seasonal_pct=100 * (1 - mae(y, p) / base), fwd_MAE_min=np.nan))
    out = s[["date", "kind", "month", "offset_min"]].copy()
    for n, p in P.items():
        out[n] = p
    preds.append(out)
    # feature importance from the full-data random forest
    rf = MODELS["M2 Random Forest"]().fit(Xaw, y)
    imp = pd.Series(rf.feature_importances_, index=ASTRO + WF).sort_values(ascending=False)
    imp.to_csv(TAB / f"light_feature_importance_{kind}.csv", header=["importance"])

res = pd.DataFrame(rows); res.to_csv(TAB / "light_model_results.csv", index=False)
sg = pd.DataFrame(sig); sg.to_csv(TAB / "light_significance.csv", index=False)
pd.concat(preds).to_csv(PROC / "light_predictions_loso.csv", index=False)

pd.set_option("display.width", 200)
print(res[~res.model.str.contains("noise")].round(2).to_string(index=False))
print("\nforecast-noise robustness:")
print(res[res.model.str.contains("noise")][["event", "model", "MAE_min", "gain_vs_seasonal_pct"]].round(2).to_string(index=False))
print("\npaired bootstrap vs seasonal clock (95% CI, minutes of MAE saved):")
print(sg.round(2).to_string(index=False))
