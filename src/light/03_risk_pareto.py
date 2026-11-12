#!/usr/bin/env python3
"""
Step 3 - Risk-tuned actuation. Lighting cost is asymmetric:
  dusk: switching ON late  -> unlit road (safety violation); early -> wasted energy
  dawn: switching OFF early -> unlit road (safety violation); late  -> wasted energy

Each policy predicts an offset and is shifted by a safety margin d (or, for the
quantile forest, uses a quantile q of the per-tree predictions). Sweeping d / q traces
a safety-vs-energy frontier. Compared at equal violation rate.

Lamp power is a parameter (default 40 W LED street luminaire).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
warnings.filterwarnings("ignore")

LAMP_W, TOL_MIN = 40.0, 5.0      # a violation = road unlit for more than 5 min
WF = [v + "_w" for v in WEATHER_VARS]
ev = pd.read_csv(PROC / "light_events.csv", parse_dates=["date"])

def costs(kind, y, switch):
    e = switch - y                           # +ve = later than the true dark/light instant
    dark = np.maximum(0, e) if kind == "dusk" else np.maximum(0, -e)
    waste = np.maximum(0, -e) if kind == "dusk" else np.maximum(0, e)
    return (dark > TOL_MIN).mean(), dark.mean(), waste.mean()

def rf_tree_preds(Xtr, ytr, Xte):
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=0.4,
                               random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    return np.stack([t.predict(Xte) for t in rf.estimators_], axis=1)

curves = []
for kind in ["dusk", "dawn"]:
    s = ev[ev.kind == kind].reset_index(drop=True)
    y, g = s.offset_min.values, s.month.values
    Xa = s[ASTRO].values
    Xaw = np.column_stack([Xa, s[WF].fillna(s[WF].median()).values])
    seasonal = leave_one_month_out(Xa, y, g, lambda: Ridge(alpha=1.0))
    T = np.full((len(y), 500), np.nan)                 # out-of-fold per-tree predictions
    for m in np.unique(g):
        tr, te = g != m, g == m
        T[te] = rf_tree_preds(Xaw[tr], y[tr], Xaw[te])
    rf_mean = T.mean(1)
    sign = -1 if kind == "dusk" else +1                # direction that is "safer"
    for d in np.arange(-30, 121, 1.0):
        for name, p in [("Fixed astronomical clock", np.zeros_like(y)),
                        ("Seasonal clock", seasonal), ("RF virtual photocell", rf_mean)]:
            v, dk, w = costs(kind, y, p + sign * d)
            curves.append(dict(event=kind, policy=name, knob=d, viol_rate=v, dark_min=dk, waste_min=w))
    for q in np.linspace(0.02, 0.98, 49):
        qq = q if kind == "dawn" else 1 - q            # dawn: upper quantile = later off = safer
        p = np.quantile(T, qq, axis=1)
        v, dk, w = costs(kind, y, p)
        curves.append(dict(event=kind, policy="Quantile RF (risk-tuned)", knob=q, viol_rate=v, dark_min=dk, waste_min=w))

C = pd.DataFrame(curves); C.to_csv(TAB / "light_pareto_curves.csv", index=False)

def waste_at(Cp, target):
    ok = Cp[Cp.viol_rate <= target]
    return ok.waste_min.min() if len(ok) else np.nan

rows = []
for target in [0.20, 0.10, 0.05]:
    for pol in C.policy.unique():
        wd = waste_at(C[(C.event == "dusk") & (C.policy == pol)], target)
        wa = waste_at(C[(C.event == "dawn") & (C.policy == pol)], target)
        kwh = (wd + wa) * 365 / 60 * LAMP_W / 1000
        rows.append(dict(max_violation_rate=target, policy=pol, dusk_waste_min=wd,
                         dawn_waste_min=wa, daily_waste_min=wd + wa, kWh_per_lamp_year=kwh))
T2 = pd.DataFrame(rows)
for target in T2.max_violation_rate.unique():
    blk = T2.max_violation_rate == target
    ref = T2[blk & (T2.policy == "Fixed astronomical clock")].kWh_per_lamp_year.values[0]
    T2.loc[blk, "saving_vs_fixed_pct"] = 100 * (1 - T2.loc[blk, "kWh_per_lamp_year"] / ref)
T2.to_csv(TAB / "light_energy_at_equal_safety.csv", index=False)
print(f"Energy wasted at EQUAL SAFETY (violation = unlit > {TOL_MIN:.0f} min, lamp {LAMP_W:.0f} W)\n")
print(T2.round(2).to_string(index=False))
