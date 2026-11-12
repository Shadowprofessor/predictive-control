#!/usr/bin/env python3
"""
Solar Step 3 - Three PV forecasting models on the SIMULATED series (labelled).

Horizons
  H1  : 1 hour ahead     (inputs: past power/kt, weather observed now, clear-sky at target)
  H24 : day ahead (24 h) (inputs: today's power, clear-sky at target, and an EMULATED
        day-ahead weather forecast = reanalysis + realistic noise: GHI x(1+0.25e), cloud +-15 pp)
  E   : daily energy, built by summing the H24 hourly predictions per day

Models
  M1 XGBoost            - gradient-boosted trees on tabular lag + weather features
  M2 LSTM               - 1-layer LSTM over the last 48 h, plus target-hour covariates
  M3 Physics-hybrid     - Ridge predicts clear-sky index kt; pvlib clear-sky power x kt
Baselines
  Persistence, Smart persistence (kt persistence x clear-sky power at target)

Split: train 2025, test 2026-01-01..2026-11-11 (strictly forward). Daytime scoring only.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
warnings.filterwarnings("ignore")
PDC0 = 50.0
try:
    import torch, torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    from sklearn.neural_network import MLPRegressor

d = pd.read_csv(PROC / "solar_simulated_hourly.csv", index_col=0, parse_dates=True)
d.index = pd.to_datetime(d.index, utc=True).tz_convert(TZ)
d["kt_pv"] = d["kt_pv"].fillna(0)
hod = d.index.hour.values
d["h_sin"], d["h_cos"] = np.sin(2*np.pi*hod/24), np.cos(2*np.pi*hod/24)
dh = doy_harmonics(d.index.tz_localize(None).normalize()); dh.index = d.index
d = d.join(dh)
# Emulated day-ahead NWP forecast. Real forecast error is persistent, not i.i.d.:
# a day-level bias plus an AR(1) hourly error on the clear-sky index, and a
# correlated cloud-cover error. The resulting GHI forecast error is printed so it
# can be compared with published day-ahead NWP skill.
rs = np.random.RandomState(SEED)
FC_DAY_SD, FC_HR_SD, FC_RHO = 0.12, 0.15, 0.7
day_id = pd.factorize(d.index.date)[0]
day_bias = rs.randn(day_id.max() + 1)[day_id] * FC_DAY_SD
ar = np.zeros(len(d)); e = rs.randn(len(d)) * FC_HR_SD * np.sqrt(1 - FC_RHO**2)
for i in range(1, len(d)):
    ar[i] = FC_RHO * ar[i-1] + e[i]
kt_ghi = (d.ghi / d.ghi_clearsky.replace(0, np.nan)).fillna(0).clip(0, 1.2)
d["kt_ghi_fc"] = (kt_ghi + day_bias + ar).clip(0, 1.2)
d["ghi_fc"] = d.kt_ghi_fc * d.ghi_clearsky
d["cloud_fc"] = (d.cloud - 100 * (day_bias + ar)).clip(0, 100)       # cloud error anti-correlated with kt error
_dt = d.ghi_clearsky > 50
FC_GHI_NRMSE = 100 * np.sqrt(((d.ghi_fc - d.ghi)[_dt] ** 2).mean()) / d.ghi[_dt].mean()

def make(h):
    f = pd.DataFrame(index=d.index)
    f["y"] = d.p_sim_W.shift(-h)
    f["pcs_tgt"] = d.p_clearsky_W.shift(-h)
    f["elev_tgt"] = d.sun_elev.shift(-h)
    f["h_sin_tgt"], f["h_cos_tgt"] = d.h_sin.shift(-h), d.h_cos.shift(-h)
    for c in ASTRO: f[c] = d[c]
    for L in [0, 1, 2, 23]:
        f[f"p_l{L}"] = d.p_sim_W.shift(L); f[f"kt_l{L}"] = d.kt_pv.shift(L)
    f["cloud_now"], f["rh_now"], f["temp_now"] = d.cloud, d.rh, d.temp_air
    f["E_today_so_far"] = d.p_sim_W.groupby(d.index.date).cumsum()
    if h >= 24:                                   # day-ahead: emulated NWP for target hour
        f["ghi_fc_tgt"], f["cloud_fc_tgt"] = d.ghi_fc.shift(-h), d.cloud_fc.shift(-h)
        f["kt_fc_tgt"] = d.kt_ghi_fc.shift(-h)
        f["temp_fc_tgt"] = d.temp_air.shift(-h)
    f["persist"] = d.p_sim_W if h < 24 else d.p_sim_W     # same hour (h=24 -> same hour today)
    f["smart_persist"] = d.kt_pv * f["pcs_tgt"]
    f["kt_tgt"] = d.kt_pv.shift(-h)
    return f.dropna()

def lstm_predict(h, Ftr, Fte, cov_cols, seq=48, epochs=25):
    seq_cols = ["p_sim_W", "kt_pv", "p_clearsky_W", "h_sin", "h_cos", "cloud"]
    S = d[seq_cols].copy(); mu, sd = S.mean(), S.std().replace(0, 1); S = ((S - mu) / sd).values
    pos = {t: i for i, t in enumerate(d.index)}
    def build(F):
        idx = np.array([pos[t] for t in F.index]); ok = idx >= seq
        win = np.stack([S[i-seq+1:i+1] for i in idx[ok]])
        return win, F[cov_cols].values[ok], F["y"].values[ok], ok
    Xs_tr, Xc_tr, y_tr, _ = build(Ftr); Xs_te, Xc_te, y_te, ok_te = build(Fte)
    cs = StandardScaler().fit(Xc_tr); Xc_tr, Xc_te = cs.transform(Xc_tr), cs.transform(Xc_te)
    if not HAVE_TORCH:
        X1 = np.column_stack([Xs_tr.reshape(len(Xs_tr), -1), Xc_tr])
        X2 = np.column_stack([Xs_te.reshape(len(Xs_te), -1), Xc_te])
        m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=SEED).fit(X1, y_tr)
        p = np.full(len(Fte), np.nan); p[ok_te] = m.predict(X2); return p, "MLP (torch unavailable)"
    torch.manual_seed(SEED)
    class Net(nn.Module):
        def __init__(s, ns, nc):
            super().__init__(); s.l = nn.LSTM(ns, 32, batch_first=True)
            s.h = nn.Sequential(nn.Linear(32 + nc, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(s, xs, xc):
            o, _ = s.l(xs); return s.h(torch.cat([o[:, -1], xc], 1)).squeeze(1)
    net = Net(Xs_tr.shape[2], Xc_tr.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3)
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    nv = int(0.1 * len(y_tr)); tr_s, tr_c, tr_y = T(Xs_tr[:-nv]), T(Xc_tr[:-nv]), T(y_tr[:-nv])
    va_s, va_c, va_y = T(Xs_tr[-nv:]), T(Xc_tr[-nv:]), T(y_tr[-nv:])
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(tr_y))
        for i in range(0, len(perm), 256):
            b = perm[i:i+256]; opt.zero_grad()
            loss = ((net(tr_s[b], tr_c[b]) - tr_y[b]) ** 2).mean(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = ((net(va_s, va_c) - va_y) ** 2).mean().item()
        if v < best: best, best_state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(): pe = net(T(Xs_te), T(Xc_te)).numpy()
    p = np.full(len(Fte), np.nan); p[ok_te] = np.clip(pe, 0, None); return p, "LSTM"

rows, daily_rows, keep_pred = [], [], {}
for h in [1, 24]:
    F = make(h)
    tr = F.index < pd.Timestamp("2026-01-01", tz=TZ); te = ~tr
    Ftr, Fte = F[tr], F[te]
    tab = [c for c in F.columns if c not in ("y", "persist", "smart_persist", "kt_tgt")]
    P = {"Persistence": Fte.persist.values, "Smart persistence": Fte.smart_persist.values}
    xgb = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
                       colsample_bytree=0.8, random_state=SEED, n_jobs=4).fit(Ftr[tab], Ftr.y)
    P["M1 XGBoost"] = np.clip(xgb.predict(Fte[tab]), 0, None)
    P["M2 " + ("LSTM" if HAVE_TORCH else "MLP")], _ = lstm_predict(h, Ftr, Fte, tab)
    kt_cols = [c for c in tab if c.startswith("kt_") or c.startswith("cloud") or c in ("h_sin_tgt", "h_cos_tgt", "elev_tgt") + tuple(ASTRO)]
    day_tr = Ftr.pcs_tgt > 2
    rg = make_ridge = Ridge(alpha=1.0).fit(Ftr.loc[day_tr, kt_cols], Ftr.loc[day_tr, "kt_tgt"])
    P["M3 Physics-hybrid (kt x clear-sky)"] = np.clip(rg.predict(Fte[kt_cols]), 0, 1.3) * Fte.pcs_tgt.values
    day = (Fte.pcs_tgt > 2).values; y = Fte.y.values
    sp_rmse = rmse(y[day], P["Smart persistence"][day])
    for n, p in P.items():
        ok = day & ~np.isnan(p)
        rows.append(dict(horizon=f"{h} h", model=n, MAE_W=mae(y[ok], p[ok]), RMSE_W=rmse(y[ok], p[ok]),
                         nRMSE_pct=100 * rmse(y[ok], p[ok]) / PDC0,
                         skill_vs_smart_persistence=1 - rmse(y[ok], p[ok]) / sp_rmse))
    keep_pred[h] = pd.DataFrame({"y": y, **P}, index=Fte.index)
    if h == 24:   # daily energy (kWh) from summed hourly day-ahead forecasts
        K = keep_pred[24].copy(); K.index = K.index + pd.Timedelta(hours=24)
        K = K.fillna(0).resample("D").sum() / 1000
        K = K[(K.y > 0)]
        for n in P:
            daily_rows.append(dict(model=n, MAE_kWh=mae(K.y.values, K[n].values),
                                   MAPE_pct=100 * np.mean(np.abs(K.y - K[n]) / K.y),
                                   RMSE_kWh=rmse(K.y.values, K[n].values)))
        K.to_csv(PROC / "solar_daily_energy_predictions.csv")
    keep_pred[h].to_csv(PROC / f"solar_predictions_h{h}.csv")

R = pd.DataFrame(rows); R.to_csv(TAB / "solar_sim_model_results.csv", index=False)
DR = pd.DataFrame(daily_rows); DR.to_csv(TAB / "solar_sim_daily_energy_results.csv", index=False)
print("*** SIMULATED DATA *** train 2025, test 2026, daytime hours only\n")
print(R.round(3).to_string(index=False))
print("\nDaily energy from day-ahead hourly forecasts (test 2026):")
print(DR.round(4).to_string(index=False))
print(f"\nemulated day-ahead GHI forecast error: nRMSE {FC_GHI_NRMSE:.1f}% of mean daytime GHI")
print(f"neural model backend: {'PyTorch LSTM' if HAVE_TORCH else 'sklearn MLP fallback'}")
