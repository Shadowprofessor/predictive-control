#!/usr/bin/env python3
"""Paper figures (IEEE single column = 3.5 in, double = 7.16 in). PNG 300 dpi + PDF."""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
warnings.filterwarnings("ignore")

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
GRAY, INK, INK2, GRID = "#898781", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": GRAY,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 150})
POL = {"Fixed astronomical clock": GRAY, "Seasonal clock": ORANGE,
       "RF virtual photocell": BLUE, "Quantile RF (risk-tuned)": AQUA}
PAPER_NAMES = {"Fixed astronomical clock": "Fixed clock", "RF virtual photocell": "Random Forest",
               "Quantile RF (risk-tuned)": "Quantile RF", "M1 XGBoost": "XGBoost", "M2 MLP": "Neural network",
               "M2 LSTM": "Neural network", "M3 Physics-hybrid (kt x clear-sky)": "Physics hybrid"}

PAPER = os.environ.get("PAPER") == "1"      # PAPER=1: serif text, panel labels instead of titles
if PAPER:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7})
    FIG = Path(os.environ.get("PAPER_FIG_DIR", str(FIG)))
    FIG.mkdir(parents=True, exist_ok=True)

def title(ax, full, short):
    ax.set_title(short if PAPER else full, loc="left", color=INK)

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig); print("  ", name)

ev = pd.read_csv(PROC / "light_events.csv", parse_dates=["date"])

# Fig. 3 - offset vs day of year with seasonal fit
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3), sharex=True)
for ax, kind, col in [(axs[0], "dusk", BLUE), (axs[1], "dawn", ORANGE)]:
    s = ev[ev.kind == kind]
    ax.scatter(s.doy, s.offset_min, s=9, color=col, alpha=0.75, edgecolor="white", linewidth=0.4, label="Observed photocell event")
    rg = Ridge(alpha=1.0).fit(s[ASTRO], s.offset_min)
    dd = np.arange(s.doy.min(), s.doy.max() + 1)
    ax.plot(dd, rg.predict(doy_harmonics(pd.to_datetime("2026-01-01") + pd.to_timedelta(dd - 1, "D"))), color=INK, lw=1.4, label="Seasonal clock (fit)")
    ax.axhline(0, color=GRAY, lw=1, ls="--", label=f"Astronomical {'sunset' if kind=='dusk' else 'sunrise'}")
    title(ax, f"{kind.capitalize()}: switch time relative to {'sunset' if kind=='dusk' else 'sunrise'}", f"({'a' if kind=='dusk' else 'b'}) {kind.capitalize()}")
    ax.set_xlabel("Day of year (2026)"); ax.set_ylabel("Offset (min)")
axs[0].legend(loc="lower right")
save(fig, "fig3_offset_vs_doy")

# Fig. 4 - residual spread by month (monsoon signature)
fig, ax = plt.subplots(figsize=(3.5, 2.1))
w = 0.38; months = [5, 6, 7, 8, 10]; lab = ["May", "Jun", "Jul", "Aug", "Oct"]
for i, (kind, col) in enumerate([("dusk", BLUE), ("dawn", ORANGE)]):
    s = ev[ev.kind == kind].copy()
    s["res"] = s.offset_min - Ridge(alpha=1.0).fit(s[ASTRO], s.offset_min).predict(s[ASTRO])
    sd = s.groupby("month").res.std().reindex(months)
    ax.bar(np.arange(5) + (i - 0.5) * w, sd.values, w - 0.03, color=col, label=kind.capitalize())
ax.set_xticks(np.arange(5)); ax.set_xticklabels(lab)
ax.set_ylabel("Residual SD after seasonal fit (min)")
title(ax, "Day-to-day scatter peaks in the monsoon", "")
ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
save(fig, "fig4_monsoon_residual")

# Fig. 5 - model comparison
R = pd.read_csv(TAB / "light_model_results.csv"); R = R[~R.model.str.contains("noise")]
order = ["B1 Fixed astronomical clock", "B2 Calibrated clock", "B3 Seasonal clock", "M1 Ridge", "M2 Random Forest", "M3 XGBoost"]
short = ["Fixed\nclock", "Calib.\nclock", "Seasonal\nclock", "Ridge", "Random\nForest", "XGBoost"]
fig, ax = plt.subplots(figsize=(3.5, 2.2))
for i, (kind, col) in enumerate([("dusk", BLUE), ("dawn", ORANGE)]):
    v = R[R.event == kind].set_index("model").loc[order, "MAE_min"].values
    b = ax.bar(np.arange(6) + (i - 0.5) * w, v, w - 0.03, color=col, label=kind.capitalize())
    for x, y in zip(np.arange(6) + (i - 0.5) * w, v):
        ax.text(x, y + 0.4, f"{y:.1f}", ha="center", va="bottom", fontsize=5.6, color=INK2)
ax.axvline(2.5, color=GRAY, lw=0.8, ls=":")
ax.text(1.0, 31, "baselines", ha="center", color=INK2, fontsize=6.5); ax.text(4.0, 31, "astro + weather", ha="center", color=INK2, fontsize=6.5)
ax.set_xticks(np.arange(6)); ax.set_xticklabels(short, fontsize=6.3); ax.set_ylim(0, 33)
ax.set_ylabel("MAE (min), leave-one-month-out"); ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))
save(fig, "fig5_model_comparison")

# Fig. 6 - safety vs energy frontier
C = pd.read_csv(TAB / "light_pareto_curves.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.4))
for ax, kind in zip(axs, ["dusk", "dawn"]):
    for pol, col in POL.items():
        c = C[(C.event == kind) & (C.policy == pol)].sort_values("viol_rate")
        # efficient frontier: least waste achievable at a violation rate <= v
        c = c.groupby(c.viol_rate.round(3)).waste_min.min().reset_index().sort_values("viol_rate")
        c["env"] = c.waste_min.cummin()
        ax.step(100 * c.viol_rate, c.env, where="post", color=col, lw=1.8,
                ls="--" if pol.startswith("Fixed") else "-", label=PAPER_NAMES.get(pol, pol) if PAPER else pol)
    ax.axvline(5, color=GRAY, lw=0.8, ls=":"); ax.axvline(10, color=GRAY, lw=0.8, ls=":")
    ax.set_xlim(0, 40); ax.set_ylim(0, 70)
    ax.set_xlabel("Nights with road unlit > 5 min (%)"); ax.set_ylabel("Wasted lamp-on time (min / event)")
    title(ax, f"{kind.capitalize()}: safety vs energy frontier (lower-left is better)", f"({'a' if kind=='dusk' else 'b'}) {kind.capitalize()}")
axs[1].legend(loc="upper right")
save(fig, "fig6_pareto")

# Fig. 7 - measured solar: V vs cell temperature, and DHT11 heating bias
q = pd.read_csv(PROC / "solar_measured_15min.csv", index_col=0, parse_dates=True)
day = q[(q.sun_elev > 10) & (q.shortwave_radiation > 50) & (q.v_panel > 1)]
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw={"wspace": 0.55})
sc = axs[0].scatter(day.t_cell, day.v_panel, c=day.shortwave_radiation, cmap="Blues", s=5, vmin=0, vmax=1000, edgecolor="none")
cb = fig.colorbar(sc, ax=axs[0], pad=0.02); cb.set_label("GHI (W/m²)", color=INK2); cb.outline.set_visible(False)
axs[0].set_xlabel("Estimated cell temperature (°C)"); axs[0].set_ylabel("Panel open-circuit voltage (V)")
title(axs[0], "MEASURED Feb–Apr 2026: Voc vs cell temperature", "(a)")
axs[0].set_ylim(12, 25)
b = pd.read_csv(TAB / "solar_dht_heating_bias.csv")
labels = ["≤10", "10–\n200", "200–\n400", "400–\n600", "600–\n800", ">800"]
axs[1].bar(range(6), b["mean"], 0.7, color=ORANGE)
for i, v in enumerate(b["mean"]):
    axs[1].text(i, v + 0.12, f"+{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
axs[1].set_xticks(range(6)); axs[1].set_xticklabels(labels)
axs[1].set_xlabel("Global horizontal irradiance (W/m²)"); axs[1].set_ylabel("DHT11 − reanalysis air temp (°C)")
title(axs[1], "MEASURED: enclosure solar heating of the DHT11", "(b)"); axs[1].grid(axis="x", visible=False)
save(fig, "fig7_measured_solar")

# Fig. 8 - simulated PV forecasting (only if Step 3 has run)
if (TAB / "solar_sim_model_results.csv").exists():
    S = pd.read_csv(TAB / "solar_sim_model_results.csv")
    names = S.model.unique().tolist()
    lab = (lambda n: PAPER_NAMES.get(n, n)) if PAPER else (lambda n: n)
    cols = {n: c for n, c in zip(names, [GRAY, INK2, BLUE, ORANGE, AQUA])}
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.4), gridspec_kw={"width_ratios": [1, 1.5]})
    for i, h in enumerate(["1 h", "24 h"]):
        v = S[S.horizon == h].set_index("model").loc[names, "nRMSE_pct"].values
        for j, n in enumerate(names):
            axs[0].bar(i + (j - 2) * 0.16, v[j], 0.15, color=cols[n], label=lab(n) if i == 0 else None)
    axs[0].set_xticks([0, 1]); axs[0].set_xticklabels(["1 h ahead", "Day ahead"])
    axs[0].set_ylabel("nRMSE (% of rated power)"); axs[0].grid(axis="x", visible=False)
    title(axs[0], "SIMULATED: forecast error, test 2026", "(a)")
    axs[0].set_ylim(0, 27); axs[0].legend(loc="upper right", fontsize=5.6, ncol=1)
    P = pd.read_csv(PROC / "solar_predictions_h24.csv", index_col=0, parse_dates=True)
    P.index = pd.to_datetime(P.index, utc=True).tz_convert(TZ) + pd.Timedelta(hours=24)
    wk = P.loc["2026-07-14":"2026-07-20"]
    axs[1].plot(wk.index, wk.y, color=INK, lw=1.6, label="Simulated power (truth)")
    for n in names[2:]:
        axs[1].plot(wk.index, wk[n], color=cols[n], lw=1.1, label=lab(n))
    axs[1].set_ylabel("Power (W)"); axs[1].set_ylim(0, 44); axs[1].legend(loc="upper center", fontsize=5.6, ncol=2)
    title(axs[1], "SIMULATED: day-ahead forecasts, monsoon week", "(b)")
    axs[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b", tz=TZ))
    save(fig, "fig8_pv_forecast_sim")
