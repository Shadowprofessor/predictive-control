#!/usr/bin/env python3
"""
Solar Step 1 - MEASURED data (3 Feb - 29 Apr 2026, panel 1, no battery attached).

Panel terminal voltage with no battery connected behaves as open-circuit voltage
(Voc). Physics: Voc rises with ln(irradiance) and falls with cell temperature.
We test whether the measured data show both effects, quantify them, and
characterise the DHT11 enclosure-heating bias against reanalysis air temperature.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, pvlib
from sklearn.linear_model import LinearRegression
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
warnings.filterwarnings("ignore")
NOCT = 45.0   # nominal operating cell temperature, typical c-Si module

d = pd.read_csv(RAW / "solar_feeds.csv", low_memory=False)
d["t"] = pd.to_datetime(d["created_at"], format="ISO8601")
for f in ["field1", "field2", "field3"]:
    d[f] = pd.to_numeric(d[f], errors="coerce")
d = d[(d.t >= "2026-02-01") & (d.t < "2026-05-02")]
funnel = {"rows Feb-Apr": len(d)}
d = d[d.field1.between(-5, 55) & d.field2.between(0, 100)];  funnel["plausible T/RH"] = len(d)
d = d.rename(columns={"field1": "t_dht", "field2": "rh_dht", "field3": "v_panel"}).set_index("t")
q = d[["t_dht", "rh_dht", "v_panel"]].resample("15min").mean().dropna()
funnel["15-min bins"] = len(q)

wx = load_weather()[["temperature_2m", "shortwave_radiation", "direct_normal_irradiance",
                     "diffuse_radiation", "cloud_cover", "relative_humidity_2m"]]
wx = wx.resample("15min").interpolate("linear")
q = q.join(wx, how="inner")
site = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)
sp = site.get_solarposition(q.index)
cs = site.get_clearsky(q.index, model="ineichen")
q["sun_elev"] = sp["apparent_elevation"].values
q["ghi_cs"] = cs["ghi"].values
q["kt"] = (q.shortwave_radiation / q.ghi_cs.replace(0, np.nan)).clip(0, 1.2)
q["t_cell"] = q.temperature_2m + q.shortwave_radiation / 800 * (NOCT - 20)
q.to_csv(PROC / "solar_measured_15min.csv")

day = q[(q.sun_elev > 10) & (q.shortwave_radiation > 50) & (q.v_panel > 1)].copy()
funnel["daytime bins (elev>10, GHI>50)"] = len(day)
night = q[q.sun_elev < -6]
out = {}
out["night: share of bins at 0 V"] = (night.v_panel < 0.5).mean()
out["noon median V"] = day[(day.index.hour >= 11) & (day.index.hour <= 14)].v_panel.median()

# physics regression  V = a + b ln(GHI) + c T_cell
X = np.column_stack([np.log(day.shortwave_radiation), day.t_cell])
lr = LinearRegression().fit(X, day.v_panel)
out["fit: b  dV/dln(GHI)  [V]"] = lr.coef_[0]
out["fit: c  dV/dT_cell  [V/degC]"] = lr.coef_[1]
out["fit: c as %/degC of median V"] = 100 * lr.coef_[1] / day.v_panel.median()
out["fit: R2"] = lr.score(X, day.v_panel)
# expected for a 36-cell module: ~ -2.2 mV/degC/cell, n*Vt ~ 1.2*25.7 mV per cell
out["expected c for 36-cell module [V/degC]"] = -0.0022 * 36
out["expected b for 36-cell module [V]"] = 36 * 1.2 * 0.0257

# temperature coefficient on the high-irradiance subset, where ln(G) barely varies
# (removes the collinearity between irradiance and cell temperature)
hg = day[(day.shortwave_radiation > 700) & (day.v_panel > 18.5)]
lr_hg = LinearRegression().fit(hg[["t_cell"]], hg.v_panel)
out["high-GHI (>700) subset n"] = len(hg)
out["high-GHI fit: dV/dT_cell [V/degC]"] = lr_hg.coef_[0]
out["high-GHI fit: %/degC of median V"] = 100 * lr_hg.coef_[0] / hg.v_panel.median()
out["high-GHI fit: R2"] = lr_hg.score(hg[["t_cell"]], hg.v_panel)
out["daytime bins in 15-18.5 V low cluster"] = int(day.v_panel.between(15, 18.5).sum())

# cloud effect: clear (kt>0.7) vs cloudy (kt<0.4) at similar sun elevation
mid = day[day.sun_elev.between(30, 60)]
out["V clear-sky bins (kt>0.7)"] = mid[mid.kt > 0.7].v_panel.mean()
out["V cloudy bins (kt<0.4)"] = mid[mid.kt < 0.4].v_panel.mean()
out["n clear / n cloudy"] = f"{(mid.kt>0.7).sum()} / {(mid.kt<0.4).sum()}"

# DHT11 enclosure heating: sensor minus reanalysis air temp, vs irradiance
q["dT_dht"] = q.t_dht - q.temperature_2m
bias = q.groupby(pd.cut(q.shortwave_radiation, [-1, 10, 200, 400, 600, 800, 1200]))["dT_dht"].agg(["mean", "std", "count"])
bias.to_csv(TAB / "solar_dht_heating_bias.csv")
out["DHT - air temp at night [degC]"] = q[q.sun_elev < -6].dT_dht.mean()
out["DHT - air temp at GHI>600 [degC]"] = q[q.shortwave_radiation > 600].dT_dht.mean()

corr = day[["v_panel", "shortwave_radiation", "kt", "cloud_cover", "sun_elev", "t_cell",
            "temperature_2m", "t_dht", "rh_dht"]].corr()["v_panel"].drop("v_panel")
corr.to_csv(TAB / "solar_measured_correlations.csv", header=["pearson_r_with_V"])
pd.Series({**funnel, **out}).to_csv(TAB / "solar_measured_summary.csv", header=["value"])

for k, v in {**funnel, **out}.items():
    print(f"  {k:42s} {v:.3f}" if isinstance(v, float) else f"  {k:42s} {v}")
print("\ncorrelation with panel V (daytime):"); print(corr.round(3).to_string())
print("\nDHT11 minus reanalysis air temperature by irradiance band:"); print(bias.round(2).to_string())
