#!/usr/bin/env python3
"""
Solar Step 2 - SIMULATED PV power and energy.

  *** SIMULATION - NOT MEASURED DATA ***
  Power is computed with pvlib from Open-Meteo reanalysis weather for the NIIT
  site (2025-01-01 .. 2026-11-11). It exists to develop and benchmark the
  forecasting pipeline before INA226 current logging is deployed. Every figure
  and table derived from it must be labelled "simulated".

Assumed system (CONFIRM against the real panel's datasheet):
  36-cell mono-Si module, 50 Wp STC, gamma_pdc = -0.40 %/degC,
  fixed tilt 28 deg (~latitude), azimuth 180 deg (south), Faiman cell-temperature model,
  PVWatts DC model, 14 % lumped system losses.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, pvlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
warnings.filterwarnings("ignore")

SYS = dict(pdc0=50.0, gamma_pdc=-0.0040, tilt=28.0, azimuth=180.0, losses=0.14)

wx = load_weather()
site = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)
sp = site.get_solarposition(wx.index + pd.Timedelta(minutes=30))   # mid-hour sun position
sp.index = wx.index
cs = site.get_clearsky(wx.index + pd.Timedelta(minutes=30), model="ineichen"); cs.index = wx.index

def pv_power(ghi, dni, dhi, temp_air, wind):
    dni_extra = pvlib.irradiance.get_extra_radiation(wx.index)
    poa = pvlib.irradiance.get_total_irradiance(
        SYS["tilt"], SYS["azimuth"], sp["apparent_zenith"], sp["azimuth"],
        dni=dni, ghi=ghi, dhi=dhi, dni_extra=dni_extra, model="haydavies")["poa_global"].fillna(0)
    t_cell = pvlib.temperature.faiman(poa, temp_air, wind)
    p = pvlib.pvsystem.pvwatts_dc(poa, t_cell, SYS["pdc0"], SYS["gamma_pdc"]) * (1 - SYS["losses"])
    return p.clip(lower=0), poa, t_cell

wind = wx["wind_speed_10m"] / 3.6                                          # km/h -> m/s
p, poa, tcell = pv_power(wx["shortwave_radiation"], wx["direct_normal_irradiance"],
                         wx["diffuse_radiation"], wx["temperature_2m"], wind)
p_cs, poa_cs, _ = pv_power(cs["ghi"], cs["dni"], cs["dhi"], wx["temperature_2m"], wind)

out = pd.DataFrame({
    "p_sim_W": p, "poa_W_m2": poa, "t_cell_C": tcell, "p_clearsky_W": p_cs,
    "ghi": wx["shortwave_radiation"], "ghi_clearsky": cs["ghi"], "dni": wx["direct_normal_irradiance"],
    "dhi": wx["diffuse_radiation"], "temp_air": wx["temperature_2m"], "rh": wx["relative_humidity_2m"],
    "cloud": wx["cloud_cover"], "cloud_low": wx["cloud_cover_low"], "cloud_mid": wx["cloud_cover_mid"],
    "cloud_high": wx["cloud_cover_high"], "precip": wx["precipitation"], "wind": wind,
    "pm2_5": wx["pm2_5"], "aod": wx["aerosol_optical_depth"], "dust": wx["dust"],
    "sun_elev": sp["apparent_elevation"],
})
out["kt_pv"] = (out.p_sim_W / out.p_clearsky_W.replace(0, np.nan)).where(out.p_clearsky_W > 2).clip(0, 1.3)
out.to_csv(PROC / "solar_simulated_hourly.csv")

daily = out["p_sim_W"].resample("D").sum() / 1000                         # kWh/day (hourly W -> Wh)
daily.to_csv(PROC / "solar_simulated_daily_kWh.csv", header=["E_sim_kWh"])
mon = daily.groupby(daily.index.to_period("M")).agg(["mean", "sum"])
mon.columns = ["mean_kWh_per_day", "total_kWh"]
mon.to_csv(TAB / "solar_sim_monthly_energy.csv")
print("SIMULATED - assumed 50 Wp module, tilt 28, south")
print(f"hours {len(out):,}   peak {out.p_sim_W.max():.1f} W   annual 2025 energy "
      f"{daily['2025'].sum():.1f} kWh   specific yield {daily['2025'].sum()/SYS['pdc0']*1000:.0f} kWh/kWp")
print(mon.round(3).to_string())
