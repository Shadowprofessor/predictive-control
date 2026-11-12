#!/usr/bin/env bash
# Reproduce every table and figure from the raw data in data/.
set -euo pipefail
cd "$(dirname "$0")"
python src/light/01_build_events.py
python src/light/02_models.py
python src/light/03_risk_pareto.py
python src/light/04_hourly_classifier.py
python src/light/05_train_export.py
python src/solar/01_measured_analysis.py
python src/solar/02_simulate_pv.py
python src/solar/03_pv_models.py
python src/figures.py
echo "done - see results/tables and results/figures"
