# Setup: rebuilding the system from scratch

## 1. Hardware

### Light node
| Part | Notes |
|---|---|
| Arduino Uno | relay driver + light-sensor logger |
| ESP32 DevKit | controller: WiFi, NTP, scheduling |
| LDR + 10 kΩ resistor | analog divider on A0 (reading rises as it gets darker) |
| 1-channel relay module | active-high, input on D7 |
| LED + 220 Ω | status on D13 |
| 10 kΩ + 20 kΩ resistors | divider on Arduino TX (5 V) → ESP32 RX (3.3 V) |
| 5 V supply, enclosure, jumper wires | |

Wiring
```
Arduino A0  <- LDR divider          Arduino D7  -> relay IN
Arduino D13 -> LED                  Arduino D10 <- ESP32 GPIO17 (TX2)
Arduino D11 -> divider -> ESP32 GPIO16 (RX2)      GND <-> GND
```

### Solar node
| Part | Notes |
|---|---|
| ESP32 DevKit | |
| INA226 module | **replace the 0.1 Ω shunt**: R = 0.08192 / (1.5 × Isc). About 0.02 Ω, 1 W for a 50 W panel |
| DS18B20 waterproof probe + 4.7 kΩ | taped behind a centre cell |
| DHT22 | inside a ventilated radiation shield, not a closed box |
| 100 kΩ + 22 kΩ resistors | battery voltage divider |

Wiring
```
INA226 SDA -> GPIO21, SCL -> GPIO22, VCC -> 3V3
INA226 IN+ <- panel +,  IN- -> charge controller PV+   (measures generated power)
DS18B20 data -> GPIO4 (4.7 kΩ pull-up to 3V3)          DHT22 data -> GPIO15
Battery + -> 100 kΩ -> GPIO34 -> 22 kΩ -> GND
```

## 2. Software
- **Arduino IDE 2.x** with the **ESP32 board package** (Espressif).
- Arduino libraries (Library Manager): **INA226** by Rob Tillaart, **OneWire**, **DallasTemperature**, **DHT sensor library** (Adafruit) + **Adafruit Unified Sensor**.
- **Python 3.10+**, then `pip install -r requirements.txt`.

## 3. Accounts and keys
| Service | What to create |
|---|---|
| ThingSpeak | **Light data channel** (8 fields), **light control channel** (5 fields: date, dusk offset, dawn offset, model version, override), **solar channel** (8 fields) |
| Open-Meteo | nothing, no key needed (non-commercial use) |

Copy `secrets.example.h` to `secrets.h` in each firmware folder and fill in WiFi and ThingSpeak keys. `secrets.h` is never committed.

## 4. Clone and run
```bash
git clone https://github.com/Shadowprofessor/predictive-control.git
cd predictive-control
pip install -r requirements.txt
```
1. Put your ThingSpeak CSV exports in `data/raw/` as `light_feeds.csv` and `solar_feeds.csv`.
2. `python src/fetch_weather.py` downloads the weather history into `data/weather/`.
3. `./run_all.sh` rebuilds the dataset, trains the models, writes `server/models/` and regenerates `firmware/light_node_hybrid/esp32_controller/seasonal_model.h`.
4. Flash `arduino_actuator` to the Uno and `esp32_controller` to the ESP32. Flash `solar_node_ina226` to the solar ESP32.
5. Run the daily predictor before dusk:
   ```bash
   export TS_CTRL_WRITE_KEY=...
   python server/daily_predictor.py --dry-run
   # cron, IST:  30 14 * * *  cd /path/predictive-control && python server/daily_predictor.py
   ```
6. Retrain once a month (`./run_all.sh`) and re-flash the ESP32 with the new `seasonal_model.h`.

## 5. Checks after flashing
- ESP32 serial (115200): `[BOOT]` line, then `[PLAN]` once the server has posted a plan.
- Light data channel: field 4 = mode (0 server, 1 on-chip, 2/3 manual, 9 fallback), field 5 = raw LDR, field 8 = tier.
- Solar channel: current stays at 0 A at night; power peaks near midday.
