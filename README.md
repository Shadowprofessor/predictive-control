# predictive-control

Sensorless, NTP-timed street-light control and solar-panel monitoring on ESP32 + Arduino.

The light node switches on and off at times it predicts, not from a light sensor. It works out sunset and sunrise from the date and the site location, and a small model corrects that time for the season and the weather forecast. A light sensor stays on the node only to log what really happened.

A second node monitors a solar panel (voltage, current, power, energy, panel and air temperature).

## How it works
```
switch ON  = sunset today  + predicted dusk offset
switch OFF = sunrise today + predicted dawn offset
```
| Tier | Offset comes from | Used when |
|---|---|---|
| 1 Server | daily prediction from the weather forecast | a plan for today has arrived |
| 2 On-chip | built-in seasonal table on the ESP32 | no internet or no plan |
| 3 Fallback | Arduino switches on its light sensor | ESP32 silent for 10 min |

## Contents
```
firmware/light_node_hybrid/esp32_controller/   ESP32: NTP, sunset/sunrise, tiers, logging
firmware/light_node_hybrid/arduino_actuator/   Arduino: relay, sensor logging, fallback
firmware/solar_node_ina226/                    ESP32 solar node: INA226, DS18B20, DHT22
server/daily_predictor.py                      daily plan -> ThingSpeak control channel
src/                                           data processing and model training
```

Setup and cloning steps: **[SETUP.md](SETUP.md)**

## Authors
Sachit Sharma, Jetendra Joshi
