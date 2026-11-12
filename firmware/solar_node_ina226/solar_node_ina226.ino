/*
 * ============================================================================
 *  solar_node_ina226.ino  -  PV monitoring node v2 (ESP32)  [proposed / Phase 2]
 * ============================================================================
 *  Adds what the trial node could not measure:
 *    INA226   panel voltage, CURRENT and POWER  ->  P and E become measurable
 *    DS18B20  panel back-sheet (cell) temperature
 *    DHT22    ambient T/RH (replaces DHT11; mount in a ventilated radiation
 *             shield - the trial DHT11 read +5 degC high in full sun)
 *    divider  battery voltage on GPIO34
 *  Every sample is NTP-timestamped. Energy is integrated on-device (Wh).
 *
 *  Libraries: INA226 (Rob Tillaart), OneWire, DallasTemperature, DHT sensor library
 *
 *  Wiring
 *    INA226  SDA->21 SCL->22 VCC->3V3 GND->GND, IN+ from panel +, IN- to charge controller PV+
 *            (high-side, panel side: measures GENERATED power)
 *    DS18B20 data->GPIO4 (4.7k pull-up to 3V3), probe taped behind a centre cell
 *    DHT22   data->GPIO15
 *    Battery + -> 100k -> GPIO34 -> 22k -> GND   (max 15 V -> 2.7 V)
 *
 *  SHUNT: stock modules ship with 0.1 ohm = +-0.82 A max - too small for most panels.
 *         Size it for ~1.5 x Isc: R = 0.08192 V / I_max.
 *         50 W, 12 V panel (Isc ~3 A) -> I_max 4.5 A -> R_SHUNT 0.018 ohm (use 0.02 ohm, 1 W).
 *
 *  ThingSpeak fields (new solar channel):
 *    1 V_panel  2 I_panel  3 P_panel  4 E_today_Wh  5 T_panel  6 T_amb  7 RH_amb  8 V_batt
 * ============================================================================
 */
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <time.h>
#include <INA226.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <DHT.h>
#include "secrets.h"            // WIFI_SSID, WIFI_PASS, TS_SOLAR_WRITE_KEY

const float R_SHUNT = 0.020f;   // ohm - CHANGE to the fitted shunt
const float I_MAX   = 4.0f;     // A   - full-scale current for calibration
const float VDIV    = (100.0f + 22.0f) / 22.0f;
const unsigned long SAMPLE_MS = 1000, LOG_MS = 20000;

INA226 ina(0x40);
OneWire ow(4); DallasTemperature ds(&ow);
DHT dht(15, DHT22);

double energyWh = 0; int energyDay = -1;
float vSum = 0, iSum = 0, pSum = 0; int n = 0;
unsigned long lastSample = 0, lastLog = 0;

float readBattery() {
  uint32_t mv = 0; for (int i = 0; i < 16; i++) mv += analogReadMilliVolts(34);
  return (mv / 16.0f) / 1000.0f * VDIV;
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  if (!ina.begin()) Serial.println("[ERR] INA226 not found - check I2C wiring");
  ina.setMaxCurrentShunt(I_MAX, R_SHUNT);
  ina.setAverage(INA226_64_SAMPLES);            // ~ 70 ms conversion, low noise
  ds.begin(); dht.begin();
  analogSetPinAttenuation(34, ADC_11db);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  configTzTime("IST-5:30", "in.pool.ntp.org", "pool.ntp.org", "time.google.com");
}

void loop() {
  if (millis() - lastSample >= SAMPLE_MS) {
    float dt_h = (millis() - lastSample) / 3600000.0f; lastSample = millis();
    float v = ina.getBusVoltage(), i = ina.getCurrent(), p = ina.getPower();
    if (i < 0.002f) { i = 0; p = 0; }           // suppress shunt offset at night
    time_t now = time(nullptr); struct tm t; localtime_r(&now, &t);
    if (now > 1700000000 && t.tm_yday != energyDay) { energyDay = t.tm_yday; energyWh = 0; }  // reset at local midnight
    energyWh += p * dt_h;
    vSum += v; iSum += i; pSum += p; n++;
  }
  if (millis() - lastLog >= LOG_MS && n > 0) {
    lastLog = millis();
    ds.requestTemperatures();
    float tp = ds.getTempCByIndex(0), ta = dht.readTemperature(), rh = dht.readHumidity(), vb = readBattery();
    String url = String("https://api.thingspeak.com/update?api_key=") + TS_SOLAR_WRITE_KEY +
      "&field1=" + String(vSum / n, 3) + "&field2=" + String(iSum / n, 4) + "&field3=" + String(pSum / n, 3) +
      "&field4=" + String(energyWh, 3) + "&field5=" + String(tp, 2) + "&field6=" + String(ta, 1) +
      "&field7=" + String(rh, 1) + "&field8=" + String(vb, 3);
    Serial.printf("V %.2f  I %.3f  P %.2f  E %.2f Wh  Tp %.1f  Ta %.1f  RH %.0f  Vb %.2f\n",
                  vSum / n, iSum / n, pSum / n, energyWh, tp, ta, rh, vb);
    vSum = iSum = pSum = 0; n = 0;
    if (WiFi.status() == WL_CONNECTED) { HTTPClient h; h.begin(url); h.setTimeout(8000); h.GET(); h.end(); }
    else WiFi.reconnect();
  }
}
