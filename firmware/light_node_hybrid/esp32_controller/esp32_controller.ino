/*
 * ============================================================================
 *  esp32_controller.ino  -  Light node, PREDICTIVE CONTROLLER (ESP32)
 * ============================================================================
 *  Sensorless, NTP-disciplined, three-tier hybrid control.
 *
 *    switch_on  (dusk) = sunset(today)  + dusk_offset
 *    switch_off (dawn) = sunrise(today) + dawn_offset
 *
 *  Tier 1  SERVER     offsets from the weather-conditioned Random Forest, pushed
 *                     once a day to a ThingSpeak control channel (server/).
 *  Tier 2  ON-CHIP    no valid server plan -> seasonal clock from seasonal_model.h
 *                     (day-of-year harmonics + clamp + safety margin). Needs only NTP
 *                     time from the last sync; the ESP32 RTC keeps running offline.
 *  Tier 3  FALLBACK   ESP32 dead / never synced -> the Arduino switches on its LDR
 *                     after 10 min of silence (logged, excluded from evaluation).
 *
 *  Sunrise/sunset are computed ON-CHIP with the NOAA approximation (~1 min).
 *  The LDR is read and LOGGED only - it is never an input to the decision.
 *
 *  Wiring: ESP32 GPIO16 (RX2) <- Arduino D11 (via divider), GPIO17 (TX2) -> Arduino D10
 *  Copy secrets.example.h -> secrets.h before building.
 * ============================================================================
 */
#include <WiFi.h>
#include <HTTPClient.h>
#include <time.h>
#include <math.h>
#include "seasonal_model.h"
#include "secrets.h"

HardwareSerial link2(2);
const char *TZ_IST = "IST-5:30";
const unsigned long LOG_MS = 20000, PLAN_POLL_MS = 30UL * 60UL * 1000UL;

struct Plan { long date = 0; float dusk = NAN, dawn = NAN; int ver = 0; };
Plan cur, prev;
int  overrideMode = 0;                        // 0 auto, 2 manual ON, 3 manual OFF
int  ldrRaw = -1, ldrDark = -1, actRelay = -1, actFallback = 0;
bool relayCmd = false, timeOk = false;
int  tier = 2;
float todayDuskOn = NAN, todayDawnOff = NAN;  // minutes after local midnight
unsigned long lastLog = 0, lastPoll = 0, lastCmdTx = 0;
String rx;

// ---------------------------------------------------------------- astronomy
// NOAA general solar position (fractional-year) method. Returns minutes after
// LOCAL midnight. Zenith 90.833 deg = refraction + solar semidiameter.
float sunEventMin(const struct tm &t, bool rise) {
  const float D2R = M_PI / 180.0f;
  float g = 2.0f * M_PI / 365.0f * (t.tm_yday);                    // tm_yday is 0-based
  float eqt = 229.18f * (0.000075f + 0.001868f * cosf(g) - 0.032077f * sinf(g)
                       - 0.014615f * cosf(2 * g) - 0.040849f * sinf(2 * g));
  float dec = 0.006918f - 0.399912f * cosf(g) + 0.070257f * sinf(g) - 0.006758f * cosf(2 * g)
            + 0.000907f * sinf(2 * g) - 0.002697f * cosf(3 * g) + 0.00148f * sinf(3 * g);
  float lat = SITE_LAT * D2R;
  float cha = cosf(90.833f * D2R) / (cosf(lat) * cosf(dec)) - tanf(lat) * tanf(dec);
  cha = fmaxf(-1.0f, fminf(1.0f, cha));
  float ha = acosf(cha) / D2R;
  float utc = 720.0f - 4.0f * (SITE_LON + (rise ? ha : -ha)) - eqt;
  return utc + SITE_TZ_H * 60.0f;
}

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

float seasonalOffset(const struct tm &t, bool dusk) {
  float d = t.tm_yday + 1;
  float s1 = sinf(2 * M_PI * d / 365), c1 = cosf(2 * M_PI * d / 365);
  float s2 = sinf(4 * M_PI * d / 365), c2 = cosf(4 * M_PI * d / 365);
  const float *B = dusk ? DUSK_B : DAWN_B;
  float o = B[0] + B[1] * s1 + B[2] * c1 + B[3] * s2 + B[4] * c2;
  if (dusk) return clampf(o, DUSK_CLAMP_LO, DUSK_CLAMP_HI) - DUSK_SAFETY_MARGIN_MIN;  // earlier = safer
  return clampf(o, DAWN_CLAMP_LO, DAWN_CLAMP_HI) + DAWN_SAFETY_MARGIN_MIN;            // later  = safer
}

long ymd(const struct tm &t) { return (t.tm_year + 1900) * 10000L + (t.tm_mon + 1) * 100L + t.tm_mday; }

long ymdOffset(const struct tm &t, int days) {
  time_t x = mktime((struct tm *)&t) + (time_t)days * 86400;
  struct tm o; localtime_r(&x, &o); return ymd(o);
}

// ------------------------------------------------------------------ network
String field(const String &js, const char *name) {
  String k = String("\"") + name + "\":";
  int i = js.indexOf(k); if (i < 0) return "";
  i += k.length(); if (js.charAt(i) == '"') i++;
  int j = i; while (j < (int)js.length() && js.charAt(j) != '"' && js.charAt(j) != ',' && js.charAt(j) != '}') j++;
  return js.substring(i, j);
}

void pollPlan() {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  String url = String("https://api.thingspeak.com/channels/") + TS_CTRL_CHANNEL_ID +
               "/feeds/last.json?api_key=" + TS_CTRL_READ_KEY;
  http.begin(url); http.setTimeout(8000);
  if (http.GET() == 200) {
    String js = http.getString();
    long d = field(js, "field1").toInt();
    if (d > 20250000 && d != cur.date) {
      prev = cur;
      cur.date = d; cur.dusk = field(js, "field2").toFloat(); cur.dawn = field(js, "field3").toFloat();
      cur.ver = field(js, "field4").toInt();
      Serial.printf("[PLAN] new plan %ld  dusk %+0.1f  dawn %+0.1f  v%d\n", cur.date, cur.dusk, cur.dawn, cur.ver);
    }
    String ov = field(js, "field5"); overrideMode = ov.length() ? ov.toInt() : 0;
  }
  http.end();
}

void logToCloud() {
  if (WiFi.status() != WL_CONNECTED) return;
  int mode = overrideMode ? overrideMode : (actFallback ? 9 : (tier == 1 ? 0 : 1));
  HTTPClient http;
  String url = String("https://api.thingspeak.com/update?api_key=") + TS_DATA_WRITE_KEY +
    "&field1=" + ldrDark + "&field2=" + (relayCmd ? 1 : 0) + "&field3=" + (relayCmd ? 1 : 0) +
    "&field4=" + mode + "&field5=" + ldrRaw + "&field6=" + String(todayDuskOn, 1) +
    "&field7=" + String(todayDawnOff, 1) + "&field8=" + (actFallback ? 3 : tier);
  http.begin(url); http.setTimeout(8000); http.GET(); http.end();
}

// ----------------------------------------------------------------- control
void decide() {
  time_t now = time(nullptr);
  timeOk = now > 1700000000;                         // NTP has synced at least once
  if (!timeOk) return;                               // say nothing -> Arduino tier 3 after 10 min
  struct tm t; localtime_r(&now, &t);
  float nowMin = t.tm_hour * 60 + t.tm_min + t.tm_sec / 60.0f;
  long today = ymd(t), yesterday = ymdOffset(t, -1);

  // dusk of D uses plan D;  dawn of D uses plan D-1 (it predicted "tomorrow's dawn")
  float duskOff = NAN, dawnOff = NAN;
  if (cur.date == today) duskOff = cur.dusk;
  if (cur.date == yesterday) dawnOff = cur.dawn; else if (prev.date == yesterday) dawnOff = prev.dawn;
  bool serverDusk = !isnan(duskOff), serverDawn = !isnan(dawnOff);
  if (!serverDusk) duskOff = seasonalOffset(t, true);
  if (!serverDawn) dawnOff = seasonalOffset(t, false);
  bool evening = nowMin >= 720;
  tier = (evening ? serverDusk : serverDawn) ? 1 : 2;

  todayDuskOn  = sunEventMin(t, false) + duskOff;
  todayDawnOff = sunEventMin(t, true)  + dawnOff;
  bool want = (nowMin >= todayDuskOn) || (nowMin < todayDawnOff);
  if (overrideMode == 2) want = true;
  if (overrideMode == 3) want = false;

  // send on change, and re-assert every 60 s as the Arduino's heartbeat
  if (want != relayCmd || millis() - lastCmdTx > 60000) {
    if (want != relayCmd) Serial.printf("[CTRL] %02d:%02d  relay -> %s  (tier %d, dusk %.1f, dawn %.1f)\n",
                                        t.tm_hour, t.tm_min, want ? "ON" : "OFF", tier, todayDuskOn, todayDawnOff);
    relayCmd = want; link2.println(want ? "R1" : "R0"); lastCmdTx = millis();
  }
}

void readArduino() {
  while (link2.available()) {
    char c = link2.read();
    if (c == '\n') {
      if (rx.startsWith("L,")) {                     // L,<raw>,<dark>,<relay>,<fallback>
        int a = rx.indexOf(',', 2), b = rx.indexOf(',', a + 1), d = rx.indexOf(',', b + 1);
        ldrRaw = rx.substring(2, a).toInt(); ldrDark = rx.substring(a + 1, b).toInt();
        actRelay = rx.substring(b + 1, d).toInt(); actFallback = rx.substring(d + 1).toInt();
      }
      rx = "";
    } else if (c != '\r' && rx.length() < 40) rx += c;
  }
}

void setup() {
  Serial.begin(115200);
  link2.begin(9600, SERIAL_8N1, 16, 17);
  WiFi.mode(WIFI_STA); WiFi.begin(WIFI_SSID, WIFI_PASS);
  configTzTime(TZ_IST, "in.pool.ntp.org", "pool.ntp.org", "time.google.com");   // SNTP resyncs hourly
  Serial.println("[BOOT] predictive light controller - tiers: server / on-chip / LDR fallback");
}

void loop() {
  static unsigned long lastWifiTry = 0;
  if (WiFi.status() != WL_CONNECTED && millis() - lastWifiTry > 30000) { lastWifiTry = millis(); WiFi.reconnect(); }
  readArduino();
  static unsigned long lastDecide = 0;
  if (millis() - lastDecide >= 1000) { lastDecide = millis(); decide(); }
  if (millis() - lastPoll >= PLAN_POLL_MS || lastPoll == 0) { lastPoll = millis(); pollPlan(); }
  if (millis() - lastLog >= LOG_MS) { lastLog = millis(); logToCloud(); }
}
