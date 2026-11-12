/*
 * ============================================================================
 *  arduino_actuator.ino  -  Light node, ACTUATOR + REFEREE SENSOR (Arduino Uno)
 * ============================================================================
 *  Role in the hybrid predictive controller:
 *    - Drives the relay + LED ONLY on command from the ESP32 controller.
 *    - Reads the LDR every second and REPORTS it (raw 0-1023 + binary dark flag).
 *      The LDR is a ground-truth referee: it is never used for control while the
 *      ESP32 is alive.
 *    - Tier-3 emergency fallback: if no command arrives from the ESP32 for
 *      LINK_TIMEOUT_MS, the Arduino falls back to classic LDR switching so the
 *      road is never left dark. This is logged as mode 9 and excluded from the
 *      sensorless evaluation.
 *
 *  Wiring (unchanged from the light-event repo):
 *    LDR divider   -> A0   (reading RISES as it gets darker)
 *    Relay IN      -> D7   (ACTIVE HIGH)
 *    LED           -> D13
 *    ESP32 TX(17)  -> D10  (SoftwareSerial RX)
 *    ESP32 RX(16)  <- D11  (SoftwareSerial TX, via 10k/20k divider to 3.3 V)
 *
 *  Serial protocol (9600 baud, '\n' terminated)
 *    ESP32 -> Arduino :  R1  relay ON      R0  relay OFF      P  ping
 *    Arduino -> ESP32 :  L,<raw>,<dark>,<relay>,<fallback>   every 2 s
 * ============================================================================
 */
#include <SoftwareSerial.h>

#define LDR_PIN   A0
#define RELAY_PIN 7
#define LED_PIN   13
const bool RELAY_ACTIVE_LOW = false;

const int THRESHOLD = 620;            // >= 620 -> dark (same as the original firmware)
const int HYSTERESIS = 4;             // <= 615 -> bright
const unsigned long REPORT_MS = 2000;
const unsigned long LINK_TIMEOUT_MS = 10UL * 60UL * 1000UL;   // 10 min without ESP32 -> fallback

SoftwareSerial esp(10, 11);           // RX, TX

bool relayOn = false, ldrDark = false, fallback = false;
int raw = 0;
unsigned long lastCmd = 0, lastReport = 0;
String buf;

void setRelay(bool on) {
  if (on == relayOn) return;          // event-actuated: write only on change
  relayOn = on;
  digitalWrite(RELAY_PIN, (on ^ RELAY_ACTIVE_LOW) ? HIGH : LOW);
  digitalWrite(LED_PIN, on ? HIGH : LOW);
}

void sampleLdr() {
  raw = analogRead(LDR_PIN);
  if (raw >= THRESHOLD) ldrDark = true;
  else if (raw <= THRESHOLD - HYSTERESIS - 1) ldrDark = false;   // hold inside the band
}

void handle(const String &c) {
  lastCmd = millis();
  if (fallback) { fallback = false; Serial.println(F("[ACT] link restored, leaving fallback")); }
  if (c == "R1") setRelay(true);
  else if (c == "R0") setRelay(false);
}

void setup() {
  digitalWrite(RELAY_PIN, RELAY_ACTIVE_LOW ? HIGH : LOW);   // safe OFF before pinMode
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(9600);
  esp.begin(9600);
  sampleLdr();
  lastCmd = millis();
  Serial.println(F("[ACT] actuator up - waiting for ESP32 commands"));
}

void loop() {
  while (esp.available()) {
    char ch = esp.read();
    if (ch == '\n' || ch == '\r') { if (buf.length()) { handle(buf); buf = ""; } }
    else if (buf.length() < 16) buf += ch;
  }
  static unsigned long lastSample = 0;
  if (millis() - lastSample >= 1000) { lastSample = millis(); sampleLdr(); }

  if (!fallback && millis() - lastCmd > LINK_TIMEOUT_MS) {
    fallback = true;
    Serial.println(F("[ACT] ESP32 silent - TIER 3 LDR fallback"));
  }
  if (fallback) setRelay(ldrDark);

  if (millis() - lastReport >= REPORT_MS) {
    lastReport = millis();
    esp.print(F("L,")); esp.print(raw); esp.print(','); esp.print(ldrDark ? 1 : 0);
    esp.print(','); esp.print(relayOn ? 1 : 0); esp.print(','); esp.println(fallback ? 1 : 0);
  }
}
