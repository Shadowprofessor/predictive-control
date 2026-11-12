// Copy to secrets.h and fill in. secrets.h is git-ignored - NEVER commit real keys.
#pragma once
#define WIFI_SSID        "your-ssid"
#define WIFI_PASS        "your-password"

// Data channel (the existing light channel): the node WRITES its log here
#define TS_DATA_WRITE_KEY  "XXXXXXXXXXXXXXXX"

// Control channel (NEW channel): the server WRITES tonight's plan, the node READS it
//   field1 = plan date YYYYMMDD    field2 = dusk offset (min, margin included)
//   field3 = dawn offset (min)     field4 = model version   field5 = override 0/2/3
#define TS_CTRL_CHANNEL_ID "0000000"
#define TS_CTRL_READ_KEY   "XXXXXXXXXXXXXXXX"
