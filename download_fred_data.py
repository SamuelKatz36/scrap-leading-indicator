"""
Step 1: Run this script FIRST to download all FRED data as CSVs.
It bypasses the SSL certificate issue by disabling verification.

Usage:
  python download_fred_data.py

After this runs, you'll have a folder called 'fred_data/' with CSV files.
Then run: python scrap_marketplace_analysis_v2.py
"""

import urllib.request
import ssl
import os

# Bypass SSL verification (needed for older Python / broken certs)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

OUTPUT_DIR = './fred_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All series we need
SERIES = {
    'WPU1012':      'Steel Scrap PPI',
    'WPU10230101':  'Copper Scrap PPI',
    'WPU10250201':  'Aluminum Scrap PPI',
    'WPU10220131':  'Copper Cathode PPI',
    'WPU1025':      'Aluminum Mill Shapes PPI',
    'WPU102504':    'Nickel Mill Shapes PPI',
    'WPU101':       'Primary Iron & Steel PPI',
    'PCU33133313':  'Alumina & Al Processing PPI',
    'WPUSI019011':  'Copper Products PPI',
    'PCOPPUSDM':    'Global Copper Price',
    'PNICKUSDM':    'Global Nickel Price',
    'PIORECRUSDM':  'Global Iron Ore Price',
    'PALUMUSDM':    'Global Aluminum Price',
    'PZINCUSDM':    'Global Zinc Price',
    'PLEADUSDM':    'Global Lead Price',
    'PCOBAUSDM':    'Global Cobalt Price',
    'USCONS':       'Construction Employment',
    'IPMAN':        'Manufacturing IP',
    'PERMIT':       'Building Permits',
    'MANEMP':       'Manufacturing Employment',
}

API_KEY = os.environ.get('FRED_API_KEY', '13b3dd64e18e2d4f219a4c74b7ad832c')

print("Downloading FRED data...\n")

success = 0
for sid, name in SERIES.items():
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={sid}"
           f"&api_key={API_KEY}"
           f"&file_type=json"
           f"&observation_start=2015-01-01"
           f"&observation_end=2025-12-01")

    filepath = os.path.join(OUTPUT_DIR, f"{sid}.csv")
    print(f"  {sid:16s} ({name})...", end=" ")

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ctx) as resp:
            import json
            data = json.loads(resp.read().decode())
            obs = data.get('observations', [])

            with open(filepath, 'w') as f:
                f.write("DATE,VALUE\n")
                for o in obs:
                    val = o['value']
                    if val != '.':  # FRED uses '.' for missing
                        f.write(f"{o['date']},{val}\n")

            print(f"OK ({len(obs)} observations)")
            success += 1
    except Exception as e:
        print(f"FAILED: {e}")

print(f"\nDownloaded {success} of {len(SERIES)} series to {OUTPUT_DIR}/")
print("Now run: python scrap_marketplace_analysis_v2.py")
