#!/usr/bin/env python3
"""
Download NSE F&O bhavcopy files for a date range into ./bhav.
Each daily zip is unpacked to fo_YYYYMMDD.csv. Already-downloaded days are skipped,
so the script can be re-run safely.
"""
import io
import os
import time
import zipfile
from datetime import date, timedelta

import requests

OUT = "./bhav"
START = date(2026, 5, 18)   # about six months back
END = date(2026, 11, 17)    # yesterday; today's file may not be published yet
PAUSE_SEC = 1.5             # delay between requests, to avoid being blocked

URL = ("https://nsearchives.nseindia.com/content/fo/"
       "BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def main():
    os.makedirs(OUT, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://www.nseindia.com", timeout=15)  # sets the cookies NSE expects

    saved = skipped = missing = failed = 0
    d = START
    while d <= END:
        if d.weekday() < 5:  # Monday to Friday only
            ymd = d.strftime("%Y%m%d")
            dest = os.path.join(OUT, f"fo_{ymd}.csv")
            if os.path.exists(dest):
                skipped += 1
            else:
                try:
                    r = session.get(URL.format(ymd=ymd), timeout=30)
                except requests.RequestException as e:
                    print(f"network error {ymd}: {e}")
                    failed += 1
                    d += timedelta(days=1)
                    continue

                if r.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        with open(dest, "wb") as f:
                            f.write(z.read(z.namelist()[0]))
                    print("saved", dest)
                    saved += 1
                    time.sleep(PAUSE_SEC)
                elif r.status_code == 404:
                    print("no file (holiday):", ymd)
                    missing += 1
                    time.sleep(PAUSE_SEC)
                else:
                    print(f"unexpected status {r.status_code} for {ymd}")
                    failed += 1
                    time.sleep(PAUSE_SEC)
        d += timedelta(days=1)

    print(f"\nDone. saved={saved} already_had={skipped} holidays={missing} failed={failed}")


if __name__ == "__main__":
    main()
    