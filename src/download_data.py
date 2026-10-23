"""Download all raw datasets for the smart-grid load forecasting study."""
import os, sys, time, zipfile, io, urllib.request, ssl, json

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(RAW, exist_ok=True)
ctx = ssl.create_default_context()

HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-download/1.0"}


def fetch(url, dest, timeout=300):
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f"  [skip] {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.1f} MB)")
        return True
    print(f"  [get ] {url}")
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers=HDRS)
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print(f"  [ok  ] {os.path.basename(dest)} {os.path.getsize(dest)/1e6:.1f} MB in {time.time()-t0:.0f}s")
        return True
    except Exception as e:
        print(f"  [FAIL] {url} -> {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"

    if what in ("all", "ett"):
        print("== ETT (Informer / AAAI-21 benchmark) ==")
        base = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
        for n in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
            fetch(base + n + ".csv", os.path.join(RAW, n + ".csv"))

    if what in ("all", "ecl"):
        print("== ECL / Electricity (UCI LD2011_2014) ==")
        dest = os.path.join(RAW, "LD2011_2014.txt")
        if not (os.path.exists(dest) and os.path.getsize(dest) > 1e6):
            zp = os.path.join(RAW, "ecl.zip")
            urls = [
                "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip",
            ]
            got = False
            for u in urls:
                if fetch(u, zp, timeout=900):
                    got = True
                    break
            if got:
                try:
                    with zipfile.ZipFile(zp) as z:
                        names = z.namelist()
                        print("   zip contents:", names)
                        inner = [n for n in names if n.lower().endswith((".txt", ".csv", ".zip"))][0]
                        if inner.lower().endswith(".zip"):
                            with z.open(inner) as fh:
                                inner_bytes = fh.read()
                            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as z2:
                                n2 = z2.namelist()[0]
                                with z2.open(n2) as fh2, open(dest, "wb") as out:
                                    out.write(fh2.read())
                        else:
                            with z.open(inner) as fh, open(dest, "wb") as out:
                                out.write(fh.read())
                    print(f"  [ok  ] LD2011_2014.txt {os.path.getsize(dest)/1e6:.1f} MB")
                except Exception as e:
                    print("  [FAIL] unzip:", e)

    if what in ("all", "opsd"):
        print("== OPSD European grid time series (60 min) ==")
        for u in [
            "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv",
            "https://data.open-power-system-data.org/index.php?package=time_series&version=latest&action=customDownload&resource=3&filter%5B_contentfilter_cet_cest_timestamp%5D%5Bfrom%5D=2005-01-01&filter%5B_contentfilter_cet_cest_timestamp%5D%5Bto%5D=2020-10-01&downloadCSV=Download+CSV",
        ]:
            if fetch(u, os.path.join(RAW, "opsd_60min.csv"), timeout=1800):
                break

    if what in ("all", "weather"):
        print("== ERA5 reanalysis temperature (Open-Meteo archive) ==")
        # one interior point per system, near its geographic centre (ES: the Madrid region)
        sites = {
            "DE": (51.10, 10.40),   # geographic centre of Germany
            "FR": (46.60, 2.40),    # centre of France
            "GB": (53.00, -1.60),   # centre of Great Britain
            # v2: five further national systems (same rule)
            "ES": (40.30, -3.70),   # central Spain (Madrid region)
            "IT": (43.50, 11.90),   # central Italy
            "NL": (52.20, 5.30),    # central Netherlands
            "PL": (52.00, 19.50),   # central Poland
            "BE": (50.80, 4.50),    # central Belgium
        }
        for cc, (la, lo) in sites.items():
            url = ("https://archive-api.open-meteo.com/v1/archive?"
                   f"latitude={la}&longitude={lo}&start_date=2014-12-01&end_date=2020-10-01"
                   "&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation,wind_speed_10m"
                   "&timezone=UTC&format=csv")
            fetch(url, os.path.join(RAW, f"era5_{cc}.csv"), timeout=600)

    print("\n== RAW DIR ==")
    for f in sorted(os.listdir(RAW)):
        p = os.path.join(RAW, f)
        print(f"  {f:40s} {os.path.getsize(p)/1e6:10.2f} MB")


if __name__ == "__main__":
    main()
