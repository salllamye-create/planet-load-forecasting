"""Population-weighted multi-point weather for the sensitivity analysis of the revision.

The corpus represents each system's weather by ERA5 at one interior point. National load
responds to population-weighted weather, so for three systems (FR, ES, IT: the ones where the
weather carries the most load) this script fetches ERA5 (Open-Meteo archive, the same source and
variables as download_data.py) at the six largest metropolitan areas, forms the population-
weighted mean of every weather variable, and writes a copy of the processed corpus with the four
weather channels replaced: data/proc_v2/OPSD-<CC>-mp.npz|json. Everything else (load, generation,
operator forecast, timestamps, holiday flags, splits) is byte-identical to the single-point file,
so a run on OPSD-<CC>-mp differs from the headline run in the weather representation only.

Weights are metropolitan-area populations (millions, Eurostat/INSEE/INE/ISTAT functional urban
areas, rounded); they only need to be of the right order.
"""
import json, os, sys, time, urllib.request, ssl
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")
PROC = os.path.join(HERE, "data", "proc_v2")
POINTS = {
    "FR": [("Paris", 48.86, 2.35, 12.3), ("Lyon", 45.76, 4.84, 2.3), ("Marseille", 43.30, 5.37, 1.8),
           ("Toulouse", 43.60, 1.44, 1.4), ("Lille", 50.63, 3.06, 1.2), ("Bordeaux", 44.84, -0.58, 1.3)],
    "ES": [("Madrid", 40.42, -3.70, 6.7), ("Barcelona", 41.39, 2.17, 5.6), ("Valencia", 39.47, -0.38, 1.8),
           ("Sevilla", 37.39, -5.98, 1.5), ("Bilbao", 43.26, -2.93, 1.0), ("Malaga", 36.72, -4.42, 1.0)],
    "IT": [("Milano", 45.46, 9.19, 4.3), ("Roma", 41.90, 12.50, 4.3), ("Napoli", 40.85, 14.27, 3.1),
           ("Torino", 45.07, 7.69, 1.7), ("Palermo", 38.12, 13.36, 1.0), ("Bologna", 44.49, 11.34, 1.0)],
}
VARS = ["temperature_2m", "relative_humidity_2m", "shortwave_radiation", "wind_speed_10m"]
CHAN = ["temp", "rhum", "rad", "wspd"]
HDRS = {"User-Agent": "Mozilla/5.0 (research script)"}


def fetch(url, dest, timeout=600):
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return True
    ctx = ssl.create_default_context()
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r, open(dest, "wb") as f:
                f.write(r.read())
            return True
        except Exception as e:                       # noqa: BLE001
            print(f"  [retry {attempt+1}] {os.path.basename(dest)}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    return False


def load_point(cc, city, la, lo):
    dest = os.path.join(RAW, f"era5_mp_{cc}_{city}.csv")
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={la}&longitude={lo}&start_date=2014-12-01&end_date=2020-10-01"
           f"&hourly={','.join(VARS)}&timezone=UTC&format=csv")
    if not fetch(url, dest):
        raise RuntimeError(f"download failed for {cc}/{city}")
    w = pd.read_csv(dest, skiprows=2)
    w.columns = ["time"] + CHAN
    w["time"] = pd.to_datetime(w["time"])
    return w.set_index("time").sort_index()


def build(cc):
    pts = POINTS[cc]
    frames, weights = [], []
    for city, la, lo, pop in pts:
        frames.append(load_point(cc, city, la, lo)); weights.append(pop)
        print(f"  {cc}/{city}: {len(frames[-1])} rows, weight {pop}", flush=True)
    wsum = float(sum(weights))
    idx = frames[0].index
    agg = sum(f.reindex(idx)[CHAN] * (p / wsum) for f, p in zip(frames, weights))
    z = np.load(os.path.join(PROC, f"OPSD-{cc}.npz"))
    meta = json.load(open(os.path.join(PROC, f"OPSD-{cc}.json")))
    stamps = pd.to_datetime(z["stamps"].astype(np.int64), unit="s")
    data = z["data"].astype(np.float32).copy()
    a = agg.reindex(stamps)
    a = a.interpolate(limit=6, limit_direction="both").ffill().bfill()
    n_fill = int(agg.reindex(stamps).isna().sum().sum())
    for ch in CHAN:
        j = meta["channels"].index(ch)
        data[:, j] = a[ch].to_numpy(dtype=np.float32)
    out = {k: z[k] for k in z.files}
    out["data"] = data
    name = f"OPSD-{cc}-mp"
    np.savez_compressed(os.path.join(PROC, name + ".npz"), **out)
    meta = dict(meta)
    meta["name"] = name
    meta["notes"] = (meta.get("notes", "") + f" MULTI-POINT WEATHER VARIANT (revision 2026-10-19): the four "
                     f"weather channels are the population-weighted mean of ERA5 at {len(pts)} metropolitan "
                     f"areas ({', '.join(f'{c} {p}M' for c, _, _, p in pts)}); {n_fill} cells interpolated; "
                     f"every other column identical to OPSD-{cc}.")
    meta["weather_points"] = [dict(city=c, lat=la, lon=lo, weight_millions=p) for c, la, lo, p in pts]
    json.dump(meta, open(os.path.join(PROC, name + ".json"), "w"), indent=2)
    # report how different the two representations are
    j = meta["channels"].index("temp")
    t1 = z["data"][:, j].astype(float); t2 = data[:, j].astype(float)
    print(f"  [{name}] written; temp: single-point vs weighted mean diff {np.mean(t2 - t1):+.2f} K, "
          f"rmsd {np.sqrt(np.mean((t2 - t1) ** 2)):.2f} K, corr {np.corrcoef(t1, t2)[0, 1]:.4f}", flush=True)


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    for cc in (sys.argv[1:] or list(POINTS)):
        build(cc)
