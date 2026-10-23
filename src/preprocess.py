"""Build the processed benchmark corpus for the smart-grid load forecasting study.

Every dataset is stored as an .npz holding
    data   : float32 (T, C)  -- channel order, target channel LAST (LTSF convention)
    stamps : int64  (T,)     -- true UTC epoch seconds for each row (needed for phase-locked bases)
    aux    : float32 (T, A)  -- optional incumbent forecasts / reference series (not model inputs)
and a JSON sidecar with channel names, frequency, and the train/val/test borders.
"""
import os, json, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")
PROC = os.path.join(HERE, "data", "proc")
os.makedirs(PROC, exist_ok=True)


def save(name, df, target, freq, splits, aux=None, notes=""):
    """df: DatetimeIndex + numeric columns. target moved to last column."""
    cols = [c for c in df.columns if c != target] + [target]
    df = df[cols]
    data = df.to_numpy(dtype=np.float32)
    stamps = (df.index.view("int64") // 10**9).astype(np.int64)
    out = {"data": data, "stamps": stamps}
    aux_names = []
    if aux is not None:
        aux = aux.reindex(df.index)
        out["aux"] = aux.to_numpy(dtype=np.float32)
        aux_names = list(aux.columns)
    np.savez_compressed(os.path.join(PROC, name + ".npz"), **out)
    meta = {
        "name": name, "channels": cols, "target": target, "target_idx": len(cols) - 1,
        "freq": freq, "T": int(len(df)), "C": int(len(cols)),
        "start": str(df.index[0]), "end": str(df.index[-1]),
        "splits": splits, "aux": aux_names, "notes": notes,
    }
    with open(os.path.join(PROC, name + ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    nan = int(np.isnan(data).sum())
    print(f"  [save] {name:10s} T={len(df):6d} C={len(cols):4d} {df.index[0]} -> {df.index[-1]}  nan={nan}  splits={splits}")
    return meta


# --------------------------------------------------------------------------- ETT
def build_ett():
    print("== ETT ==")
    for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        df = pd.read_csv(os.path.join(RAW, name + ".csv"), parse_dates=["date"]).set_index("date")
        df = df.astype(np.float32)
        # Informer/Autoformer protocol: 12 / 4 / 4 months
        m = 30 * 24 if name.startswith("ETTh") else 30 * 24 * 4
        n_tr, n_va, n_te = 12 * m, 4 * m, 4 * m
        splits = {"train": [0, n_tr], "val": [n_tr, n_tr + n_va], "test": [n_tr + n_va, n_tr + n_va + n_te]}
        df = df.iloc[: n_tr + n_va + n_te]
        save(name, df, target="OT", freq="1h" if name.startswith("ETTh") else "15min",
             splits=splits, notes="Electricity Transformer Temperature (Zhou et al., AAAI-21). 12/4/4-month split.")


# --------------------------------------------------------------------------- ECL
def build_ecl():
    print("== ECL (UCI LD2011_2014 -> 321 clients, hourly, true timestamps) ==")
    p = os.path.join(RAW, "LD2011_2014.txt")
    df = pd.read_csv(p, sep=";", decimal=",", index_col=0, parse_dates=True, low_memory=False)
    df.index.name = "date"
    print(f"   raw: {df.shape}, {df.index[0]} -> {df.index[-1]}")
    # Canonical client selection (reproduces the 321-client benchmark of Lai et al. / Zhou et al.):
    # keep every client whose meter is already active at the start of the 2012 record.
    a = df.to_numpy(dtype=np.float32)
    nz = a > 0
    first = np.where(nz.any(0), np.argmax(nz, axis=0), len(a))
    cutoff = df.index.searchsorted(pd.Timestamp("2012-02-01"))
    active = [c for c, f in zip(df.columns, first) if f < cutoff]
    print(f"   clients metering before 2012-02-01: {len(active)} of {df.shape[1]}")
    # 15-min kW readings -> hourly consumption
    h = df[active].resample("1h", label="left", closed="right").mean()
    h = h.loc["2012-01-01":"2014-12-31"].astype(np.float32)
    assert len(h) == 26304, len(h)
    assert h.shape[1] == 321, h.shape
    n = len(h)
    n_tr, n_va = int(n * 0.7), int(n * 0.1)
    splits = {"train": [0, n_tr], "val": [n_tr, n_tr + n_va], "test": [n_tr + n_va, n]}
    save("ECL", h, target=h.columns[-1], freq="1h", splits=splits,
         notes="UCI ElectricityLoadDiagrams20112014 rebuilt with true UTC timestamps; 321 clients, 2012-2014, 7:1:2 split.")


# --------------------------------------------------------------------------- OPSD
OPSD_SPEC = {
    "DE": dict(load="DE_load_actual_entsoe_transparency",
               fc="DE_load_forecast_entsoe_transparency",
               solar="DE_solar_generation_actual",
               wind="DE_wind_generation_actual"),
    "FR": dict(load="FR_load_actual_entsoe_transparency",
               fc="FR_load_forecast_entsoe_transparency",
               solar="FR_solar_generation_actual",
               wind="FR_wind_onshore_generation_actual"),
    "GB": dict(load="GB_GBN_load_actual_entsoe_transparency",
               fc="GB_GBN_load_forecast_entsoe_transparency",
               solar="GB_GBN_solar_generation_actual",
               wind="GB_GBN_wind_generation_actual"),
}


def build_opsd():
    print("== OPSD national grids + ERA5 weather ==")
    need = ["utc_timestamp"] + sorted({v for s in OPSD_SPEC.values() for v in s.values()})
    o = pd.read_csv(os.path.join(RAW, "opsd_60min.csv"), usecols=lambda c: c in need,
                    parse_dates=["utc_timestamp"], low_memory=False)
    o = o.set_index("utc_timestamp").sort_index()
    o.index = o.index.tz_convert(None) if o.index.tz is not None else o.index

    for cc, spec in OPSD_SPEC.items():
        w = pd.read_csv(os.path.join(RAW, f"era5_{cc}.csv"), skiprows=2)
        w.columns = ["time", "temp", "rhum", "rad", "wspd"]
        w["time"] = pd.to_datetime(w["time"])
        w = w.set_index("time").sort_index()
        if w.index.tz is not None:
            w.index = w.index.tz_convert(None)

        d = pd.DataFrame(index=o.index)
        d["load"] = o[spec["load"]]
        d["solar"] = o[spec["solar"]] if spec["solar"] in o else np.nan
        d["wind"] = o[spec["wind"]] if spec["wind"] in o else np.nan
        tso = o[spec["fc"]].rename("tso_forecast") if spec["fc"] in o else pd.Series(np.nan, index=o.index, name="tso_forecast")

        # restrict to the span where the load series is present
        valid = d["load"].notna()
        if valid.sum() == 0:
            print(f"   [skip] {cc}: no load data");  continue
        lo, hi = d.index[valid][0], d.index[valid][-1]
        d, tso = d.loc[lo:hi], tso.loc[lo:hi]
        d = d.join(w[["temp", "rhum", "rad", "wspd"]], how="left")

        # solar/wind may be absent early in the record -> drop leading all-NaN block, then interpolate
        for c in ["solar", "wind"]:
            if d[c].notna().sum() < 0.5 * len(d):
                d = d.drop(columns=[c]); print(f"   {cc}: dropping {c} (too sparse)")
        first_ok = d.dropna().index
        if len(first_ok):
            d = d.loc[first_ok[0]:]
            tso = tso.loc[first_ok[0]:]
        gaps = int(d.isna().sum().sum())
        d = d.interpolate(limit=6, limit_direction="both").ffill().bfill()
        d = d.astype(np.float32)
        n = len(d)
        n_tr, n_va = int(n * 0.7), int(n * 0.1)
        splits = {"train": [0, n_tr], "val": [n_tr, n_tr + n_va], "test": [n_tr + n_va, n]}
        cov = float(tso.notna().mean())
        print(f"   {cc}: interpolated {gaps} missing cells; TSO day-ahead forecast coverage {cov:.1%}")
        save(f"OPSD-{cc}", d, target="load", freq="1h", splits=splits,
             aux=tso.to_frame(),
             notes=f"ENTSO-E {cc} system load (OPSD time_series) + ERA5 weather (Open-Meteo). "
                   f"aux column holds the TSO's own published day-ahead load forecast.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ett"):  build_ett()
    if which in ("all", "opsd"): build_opsd()
    if which in ("all", "ecl"):  build_ecl()
    print("\n== PROC DIR ==")
    for f in sorted(os.listdir(PROC)):
        print("  ", f, f"{os.path.getsize(os.path.join(PROC,f))/1e6:.2f} MB")
