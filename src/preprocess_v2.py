"""Build the v2 processed corpus (data/proc_v2/) as a strict superset of the frozen v1 corpus.

Every dataset is stored as <NAME>.npz + <NAME>.json (contract: V2_SPEC.md):
    data         : float32 (T, C)  -- channel order, target channel LAST (v1 convention)
    stamps       : int64   (T,)    -- epoch seconds of each row, bit-identical to v1 where v1 exists
    local_stamps : int64   (T,)    -- local civil wall-clock time as a pseudo-UTC epoch (DST-aware)
    cal_flags    : uint8   (T, 3)  -- [holiday, bridge, xmas] evaluated on the LOCAL date
    aux          : float32 (T, A)  -- optional, not a model input (TSO day-ahead forecast; grids only)
json: v1 fields + country, tz, weather_idx, cal_flag_names, holiday_source.

Comparability: for OPSD-DE/FR/GB, ECL, ETTh1/h2/m1/m2 the arrays `data`, `stamps`, `aux` and the
`splits` are rebuilt with the v1 code path and asserted bit-identical to data/proc/ BEFORE anything is
written.  Nothing under data/proc/ is ever touched.

Usage: preprocess_v2.py [all|opsd|ecl|ett]      (idempotent; one summary line per dataset)
"""
import os, sys, json, time, datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import holidays

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")
PROC_V1 = os.path.join(HERE, "data", "proc")        # frozen v1 corpus (read-only)
PROC = os.path.join(HERE, "data", "proc_v2")
os.makedirs(PROC, exist_ok=True)

CAL_FLAG_NAMES = ["holiday", "bridge", "xmas"]
V1_NAMES = {"OPSD-DE", "OPSD-FR", "OPSD-GB", "ECL", "ETTh1", "ETTh2", "ETTm1", "ETTm2"}

# Datasets whose RAW timestamps are already local civil wall-clock time.  Verified on the raw files:
#   ECL  -> the UCI-documented DST artefacts sit at 01:15-02:00 *local* on every March (zeroed hour)
#           and October (doubled hour) change day of the raw file;
#   ETT  -> transformer oil temperature peaks at stamp-hour 14-16, i.e. Beijing afternoon.
# v1 stored those raw stamps unchanged as `stamps`, so for them local_stamps == stamps: adding the
# zone offset a second time would shift the local clock by +8 h (CN) / +1 h in summer (PT).
RAW_STAMPS_ARE_LOCAL = {"ECL", "ETTh1", "ETTh2", "ETTm1", "ETTm2"}

# Subdivision per country for holidays.country_holidays (None = national level).  GB -> England:
# England carries most of the GB load, and the package models Easter Monday and the late-August
# bank holiday (both real demand events) as England/Wales subdivision holidays, not GB-wide ones.
HOLIDAY_SUBDIV = {"GB": "ENG"}

# Exogenous-series rule (grids): a solar/wind generation series is dropped when its raw data has any
# single internal gap (a NaN run between its first and last observation) longer than GAP_MAX_H hours.
# Applied to every grid BEFORE the v1 >=50%-presence rule.  Leading/trailing NaN blocks are not gaps:
# the v1 rules handle them (span trimmed to the first fully observed row; tail ffilled).
GAP_MAX_H = 720


# --------------------------------------------------------------------------- calendar helpers
def utc_offsets(stamps, tz):
    """DST-aware UTC offset in seconds at each instant (zoneinfo)."""
    z = ZoneInfo(tz)
    return np.fromiter((int(dt.datetime.fromtimestamp(int(s), z).utcoffset().total_seconds())
                        for s in stamps), dtype=np.int64, count=len(stamps))


def calendar_flags(local_stamps, country, tz):
    """uint8 (T,3) [holiday, bridge, xmas] on the local civil date, plus the holiday table used."""
    days = local_stamps // 86400                              # local civil day index (epoch days)
    d0, d1 = int(days.min()) - 1, int(days.max()) + 1         # one day of margin for neighbours
    epoch = dt.date(1970, 1, 1)
    dates = [epoch + dt.timedelta(days=k) for k in range(d0, d1 + 1)]
    years = range(dates[0].year, dates[-1].year + 1)
    hol = holidays.country_holidays(country, subdiv=HOLIDAY_SUBDIV.get(country), years=years)
    is_hol = np.array([d in hol for d in dates], dtype=bool)
    is_wend = np.array([d.weekday() >= 5 for d in dates], dtype=bool)
    is_xmas = np.array([(d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 6)
                        for d in dates], dtype=bool)
    # classic bridge day: Mon-Fri, not a holiday, one neighbour a holiday, the other a weekend day
    bridge = np.zeros(len(dates), dtype=bool)
    for i in range(1, len(dates) - 1):
        if is_hol[i] or is_wend[i]:
            continue
        bridge[i] = (is_hol[i - 1] and is_wend[i + 1]) or (is_hol[i + 1] and is_wend[i - 1])
    idx = (days - d0).astype(np.int64)
    flags = np.stack([is_hol[idx], bridge[idx], is_xmas[idx]], axis=1).astype(np.uint8)
    sub = HOLIDAY_SUBDIV.get(country)
    source = f"holidays {holidays.__version__} {country}" + (f"/{sub}" if sub else "")
    return flags, source


def max_internal_gap(s):
    """Longest NaN run (in rows) strictly between the first and last valid observation of s."""
    fv, lv = s.first_valid_index(), s.last_valid_index()
    if fv is None:
        return 0
    isn = s.loc[fv:lv].isna().to_numpy()
    if not isn.any():
        return 0
    # run lengths of consecutive NaN rows
    edges = np.flatnonzero(np.diff(np.concatenate(([0], isn.astype(np.int8), [0]))))
    return int((edges[1::2] - edges[::2]).max())


# --------------------------------------------------------------------------- identity check
def check_v1_identity(name, out, splits):
    """Assert bit-identity of data/stamps/aux and equality of splits with the frozen v1 file."""
    if name not in V1_NAMES:
        return "n/a"
    z = np.load(os.path.join(PROC_V1, name + ".npz"))
    m = json.load(open(os.path.join(PROC_V1, name + ".json")))
    bad = []
    for k in ("data", "stamps", "aux"):
        a = z[k] if k in z.files else None
        b = out.get(k)
        if a is None and b is None:
            continue
        same = (a is not None and b is not None and a.dtype == b.dtype and a.shape == b.shape
                and a.tobytes() == b.tobytes())
        if not same:
            bad.append(k)
    if m["splits"] != splits:
        bad.append("splits")
    assert not bad, f"{name}: v2 arrays differ from v1 in {bad}"
    return "PASS"


# --------------------------------------------------------------------------- writer
def save(name, df, target, freq, splits, country, tz, aux=None, notes="", weather=True):
    """df: DatetimeIndex + numeric columns; target moved to the last column (v1 save() logic)."""
    cols = [c for c in df.columns if c != target] + [target]
    df = df[cols]
    data = df.to_numpy(dtype=np.float32)
    stamps = df.index.values.astype("datetime64[s]").astype(np.int64)
    out = {"data": data, "stamps": stamps}
    aux_names = []
    if aux is not None:
        aux = aux.reindex(df.index)
        out["aux"] = aux.to_numpy(dtype=np.float32)
        aux_names = list(aux.columns)
    ident = check_v1_identity(name, out, splits)           # raises before anything is written

    if name in RAW_STAMPS_ARE_LOCAL:
        offs = np.zeros(len(stamps), dtype=np.int64)
        notes += (f" Raw timestamps are local civil time ({tz}); v1 stored them unchanged as `stamps`, "
                  f"so local_stamps == stamps (zone offset not added).")
    else:
        offs = utc_offsets(stamps, tz)
    local = stamps + offs
    flags, hol_src = calendar_flags(local, country, tz)
    out["local_stamps"] = local
    out["cal_flags"] = flags
    np.savez_compressed(os.path.join(PROC, name + ".npz"), **out)

    meta = {
        "name": name, "channels": cols, "target": target, "target_idx": len(cols) - 1,
        "freq": freq, "T": int(len(df)), "C": int(len(cols)),
        "start": str(df.index[0]), "end": str(df.index[-1]),
        "splits": splits, "aux": aux_names, "notes": notes.strip(),
        "country": country, "tz": tz,
        "weather_idx": list(range(len(cols) - 1)) if weather else [],
        "cal_flag_names": CAL_FLAG_NAMES, "holiday_source": hol_src,
    }
    with open(os.path.join(PROC, name + ".json"), "w") as f:
        json.dump(meta, f, indent=2)

    days = local // 86400
    n_hol = len(np.unique(days[flags[:, 0] == 1]))
    n_br = len(np.unique(days[flags[:, 1] == 1]))
    n_dst = int(np.count_nonzero(np.diff(offs)))
    print(f"  [v2] {name:8s} T={len(df):6d} C={len(cols):4d} {df.index[0]} -> {df.index[-1]}  "
          f"tz={tz:16s} dst={n_dst:2d} hol_days={n_hol:3d} bridge_days={n_br:2d} nan={int(np.isnan(data).sum())}  v1={ident}")
    return meta


# --------------------------------------------------------------------------- ETT (v1 code path)
def build_ett():
    print("== ETT ==")
    for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        df = pd.read_csv(os.path.join(RAW, name + ".csv"), parse_dates=["date"]).set_index("date")
        df = df.astype(np.float32)
        m = 30 * 24 if name.startswith("ETTh") else 30 * 24 * 4       # 12 / 4 / 4 months
        n_tr, n_va, n_te = 12 * m, 4 * m, 4 * m
        splits = {"train": [0, n_tr], "val": [n_tr, n_tr + n_va], "test": [n_tr + n_va, n_tr + n_va + n_te]}
        df = df.iloc[: n_tr + n_va + n_te]
        save(name, df, target="OT", freq="1h" if name.startswith("ETTh") else "15min", splits=splits,
             country="CN", tz="Asia/Shanghai", weather=False,
             notes="Electricity Transformer Temperature (Zhou et al., AAAI-21). 12/4/4-month split. "
                   "missing cells interpolated: 0.")


# --------------------------------------------------------------------------- ECL (v1 code path)
def build_ecl():
    print("== ECL (UCI LD2011_2014 -> 321 clients, hourly) ==")
    p = os.path.join(RAW, "LD2011_2014.txt")
    df = pd.read_csv(p, sep=";", decimal=",", index_col=0, parse_dates=True, low_memory=False)
    df.index.name = "date"
    a = df.to_numpy(dtype=np.float32)
    nz = a > 0
    first = np.where(nz.any(0), np.argmax(nz, axis=0), len(a))
    cutoff = df.index.searchsorted(pd.Timestamp("2012-02-01"))
    active = [c for c, f in zip(df.columns, first) if f < cutoff]
    h = df[active].resample("1h", label="left", closed="right").mean()
    h = h.loc["2012-01-01":"2014-12-31"].astype(np.float32)
    assert len(h) == 26304, len(h)
    assert h.shape[1] == 321, h.shape
    n = len(h)
    n_tr, n_va = int(n * 0.7), int(n * 0.1)
    splits = {"train": [0, n_tr], "val": [n_tr, n_tr + n_va], "test": [n_tr + n_va, n]}
    save("ECL", h, target=h.columns[-1], freq="1h", splits=splits, country="PT", tz="Europe/Lisbon",
         weather=False,
         notes="UCI ElectricityLoadDiagrams20112014 rebuilt from the 15-min file; 321 clients, 2012-2014, "
               "7:1:2 split. missing cells interpolated: 0. The raw file carries the documented DST "
               "artefacts (zeroed 01:00-02:00 local on the March change day, doubled on the October one).")


# --------------------------------------------------------------------------- OPSD
# v1 trio: column choice pinned exactly as v1 (bit-identity is asserted).
OPSD_SPEC = {
    "DE": dict(load="DE_load_actual_entsoe_transparency", fc="DE_load_forecast_entsoe_transparency",
               solar="DE_solar_generation_actual", wind="DE_wind_generation_actual"),
    "FR": dict(load="FR_load_actual_entsoe_transparency", fc="FR_load_forecast_entsoe_transparency",
               solar="FR_solar_generation_actual", wind="FR_wind_onshore_generation_actual"),
    "GB": dict(load="GB_GBN_load_actual_entsoe_transparency", fc="GB_GBN_load_forecast_entsoe_transparency",
               solar="GB_GBN_solar_generation_actual", wind="GB_GBN_wind_generation_actual"),
}
NEW_CC = ["ES", "IT", "NL", "PL", "BE"]
OPSD_TZ = {"DE": "Europe/Berlin", "FR": "Europe/Paris", "GB": "Europe/London", "ES": "Europe/Madrid",
           "IT": "Europe/Rome", "NL": "Europe/Amsterdam", "PL": "Europe/Warsaw", "BE": "Europe/Brussels"}


def _candidates(cc):
    p = f"{cc}_GBN" if cc == "GB" else cc
    return dict(load=f"{p}_load_actual_entsoe_transparency", fc=f"{p}_load_forecast_entsoe_transparency",
                solar=[f"{p}_solar_generation_actual"],
                wind=[f"{p}_wind_generation_actual", f"{p}_wind_onshore_generation_actual"])


def build_opsd():
    print("== OPSD national grids + ERA5 weather ==")
    cand = {cc: _candidates(cc) for cc in NEW_CC}
    need = {"utc_timestamp"}
    need |= {v for s in OPSD_SPEC.values() for v in s.values()}
    need |= {v for s in cand.values() for k in ("load", "fc") for v in [s[k]]}
    need |= {v for s in cand.values() for k in ("solar", "wind") for v in s[k]}
    o = pd.read_csv(os.path.join(RAW, "opsd_60min.csv"), usecols=lambda c: c in need,
                    parse_dates=["utc_timestamp"], low_memory=False)
    o = o.set_index("utc_timestamp").sort_index()
    o.index = o.index.tz_convert(None) if o.index.tz is not None else o.index

    for cc in list(OPSD_SPEC) + NEW_CC:
        w = pd.read_csv(os.path.join(RAW, f"era5_{cc}.csv"), skiprows=2)
        w.columns = ["time", "temp", "rhum", "rad", "wspd"]
        w["time"] = pd.to_datetime(w["time"])
        w = w.set_index("time").sort_index()
        if w.index.tz is not None:
            w.index = w.index.tz_convert(None)

        # --- column selection: v1 trio pinned; new systems pick the best-covered candidate
        choice_log = []
        if cc in OPSD_SPEC:
            spec = OPSD_SPEC[cc]
        else:
            c = cand[cc]
            valid = o[c["load"]].notna()
            lo_, hi_ = o.index[valid][0], o.index[valid][-1]
            span = o.loc[lo_:hi_]
            spec = dict(load=c["load"], fc=c["fc"])
            for k in ("solar", "wind"):
                cov = {col: float(span[col].notna().mean()) for col in c[k] if col in span}
                best = max(cov, key=cov.get) if cov else None
                spec[k] = best if best is not None else "<absent>"
                choice_log.append(f"{k}: " + ", ".join(f"{col}={v:.1%}" for col, v in cov.items())
                                  + (f" -> {best}" if best else " -> none present"))

        d = pd.DataFrame(index=o.index)
        d["load"] = o[spec["load"]]
        d["solar"] = o[spec["solar"]] if spec["solar"] in o else np.nan
        d["wind"] = o[spec["wind"]] if spec["wind"] in o else np.nan
        tso = (o[spec["fc"]].rename("tso_forecast") if spec["fc"] in o
               else pd.Series(np.nan, index=o.index, name="tso_forecast"))

        valid = d["load"].notna()
        if valid.sum() == 0:
            print(f"   [skip] {cc}: no load data");  continue
        lo, hi = d.index[valid][0], d.index[valid][-1]
        d, tso = d.loc[lo:hi], tso.loc[lo:hi]
        d = d.join(w[["temp", "rhum", "rad", "wspd"]], how="left")

        dropped, gap_kept = [], {}
        # rule 1 (v2): no single internal gap longer than GAP_MAX_H hours
        for c in ["solar", "wind"]:
            g = max_internal_gap(d[c])
            if g > GAP_MAX_H:
                dropped.append(f"{c}={spec[c]} (internal gap of {g} h > {GAP_MAX_H} h)")
                d = d.drop(columns=[c])
            else:
                gap_kept[c] = g
        # rule 2 (v1): at least 50% of the load span present
        for c in ["solar", "wind"]:
            if c in d and d[c].notna().sum() < 0.5 * len(d):
                dropped.append(f"{c}={spec[c]} ({d[c].notna().mean():.1%} present < 50%)")
                d = d.drop(columns=[c]); gap_kept.pop(c, None)
        first_ok = d.dropna().index
        trimmed = ""
        if len(first_ok):
            if first_ok[0] != d.index[0]:
                trimmed = f" span start trimmed from {d.index[0]} (first load row) to {first_ok[0]} (first fully observed row)."
            d = d.loc[first_ok[0]:]
            tso = tso.loc[first_ok[0]:]
        per_col = {c: int(v) for c, v in d.isna().sum().items() if v}
        gaps = int(d.isna().sum().sum())
        d = d.interpolate(limit=6, limit_direction="both").ffill().bfill()
        d = d.astype(np.float32)
        n = len(d)
        n_tr, n_va = int(n * 0.7), int(n * 0.1)
        splits = {"train": [0, n_tr], "val": [n_tr, n_tr + n_va], "test": [n_tr + n_va, n]}
        cov = float(tso.notna().mean())
        used = {c: f"{spec[c]} (max internal gap {gap_kept[c]} h)" for c in ("solar", "wind") if c in d.columns}
        for line in choice_log:
            print(f"   {cc}: {line}")
        if dropped:
            print(f"   {cc}: dropped {'; '.join(dropped)}")
        print(f"   {cc}: kept {used if used else 'none'}; interpolated {gaps} missing cells {per_col}; "
              f"TSO day-ahead forecast coverage {cov:.1%}")
        notes = (f"ENTSO-E {cc} system load (OPSD time_series 60min, {spec['load']}) + ERA5 weather (Open-Meteo). "
                 f"aux column holds the TSO's own published day-ahead load forecast ({spec['fc']}, coverage {cov:.1%}). "
                 f"exogenous-series rules: (1) a solar/wind series is dropped if its raw data has any single internal "
                 f"gap longer than {GAP_MAX_H} h (30 d); (2) then dropped if <50% present over the load span (v1). "
                 f"generation columns kept: {used if used else 'none'}. "
                 f"dropped: [{'; '.join(dropped) if dropped else 'none'}]."
                 + trimmed
                 + f" missing cells interpolated: {gaps} (per channel {per_col if per_col else '{}'}; "
                   f"linear up to 6 h, then ffill/bfill).")
        save(f"OPSD-{cc}", d, target="load", freq="1h", splits=splits, country=cc, tz=OPSD_TZ[cc],
             aux=tso.to_frame(), notes=notes)


if __name__ == "__main__":
    t0 = time.time()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    assert which in ("all", "opsd", "ecl", "ett"), which
    if which in ("all", "ett"):  build_ett()
    if which in ("all", "opsd"): build_opsd()
    if which in ("all", "ecl"):  build_ecl()
    print(f"\n== PROC_V2 DIR ({time.time() - t0:.0f}s) ==")
    for f in sorted(os.listdir(PROC)):
        print("  ", f, f"{os.path.getsize(os.path.join(PROC, f)) / 1e6:.2f} MB")
