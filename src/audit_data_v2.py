"""Audit the v2 processed corpus (data/proc_v2/) and write data/proc_v2/AUDIT.md.

One table row per dataset: T, C, channels, span, tz, DST transitions in span, holidays per year
(mean), bridge days, missing cells interpolated, share of rows with target <= 0, duplicate or
non-uniform timestamp count (must be 0), TSO-forecast coverage, and the v1 identity check
(data / stamps / aux bit-identical, splits equal) against the frozen data/proc/.

Usage: audit_data_v2.py            (prints the table and writes AUDIT.md; exits 1 on any FAIL)
"""
import os, re, json, sys, datetime as dt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(HERE, "data", "proc_v2")
PROC_V1 = os.path.join(HERE, "data", "proc")
STEP = {"1h": 3600, "15min": 900}


def v1_identity(name, z, meta):
    p = os.path.join(PROC_V1, name + ".npz")
    if not os.path.exists(p):
        return "n/a (new)"
    z1 = np.load(p)
    m1 = json.load(open(os.path.join(PROC_V1, name + ".json")))
    bad, ok = [], []
    for k in ("data", "stamps", "aux"):
        in1, in2 = k in z1.files, k in z.files
        if not in1 and not in2:
            continue
        same = (in1 and in2 and z1[k].dtype == z[k].dtype and z1[k].shape == z[k].shape
                and z1[k].tobytes() == z[k].tobytes())
        (ok if same else bad).append(k)
    (ok if m1["splits"] == meta["splits"] else bad).append("splits")
    return "PASS (" + ",".join(ok) + ")" if not bad else "FAIL (" + ",".join(bad) + ")"


def audit_one(name):
    z = np.load(os.path.join(PROC, name + ".npz"))
    meta = json.load(open(os.path.join(PROC, name + ".json")))
    data, stamps, local, flags = z["data"], z["stamps"], z["local_stamps"], z["cal_flags"]
    T, C = data.shape
    assert (T, C) == (meta["T"], meta["C"]), name
    assert flags.dtype == np.uint8 and flags.shape == (T, 3), name
    assert local.dtype == np.int64 and stamps.dtype == np.int64, name

    step = STEP[meta["freq"]]
    d = np.diff(stamps)
    n_bad_ts = int(np.count_nonzero(d != step)) + int(len(stamps) - len(np.unique(stamps)))
    offs = local - stamps
    n_dst = int(np.count_nonzero(np.diff(offs)))
    days = local // 86400
    span_years = (days[-1] - days[0] + 1) / 365.25
    hol_days = np.unique(days[flags[:, 0] == 1])
    br_days = np.unique(days[flags[:, 1] == 1])
    xm_days = np.unique(days[flags[:, 2] == 1])
    tgt = data[:, meta["target_idx"]]
    share_nonpos = float(np.mean(tgt <= 0))
    nan_cells = int(np.isnan(data).sum())
    m = re.search(r"missing cells interpolated: (\d+)", meta["notes"])
    interp = int(m.group(1)) if m else -1
    if "aux" in z.files:
        tso = f"{float(np.mean(~np.isnan(z['aux'][:, 0]))):.1%}"
    else:
        tso = "-"
    ident = v1_identity(name, z, meta)
    basis = "local(raw)" if "local_stamps == stamps" in meta["notes"] else "UTC"
    chans = meta["channels"] if C <= 8 else meta["channels"][:2] + ["...", meta["channels"][-1]]
    md = re.search(r"dropped: \[(.*?)\]\.", meta["notes"])
    dropped = md.group(1) if md else "-"
    return dict(name=name, T=T, C=C, channels=" ".join(chans),
                span=f"{meta['start'][:16]} -> {meta['end'][:16]} ({basis})",
                tz=meta["tz"], dst=n_dst, hol=f"{len(hol_days) / span_years:.1f} ({len(hol_days)})",
                bridge=len(br_days), xmas=len(xm_days), interp=interp, nan=nan_cells,
                nonpos=f"{share_nonpos:.2%}", badts=n_bad_ts, tso=tso, ident=ident,
                holsrc=meta["holiday_source"], dropped=dropped)


def main():
    names = sorted(f[:-5] for f in os.listdir(PROC) if f.endswith(".json"))
    order = ["OPSD-DE", "OPSD-FR", "OPSD-GB", "OPSD-ES", "OPSD-IT", "OPSD-NL", "OPSD-PL", "OPSD-BE",
             "ECL", "ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    names = [n for n in order if n in names] + [n for n in names if n not in order]
    rows = [audit_one(n) for n in names]
    cols = [("name", "dataset"), ("T", "T"), ("C", "C"), ("channels", "channels (target last)"),
            ("span", "span (stamps basis)"), ("tz", "tz"), ("dst", "DST trans."),
            ("holsrc", "holiday source"),
            ("hol", "holidays/yr (total days)"), ("bridge", "bridge days"), ("xmas", "xmas days"),
            ("interp", "missing cells interpolated"), ("nan", "NaN cells"),
            ("nonpos", "rows target<=0"), ("badts", "dup/non-uniform stamps"),
            ("tso", "TSO fc coverage"), ("ident", "v1 identity")]
    hdr = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [hdr, sep] + ["| " + " | ".join(str(r[k]) for k, _ in cols) + " |" for r in rows]
    table = "\n".join(lines)
    fails = [r["name"] for r in rows if r["ident"].startswith("FAIL") or r["badts"] or r["nan"]]
    doc = [
        "# data/proc_v2 audit", "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by src/audit_data_v2.py over {len(rows)} datasets.", "",
        "Definitions: *DST trans.* = number of changes of `local_stamps - stamps` inside the span; "
        "*holidays/yr* = distinct local holiday dates in span divided by span length in years (total distinct "
        "dates in brackets); *bridge days* / *xmas days* = distinct local dates flagged; *missing cells "
        "interpolated* is taken from the `notes` field written by preprocess_v2.py; *rows target<=0* is on the "
        "target channel (system load for grids, MT_369 for ECL, oil temperature OT for ETT); *dup/non-uniform "
        "stamps* counts duplicated stamps plus steps != freq (must be 0); *v1 identity* = bit-identity of "
        "`data`, `stamps`, `aux` and equality of `splits` with the frozen v1 file in data/proc/ "
        "(n/a for systems that are new in v2). *stamps basis* `local(raw)` marks datasets whose raw "
        "timestamps are already local civil time (ECL, ETT), for which `local_stamps == stamps`. "
        "*holiday source* = `holidays` package version and country[/subdivision] used (GB uses England: "
        "Easter Monday and the late-August bank holiday are England/Wales-level in the package).", "",
        table, "",
        "## Exogenous-series rules (grids)", "",
        "1. A solar/wind generation series is dropped if its raw data contains any single internal gap "
        "(NaN run between its first and last observation) longer than 720 h (30 days).",
        "2. Then (v1 rule) it is dropped if less than 50% of the load span is present.",
        "Leading NaN blocks are not gaps: the span is trimmed to the first fully observed row (v1 rule).", "",
        "| dataset | dropped series (reason) |", "|---|---|",
    ] + [f"| {r['name']} | {r['dropped']} |" for r in rows if r["dropped"] != "-"] + [
        "",
        ("**Result: all checks passed.**" if not fails else f"**Result: FAIL for {fails}.**"), "",
    ]
    with open(os.path.join(PROC, "AUDIT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    print(table)
    print("\nAUDIT ->", os.path.join(PROC, "AUDIT.md"), "| FAIL:" if fails else "| all checks passed", fails or "")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
