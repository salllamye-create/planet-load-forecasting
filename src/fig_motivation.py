"""Motivation figure: the calendar structure PLANET exploits, and the regime shift
that sits inside the OPSD test split."""
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle as fs

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(HERE, "data", "proc")

# sequential single-hue ramp (light -> dark) for the ordered year dimension
YEAR_RAMP = ["#A8C8E8", "#7FA9D8", "#5A8BC7", "#3A6FAF", "#1F5493", "#0B3A6B"]


def load(name):
    z = np.load(os.path.join(PROC, name + ".npz"))
    meta = json.load(open(os.path.join(PROC, name + ".json")))
    idx = pd.to_datetime(z["stamps"], unit="s")
    df = pd.DataFrame(z["data"], columns=meta["channels"], index=idx)
    return df, meta


def main():
    fs.setup()
    df, meta = load("OPSD-DE")
    y = df["load"] / 1000.0                     # GW
    tr, va, te = (meta["splits"][k] for k in ("train", "val", "test"))

    fig = plt.figure(figsize=(fs.DBL, 2.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.55], wspace=0.30)

    # (a) mean load profile by hour of day, one line per year -> level and shape both drift
    ax = fig.add_subplot(gs[0, 0])
    years = sorted(y.index.year.unique())
    years = [yr for yr in years if (y.index.year == yr).sum() > 5000]
    for c, yr in zip(YEAR_RAMP, years):
        s = y[y.index.year == yr]
        ax.plot(s.groupby(s.index.hour).mean(), color=c, lw=1.3,
                label=str(yr), zorder=3)
    ax.set_xlabel("hour of day (UTC)"); ax.set_ylabel("mean load (GW)")
    ax.set_title("(a) Daily profile drifts year to year", loc="left", fontsize=7.6)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.legend(ncol=2, loc="upper left", fontsize=6.2, handlelength=1.4,
              borderaxespad=0.15, labelspacing=0.28)

    # (b) mean load profile by day of week -> the day-type structure the basis encodes
    ax = fig.add_subplot(gs[0, 1])
    for c, yr in zip(YEAR_RAMP, years):
        s = y[y.index.year == yr]
        ax.plot(s.groupby(s.index.dayofweek).mean(), color=c, lw=1.3, zorder=3)
    ax.set_xticks(range(7)); ax.set_xticklabels(list("MTWTFSS"))
    ax.set_xlabel("day of week"); ax.set_ylabel("mean load (GW)")
    ax.set_title("(b) Weekday/weekend split is stable", loc="left", fontsize=7.6)

    # (c) the whole record, with splits and the COVID-19 demand shock inside the test span
    ax = fig.add_subplot(gs[0, 2])
    w = y.resample("7D").mean()
    ax.plot(w.index, w.values, color=fs.INK, lw=0.85, zorder=4)
    for (a0, a1), lab, col in [(tr, "train", "#F2F2F2"), (va, "val", "#E2E9F2"),
                               (te, "test", "#FBE7DB")]:
        ax.axvspan(y.index[a0], y.index[min(a1, len(y) - 1)], color=col, zorder=0)
        ax.text(y.index[(a0 + min(a1, len(y) - 1)) // 2], w.max() * 1.005, lab,
                ha="center", va="bottom", fontsize=6.3, color=fs.MUTED)
    lo, hi = pd.Timestamp("2020-03-16"), pd.Timestamp("2020-05-10")
    ax.axvspan(lo, hi, color="#C44601", alpha=0.16, zorder=1)
    ax.annotate("COVID-19 lockdown\n($-$8.5 % demand, inside test)",
                xy=(pd.Timestamp("2020-04-05"), w.loc["2020-04-01":"2020-05-01"].mean()),
                xytext=(pd.Timestamp("2017-10-01"), w.min() * 0.995),
                fontsize=6.3, color="#C44601", ha="left",
                arrowprops=dict(arrowstyle="->", color="#C44601", lw=0.7,
                                shrinkA=0, shrinkB=2))
    ax.set_ylabel("weekly mean load (GW)")
    ax.set_title("(c) A regime shift falls inside the evaluation window",
                 loc="left", fontsize=7.6)
    ax.margins(x=0.01)
    fs.save(fig, "fig_motivation")

    # ---- quantitative facts quoted in the text
    pre = y.loc["2019-03-16":"2019-05-10"].mean()
    post = y.loc["2020-03-16":"2020-05-10"].mean()
    out = {
        "covid_drop_pct": round(100 * (post - pre) / pre, 2),
        "test_start": str(y.index[te[0]]), "test_end": str(y.index[-1]),
        "mean_load_GW": round(float(y.mean()), 2),
        "peak_load_GW": round(float(y.max()), 2),
        "years": [int(v) for v in years],
    }
    for nm in ["OPSD-DE", "OPSD-FR", "OPSD-GB", "ECL", "ETTh1", "ETTm1"]:
        d, m = load(nm)
        out[nm] = dict(T=m["T"], C=m["C"], start=m["start"][:10], end=m["end"][:10],
                       freq=m["freq"])
    json.dump(out, open(os.path.join(HERE, "results", "dataset_facts.json"), "w"), indent=2)
    print(json.dumps(out, indent=2)[:900])


if __name__ == "__main__":
    main()
