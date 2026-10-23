"""Plain-assert tests for src/stats.py.  Run:  python tests/test_stats.py"""
import glob
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
import stats  # noqa: E402

ERR = os.path.join(HERE, "results", "errors")


def ar1(rng, n, m, rho=0.3):
    """m loss series of length n with AR(1) dependence, positive, identical distribution."""
    z = rng.standard_normal((n, m))
    for t in range(1, n):
        z[t] = rho * z[t - 1] + np.sqrt(1 - rho ** 2) * z[t]
    return np.abs(z) + 0.5


def test_holm():
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.random(12)
        q = stats.holm(p)
        assert q.shape == p.shape
        assert np.all(q >= p - 1e-12), "Holm below raw"
        assert np.all(q <= 1.0)
        o = np.argsort(p)
        assert np.all(np.diff(q[o]) >= -1e-12), "Holm not monotone in raw order"
    # smallest p is multiplied by m, hand-checked
    q = stats.holm([0.01, 0.04, 0.03])
    assert abs(q[0] - 0.03) < 1e-12 and abs(q[2] - 0.06) < 1e-12 and abs(q[1] - 0.06) < 1e-12
    # NaN passes through and is excluded from the family
    q = stats.holm([0.01, np.nan, 0.5])
    assert np.isnan(q[1]) and abs(q[0] - 0.02) < 1e-12
    print("holm                       ok")


def test_bootstrap_indices():
    rng = np.random.default_rng(0)
    n, B, bl = 500, 200, 8
    idx = stats.stationary_bootstrap_indices(n, B, bl, rng)
    assert idx.shape == (B, n) and idx.min() >= 0 and idx.max() < n
    # consecutive draws continue the block (i+1 mod n) except at block starts
    cont = ((idx[:, 1:] - idx[:, :-1]) % n == 1).mean()
    assert abs(cont - (1 - 1 / bl)) < 0.02, cont
    # block_len 1 is the iid bootstrap
    idx1 = stats.stationary_bootstrap_indices(n, 50, 1, rng)
    assert abs(((idx1[:, 1:] - idx1[:, :-1]) % n == 1).mean() - 1 / n) < 0.01
    print("stationary bootstrap       ok")


def test_dm_import():
    rng = np.random.default_rng(3)
    e1 = np.abs(rng.standard_normal(2000))
    s, p = stats.dm_test(e1, e1.copy(), h=1)
    assert np.isnan(s) and np.isnan(p)                     # identical losses -> undefined
    s, p = stats.dm_test(e1, e1 + 0.3 + 0.1 * rng.standard_normal(2000), h=24)
    assert s < 0 and p < 1e-6, (s, p)                      # e1 clearly better -> negative stat
    tab = stats.pairwise_dm_table({"A": e1, "B": e1 + 0.3, "C": e1 + 0.001 * rng.standard_normal(2000)}, "A", 24)
    assert set(tab) == {"B", "C"} and tab["B"]["p_holm"] >= tab["B"]["p"]
    print("dm_test / pairwise table   ok")


def test_mcs_size(reps=40, n=600, m=5, B=400):
    """(a) identical distributions: all five models kept in >= 85% of replications."""
    kept = 0
    for r in range(reps):
        rng = np.random.default_rng(100 + r)
        L = ar1(rng, n, m)
        res = stats.mcs(L, alpha=0.10, B=B, seed=r)
        assert res["p_values"].shape == (m,) and len(res["elimination_order"]) == m
        order = res["elimination_order"]
        assert np.all(np.diff(res["p_values"][order]) >= 0), "MCS p-values not monotone"
        assert res["p_values"][order[-1]] == 1.0
        kept += int(res["in_set"].all())
    frac = kept / reps
    print(f"mcs size: all kept in {kept}/{reps} = {frac:.2f} (need >= 0.85)")
    assert frac >= 0.85, frac


def test_mcs_power(reps=40, n=600, m=5, B=400):
    """(b) one model clearly worse: excluded in >= 95% of replications."""
    excluded = 0
    for r in range(reps):
        rng = np.random.default_rng(500 + r)
        L = ar1(rng, n, m)
        L[:, 2] += 0.25                                     # ~0.4 sd shift on the last model
        res = stats.mcs(L, alpha=0.10, B=B, seed=r)
        excluded += int(not res["in_set"][2])
        assert res["elimination_order"][0] == 2 or res["in_set"][2]
    frac = excluded / reps
    print(f"mcs power: bad model excluded in {excluded}/{reps} = {frac:.2f} (need >= 0.95)")
    assert frac >= 0.95, frac
    # the TR statistic must reach the same verdict on a clear case
    rng = np.random.default_rng(7)
    L = ar1(rng, n, m)
    L[:, 0] += 0.25
    res = stats.mcs(L, alpha=0.10, B=B, stat="TR", seed=1)
    assert not res["in_set"][0]


def test_mcs_speed():
    rng = np.random.default_rng(11)
    L = np.abs(rng.standard_normal((10000, 25))) + 1.0
    t0 = time.time()
    res = stats.mcs(L, alpha=0.10, B=2000, seed=0)
    dt = time.time() - t0
    print(f"mcs 10000 x 25, B=2000: {dt:.1f} s, block_len={res['block_len']}, "
          f"{int(res['in_set'].sum())} in set")
    assert dt < 60, dt


def test_mcs_real():
    """(d) smoke run on the frozen v1 outputs: OPSD-DE, L=336, H=96, seed 2021, untagged."""
    pat = re.compile(r"^OPSD-DE_(?P<model>[A-Za-z]+)_L336_H96_s2021\.npz$")
    files = {}
    for f in sorted(glob.glob(os.path.join(ERR, "OPSD-DE_*_L336_H96_s2021.npz"))):
        mm = pat.match(os.path.basename(f))
        if mm:
            files[mm.group("model")] = f
    if not files:
        print("mcs real data: results/errors not present, skipped")
        return
    names = sorted(files)
    cols = [np.load(files[m])["per_window_ae"].astype(float) for m in names]
    n = {len(c) for c in cols}
    assert len(n) == 1, n
    L = np.stack(cols, axis=1)
    t0 = time.time()
    res = stats.mcs(L, alpha=0.10, B=2000, seed=0)
    dt = time.time() - t0
    print(f"mcs real data: OPSD-DE H=96, {L.shape[0]} windows x {L.shape[1]} models, "
          f"block_len={res['block_len']}, {dt:.1f} s")
    order = res["elimination_order"]
    for i in order:
        print(f"    {names[i]:14s} mean AE {L[:, i].mean():8.1f} MW   MCS p = {res['p_values'][i]:.3f}"
              f"   {'IN ' if res['in_set'][i] else 'out'}")
    surv = [names[i] for i in range(len(names)) if res["in_set"][i]]
    print("  survivors at alpha=0.10:", ", ".join(surv))
    best = names[int(np.argmin(L.mean(axis=0)))]
    assert best in surv, "the best model must always be in the MCS"
    # DM table against PLANET on the same data
    if "PLANET" in files:
        tab = stats.pairwise_dm_table({m: L[:, i] for i, m in enumerate(names)}, "PLANET", 96)
        for m in names:
            if m != "PLANET":
                r = tab[m]
                print(f"    DM PLANET vs {m:14s} stat {r['stat']:+7.2f}  p {r['p']:.4f}  Holm {r['p_holm']:.4f}")


if __name__ == "__main__":
    test_holm()
    test_bootstrap_indices()
    test_dm_import()
    test_mcs_size()
    test_mcs_power()
    test_mcs_speed()
    test_mcs_real()
    print("ALL TESTS PASSED")
