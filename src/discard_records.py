"""Move superseded result records out of results_v2/ before a stage is re-run under a changed
implementation, keeping them for the record under results_v2_discarded/<reason>/ with the same
sub-folder layout (json, errors/*.npz, failed/, stale/, prob/).

Reasons (2026-10-19 revision):
  norevin     Informer and Autoformer records made without instance normalisation
  tao_trend   Hong-regression records made before the trend / window selection
  prob_lookahead  probabilistic records calibrated with the look-ahead ACI update
"""
import argparse, glob, os, shutil, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results_v2")
OUT = os.path.join(HERE, "results_v2_discarded")

PATTERNS = {
    "norevin": (["", "errors", "failed", "stale"], ["*_Informer_*", "*_Autoformer_*"]),
    "tao_trend": (["", "errors", "failed", "stale"], ["*_TaoVanilla_*", "*_TaoRefit_*"]),
    "prob_lookahead": (["prob"], ["*"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reason", choices=list(PATTERNS))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    subs, pats = PATTERNS[a.reason]
    n = 0
    for sub in subs:
        for pat in pats:
            for p in sorted(glob.glob(os.path.join(RES, sub, pat))):
                if os.path.isdir(p):
                    continue
                dst = os.path.join(OUT, a.reason, sub, os.path.basename(p))
                n += 1
                if a.dry:
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(p, dst)
    print(f"[discard {a.reason}] {'would move' if a.dry else 'moved'} {n} files -> {os.path.join(OUT, a.reason)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
