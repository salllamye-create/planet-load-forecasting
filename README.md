# Closed-Form Calendar Estimation in Deep Load Forecasting

Code, processed corpus, result records and manuscript sources for

> A. Salam, *Closed-Form Calendar Estimation in Deep Load Forecasting: An Information-Controlled
> Benchmark on Eight National Grids*, submitted to Energy and AI (2026); preprint on arXiv.

The study compares 26 learned forecasters on the hourly national load of eight European
systems (DE, FR, GB, ES, IT, NL, PL, BE) under three controlled information regimes, with
Electricity and the four ETT series as generic benchmarks. PLANET is the proposed model: a
calendar tier whose coefficients are solved in closed form inside every forward pass, on top of
a residual deep backbone.

Every number quoted in the manuscript is a macro generated from the result records in this
repository; the build refuses a hand-typed result or an undefined macro.

## Layout

| Path | Content |
|---|---|
| `src/preprocess_v2.py`, `src/audit_data_v2.py` | corpus build from the raw sources and its audit (`data/proc_v2/AUDIT.md`) |
| `src/weather_multipoint.py` | population-weighted multi-point weather corpora (`OPSD-{FR,ES,IT}-mp`) |
| `src/data_v2.py`, `src/exo.py` | windowing, the covariate block of the three regimes, the shared exogenous head |
| `src/planet.py`, `src/planetx.py` | PLANET (closed-form calendar tier + residual backbone) and PLANET-X |
| `src/baselines.py`, `src/nf_models.py` | the re-implemented baselines and the neuralforecast library models behind one forward signature |
| `src/exp_v2.py` | the single training and evaluation loop every neural model goes through |
| `src/run_all_v2.py` | job matrix: tuning, lookback selection, headline grid, ablations, mechanism experiments |
| `src/prob_v2.py` | quantile heads, conformalised quantile regression, delayed-feedback adaptive conformal inference |
| `src/dayahead_v2.py`, `src/refit_dayahead.py` | day-ahead protocol against the operators' published forecasts; monthly-refit variant |
| `src/gbrt_v2.py`, `src/tao_vanilla.py`, `src/classic_v2.py` | gradient-boosted trees and Hong's regression (once fitted, monthly refit, with an AR residual model) |
| `src/fm_zero_shot.py` | zero-shot foundation models (Chronos-2, TimesFM-2.0), inference only |
| `src/combo_v2.py` | forecast combinations on the headline grid |
| `src/stats.py`, `tests/test_stats.py` | Diebold-Mariano, Holm, model confidence set |
| `src/paper_numbers.py` | the record selection and every quoted number (`paper/numbers.tex`) |
| `src/make_tables_v2.py`, `src/fig_results_v2.py` | every table and figure, from the same selection |
| `src/pipeline_v2.sh`, `src/pipeline_rev.sh`, `src/pipeline_rev2.sh` | the GPU stages in the order they ran |
| `data/proc_v2/` | the processed corpus (13 datasets, `.npz` + `.json` sidecars) |
| `results_v2/` | one JSON record per run, per-window errors (`errors/`), probabilistic records (`prob/`), day-ahead scores |
| `results_v2_discarded*/` | records superseded by a logged protocol correction, kept for the audit trail (in the Zenodo archive, not in this repository) |
| `paper/` | manuscript sources (IEEE and Elsevier layouts), bibliography, generated tables, build scripts |
| `PLAN_v2.md`, `V2_SPEC.md` | the decision log of the study and the file-format contracts |

## Environment

* One NVIDIA Quadro RTX 4000 (8 GB) at a 100 W power cap, Windows 11. Every neural run used
  this card; the classical baselines, the statistics and the manuscript build run on the CPU.
* Python 3.11; exact package versions in `requirements_v2_freeze.txt` (torch 2.4.1 + CUDA 12.1,
  neuralforecast 3.1.8, transformers 4.57.x, timesfm 1.3.0, chronos-forecasting 2.x,
  numpy 1.26.4, pandas 2.3.3, scipy 1.17.1).
* A TeX distribution with `pdflatex`, `bibtex`, `pdfinfo` and `pdffonts` for the manuscript.

## Reproducing the study

### 1. Corpus

```
python src/preprocess_v2.py          # data/raw -> data/proc_v2/<NAME>.npz + .json
python src/audit_data_v2.py          # data/proc_v2/AUDIT.md; exits 1 on a failed check
python src/weather_multipoint.py     # the three multi-point weather corpora
```

The processed corpus is included, so this step is only needed to rebuild it from the raw
sources (Open Power System Data, the ENTSO-E Transparency Platform, ERA5 through the
Open-Meteo archive, public-holiday calendars; `download_data.py` lists the URLs).

### 2. Experiments

```
python src/gbrt_v2.py                              # CPU: trees under R0/R1/R2, three seeds
python src/tao_vanilla.py --model TaoVanilla       # CPU: Hong's regression; --model TaoRefit
python src/tao_vanilla.py --model TaoRefitAR       #      and the refit + AR residual variant
bash   src/pipeline_v2.sh                          # GPU: the main pass (serial, resumable)
bash   src/pipeline_rev.sh                         # GPU: revision stages, part 1
bash   src/pipeline_rev2.sh                        # GPU: revision stages, part 2
python src/fm_zero_shot.py                         # GPU: zero-shot foundation models
python src/combo_v2.py                             # CPU: forecast combinations
python src/dayahead_v2.py                          # CPU: day-ahead scoring against the operators
```

`pipeline_v2.sh` runs the stages of `run_all_v2.py` in order: width and learning rate on the two
reference systems (DE, FR) and learning rate on the other six, for every model; lookback per
system and horizon for every model; the headline grid (regime R2, three seeds, eight systems,
horizons 24/96/336 h); the day-ahead cells (48 h) and regimes R0 and R1; the PLANET ablation on
both backbones; the mechanism experiments (learning curves, drift, transfer); the probabilistic
runs; Electricity and ETT. The two revision scripts add the stages the review of the study asked
for: the pathway control (TiDE via the shared head), the budget check at 90 epochs, the
probabilistic re-run with the delayed-feedback calibration, the correlated weather-noise and
multi-point weather sensitivities, the learning-rate selection under R0 and R1 and the re-run of the affected German cells
at their own rate, two further ablation seeds, the monthly-refit day-ahead protocol, and the ETT/Electricity re-run with a
learning rate selected per dataset. Every stage skips finished runs on restart; a record made
under a superseded selection is moved to `results_v2/stale/` before its stage runs; a run that
fails twice in separate processes is recorded in `results_v2/failed/` and appears as an empty
cell. Which records were discarded, when and why is in `PLAN_v2.md` and in the folder names
under `results_v2_discarded*/` (archived with the large folders below).

Each run writes `results_v2/<tag>.json` (metrics and selection fields) and
`results_v2/errors/<tag>.npz` (per-window losses, used by the tests); the probabilistic runs
write `results_v2/prob/`. The JSON records and the corpus are in this repository; the large
folders, `results_v2/errors/` (4.4 GB), `results_v2/prob/` (4.5 GB) and `results_v2_discarded*/`
(2.7 GB), are archived at ZENODO_DOI and are needed only for the significance tests and the probabilistic tables.

### 3. Numbers, tables, figures, manuscript

```
python src/paper_numbers.py --results results_v2       # paper/numbers.tex (MCS bootstrap, B = 2000)
python src/make_tables_v2.py                           # paper/tables_v2/*.tex
python src/fig_results_v2.py                           # figures_v2/*.pdf
bash   paper/build_v2.sh  --release                    # IEEE two-column layout  (paper/main_v2.pdf)
bash   paper/build_els.sh --release                    # Elsevier layout          (paper/main_els.pdf)
```

`paper_numbers.py --check --sec sec_v2 --main main_v2.tex` lists any result-like literal typed
by hand and any `\nv{}` key the generator did not emit; the release builds refuse both, as well
as an undefined reference, an overfull line, a float too large for the page or a Type-3 font.
The record selection (regime, width, lookback and learning rate on validation loss, seed
averaging) lives in `paper_numbers.py` and is imported by the table and figure scripts, so the
three cannot disagree.

## Checks a reviewer can run

* `python src/paper_numbers.py --check --sec paper/sec_v2 --main paper/main_v2.tex` — no
  hand-typed numbers in the prose.
* `python src/audit_data_v2.py` — corpus integrity (ranges, gaps, time zones, holiday
  calendars, the documented anomalies).
* `python -m pytest tests/` — the DM, Holm and MCS implementations against known cases.
* `python src/run_all_v2.py main --dry` — the job matrix of any stage without running it.
* `PLAN_v2.md` — every protocol decision and correction of the study, dated, with the records
  it affected.
