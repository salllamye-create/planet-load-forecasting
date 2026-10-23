# PLAN v2 -- PLANET smart-grid load-forecasting paper

Author: Abdulwahed Salam (independent researcher). Prepared 2026-09-27.
v1 is frozen at `../smartgrid-paper-v1/` and git tag `v1.0`. v2 is developed in this directory.

Goal: a manuscript that a demanding reviewer at IEEE Transactions on Smart Grid (primary target)
cannot reject on protocol, fairness, statistics, or reproducibility grounds, and whose
contribution is stated by mechanism rather than by leaderboard position.

---

## 1. Principles (fixed before any v2 experiment runs)

P1. **Identical information sets.** Every model receives the same inputs: past load, past
    weather, and the same known-future covariates (calendar, public holidays, NWP-degraded
    weather). Models that cannot consume future covariates natively get one shared exogenous
    head; PLANET uses that same head. Differences in the tables are then architectural.
P2. **Baseline list is pre-registered here and never edited on the basis of results.** No
    method is removed because it scores well. Dropping N-BEATS or CalTST after they beat PLANET
    would be spotted immediately (N-BEATS is the field's standard strong baseline, it is in the
    v1 code, and the code is released) and is research misconduct.
P3. **Every headline cell = mean +/- sd over 3 seeds.** Single-seed numbers appear nowhere in
    the main text.
P4. **All model selection on validation only** (lr, size, lookback, period, CQR calibration).
P5. **One source of truth for every number.** `report.py` emits `paper/numbers.tex` as LaTeX
    macros; the text never contains a hand-typed result. (v1 defect: ablation and DM counts in
    the prose disagree with the regenerated tables.)
P6. **Equal tuning budget per model** (same grid size, same epochs, same early stopping).
P7. **Failures are reported.** Any v2 hypothesis that does not hold is stated in the abstract.

## 2. Auditor findings on v1 (each becomes a v2 task)

| # | Finding | Severity | v2 fix |
|---|---------|----------|--------|
| A1 | Prose numbers stale vs tables (ablation %, DM 146/163 vs 169/168) | Reject-level if caught | P5 macros |
| A2 | Information sets differ across baselines under MS: iTransformer/Informer/Autoformer see weather, channel-independent models do not; only CalTST/PLANET/Informer/Autoformer see the calendar | Major | P1 shared exogenous head |
| A3 | Capacity confound: N-BEATS 3.55M params vs PLANET 0.93M; tuning = lr only | Major | P6 size sweep |
| A4 | 4 of 8 PLANET components hurt (own ablation) | Major (weak model) | Lean PLANET v2 |
| A5 | Seeds only on DE H=24/96 subset | Major | P3 |
| A6 | L=336 admitted sub-optimal for all | Minor | L selected on validation from {168,336,512,720} per model |
| A7 | ~400 DM tests, no multiple-comparison control | Major (IJF-style reviewer) | Holm correction + Model Confidence Set |
| A8 | Probabilistic eval on 4 models, one horizon | Minor | All headline models, H=24 + day-ahead |
| A9 | PLANET-X uses weather no baseline gets | Major | P1 |
| A10 | GB at H=336 collapse unexplained | Minor | Local-civil-time basis ablation; DST/timestamp audit |
| A11 | Year-end holidays cost 2x the COVID shock; no holiday feature anywhere | Major (domain) | Holiday covariate for all + holiday day-type in the basis |
| A12 | Closed-form solve ties CalTST -- solver contribution unsupported | Major (contribution) | Mechanism experiments where a per-window solve should matter: low-data, cross-grid transfer, temporal drift |
| A13 | Backbone-transfer experiment written, never run | -- | Pilot done (Section 2b): tier transfers; backbone becomes a validation choice |
| A14 | README says grid stride x2, code uses 5 | Trivial | Fix docs |
| A15 | 2026 reviewers will ask for TimeXer/TiDE/TFT, KAN/xPatch-era LTSF models, and zero-shot foundation models | Major | Section 5 |
| A16 | NWP-noise rule expressed wind/solar generation error in absolute MW calibrated on DE (1400/900 MW): 7.5 sd of noise on Belgian wind, 1.4 sd on GB; v1 PLANET-X GB regime numbers affected | Major (found by the FM builder, build day 1) | Capacity-relative rule for generation channels (`data_v2.nwp_noise_phys`), applied to every v2 covariate run; all R2 results produced before the fix discarded |

## 2b. Pilot evidence (2026-09-27, Germany, L=336, one seed, MS setting)

| H | Plain N-BEATS | v1 PLANET (patch) | Tier + N-BEATS stack, v1 basis | Tier + N-BEATS stack, lean basis |
|---|---|---|---|---|
| 24 h | 0.0291 / 2.10 % | 0.0373 / 2.37 % | 0.0326 / 2.18 % | 0.0331 / 2.18 % |
| 96 h | 0.0638 / 3.00 % | 0.0700 / 3.26 % | **0.0591 / 3.02 %** | 0.0647 / 3.08 % |
| 336 h | 0.1080 / 3.89 % | **0.0916 / 3.58 %** | 0.1015 / 3.84 % | 0.1074 / 3.91 % |

Cells are MSE (z-scored) / WAPE. Result tags `_v2pilot-basis`, `_v2pilot-lean-basis` in `results/`.
Reading: (i) the phase-locked tier transfers to the strongest baseline's backbone and improves
it by 7.4% MSE at 96 h and 6.0% at 336 h, and narrows the day-ahead gap to N-BEATS from 12.9%
to 3.8% WAPE; (ii) at 336 h the patch backbone is still better; (iii) the "lean basis"
inferred from v1's single-removal ablations does not hold when the removals are combined.
Seed sd for these models at 96 h in v1 was 0.04-0.08 WAPE, so (iii) is a warning, not a result.

## 3. Task definition v2

* **Headline task:** national system load forecasting, operational protocol.
  Systems: DE, FR, GB (v1) + ES, IT, NL (OPSD, subject to >= 90% coverage of load and TSO
  forecast; ERA5 via Open-Meteo, same preprocessing). Test split unchanged (chronological
  70/10/20; test = 2019-06 to 2020-09, contains year-end 2019 and COVID 2020).
* **Horizons:** 24 h, 96 h, 336 h (windowed), plus the day-ahead operator protocol (cut-off
  11:00 UTC on D-1, scored 00-23 UTC on D, leads 13-36 h) against the TSO forecast.
* **Information regimes (all models, all regimes):**
  R0 past load only; R1 + calendar + holidays (known future); R2 + NWP-degraded weather
  (reanalysis with lead-time-growing noise matched to operational skill, as in v1).
  R2 is the headline; R0/R1 quantify what each information source is worth per architecture.
* **Secondary:** Electricity-321 (M setting, as in the literature) kept in the main paper;
  ETTh/ETTm moved to an appendix as the negative-transfer boundary (not load data).
* **Metrics:** WAPE (of record), MAE/RMSE in MW, peak-height %, peak-timing h, ramp MAE;
  CRPS, PICP, MPIW, Winkler for probabilistic; parameters, train and inference time.

## 4. PLANET v2 (the model)

M1. **Basis composition decided by a proper ablation, not assumed.** v1's single-removal
    ablations said the climatological tier, local trend, Sat/Sun interactions and learned LS
    weights each hurt by 2-10%; the pilot (Section 2b) shows removing all four at once does
    *not* help on the N-BEATS backbone (slightly worse at every horizon, within seed noise).
    The removals do not compose. v2 runs a backward-elimination ablation, 3 seeds, on each
    backbone, and the paper reports the selected composition with its evidence. Kept
    regardless: absolute-time daily (6) + weekly (3) harmonics, the per-channel
    multiplicative ridge, and the seasonal-state conditioning (+20-28% at long H in v1).
M2. **Holiday day-type** -- public holidays (country-specific, `holidays` package) enter the
    basis as a day-type mask interacted with the first daily harmonics (holiday ~ Sunday), and
    as a covariate for every model.
M3. **Local civil time** -- the basis is phase-locked to local clock time, not UTC (human
    schedules follow the local clock; the DST shift is a 1 h phase error twice a year).
    Ablation: UTC vs local.
M4. **Backbone** -- residual network chosen on validation, per (system, horizon), between the
    patch Transformer and the doubly-residual MLP stack (`resid="basis"`, capacity-matched to
    N-BEATS at 3.4M parameters). Pilot (Section 2b): the tier transfers to the N-BEATS stack
    and beats plain N-BEATS at 96 h and 336 h; the patch backbone remains stronger at 336 h.
    So the story is "the tier improves whichever backbone is strongest", with the backbone
    a validation choice. Sub-task M4b: the basis stack conditions on the seasonal state only
    through a 0.01-initialised projection added to its input; replace with FiLM or per-block
    concatenation, since the state token is worth +28% at 336 h on the patch backbone and the
    basis backbone's 336 h weakness is the likely consequence.
M5. **Shared exogenous head** -- the v1 `WeatherResponse` head (point-wise response, thermal
    context, calendar-gated, short temporal conv) generalised to any covariate block; the same
    module is bolted onto every baseline that lacks native covariate inputs.
M6. **Calendar-aligned patching (ablation only)** -- patch boundaries aligned to midnight
    (patch = 24, stride = 24 or 12) vs the literature's 16/8.
M7. Everything else unchanged: per-window standardisation, MSE loss, Adam + cosine, early
    stopping on validation.

## 5. Baselines v2 (pre-registered; all under P1 with the shared head unless native)

Kept from v1 (15 + GBRT): SeasonalNaive, NLinear, DLinear, RLinear, FITS, N-BEATS, LSTM, TCN,
Informer, Autoformer, TimeMixer, iTransformer, PatchTST, PatchTST+Cal (CalTST), GBRT.
Added, native future-covariate models: TFT (Lim 2021), TiDE (Das 2023), TimeXer (Wang 2024),
N-HiTS / N-BEATSx (Challu 2023, Olivares 2023), xPatch (AAAI 2025), TimesNet (ICLR 2023).
Added, zero-shot foundation models (inference only, with the training-corpus contamination
caveat stated): Chronos-2 (covariates), TimesFM-2.5 (+XReg), Moirai-2.
Added, domain statistical benchmark: Tao Hong's vanilla GEFCom recency model (temperature
polynomials x calendar), the standard reference in the load-forecasting literature.
GBRT re-run with exactly the R1/R2 feature sets.

## 6. Fair tuning protocol

* Grid per (system, model): lr in {1e-4, 5e-4, 1e-3} x size in {small, published, large}
  (width/d_model and depth scaled so the three points bracket about 0.3x, 1x, 3x the
  published parameter count). 9 configs, 1 seed, H = 96, selected on validation; then lr
  re-checked per horizon (as v1). Same 10 epochs / patience 3 for all.
* Lookback: L in {168, 336, 512, 720} chosen on validation per (system, model, H).
* 3 seeds (2021, 2022, 2023) on every reported cell.

## 7. Mechanism experiments (turn v1's "tie" and "did not work" into evidence)

E1. **Learning curves** -- train on 10 / 25 / 50 / 100 % of the training split; hypothesis:
    the per-window closed-form fit degrades least in the low-data regime (CalTST must learn
    the calendar mapping; PLANET solves it).
E2. **Cross-grid zero-shot transfer** -- train on one system, test on another with no
    fine-tuning (load standardised); hypothesis: phase-locked coefficients re-estimated per
    window transfer, learned calendar embeddings do not.
E3. **Temporal drift** -- train on 2015-2017 only, test across 2018-2020 without retraining;
    error vs months-since-training for each method.
E4. **Regime shift** -- year-end 2019 and COVID 2020 excess WAPE (v1 design), all models.
E5. **Phase-lock ablation** -- absolute vs relative time; local vs UTC; with/without holidays.
E6. **Backbone transfer** -- tier on/off x {patch, basis} backbone (pilot first).
E7. **Horizon curve** -- error vs lead time 1..336 h for the top 6 models (v1 figure extended).

## 8. Statistics

* Diebold-Mariano on per-window absolute-error differentials, HAC variance, HLN correction
  (v1), with Holm step-down across the comparisons in each table.
* Model Confidence Set (Hansen, Lunde & Nason 2011) at alpha = 0.10 per (system, horizon):
  the set of models not significantly worse than the best. Report membership, not only rank.
* Seed-level 95% CIs in every headline table; effect sizes for each ablation.

## 9. Probabilistic forecasting

All headline models (not 4), H = 24 windowed and the day-ahead operator protocol, 7 quantiles,
CQR on validation, ACI online on test (v1 recipe). Report CRPS, PICP (raw / CQR / ACI), MPIW,
Winkler. Add a reliability diagram.

## 10. Data work

D1. OPSD extraction for ES, IT, NL (+ any other country with load actual + TSO forecast and
    >= 90% coverage over 2015-2020); ERA5 hourly via Open-Meteo for each capital region as in v1.
D2. Public holidays per country (`holidays`), plus a "bridge day" and "Christmas to New Year
    period" indicator.
D3. Local civil time per country (`zoneinfo`) alongside the UTC stamps.
D4. Timestamp/DST audit of every series (v1 GB anomaly).
D5. Leakage audit documented in the paper: calendar/holidays are known future by definition;
    weather only through the NWP-degraded regime; no test-set selection anywhere.

## 11. Compute and schedule

Measured v1 cost: about 1 min per grid run on the local Quadro RTX 4000 (thermally throttled),
about 10 min per Electricity run. v2 estimate (6 systems x 3 H x ~24 trained models):

| Stage | Runs | GPU-h (local) |
|---|---|---|
| Tuning grid (9 configs, H=96) | ~1300 | ~28 |
| Main, 3 seeds, 3 H, regime R2 | ~1300 | ~28 |
| R0/R1 information regimes (1 seed) | ~430 | ~9 |
| Ablations + mechanism E1-E7 | ~450 | ~9 |
| Probabilistic | ~300 | ~10 |
| Electricity (15 models x 2 H x 3 seeds) | ~90 | ~15 |
| Foundation models (inference) | -- | ~2 |
| **Total** | **~3900** | **~100 GPU-h: about 5 days serial locally, about 1 day on a rented A100/4090 (~$60-120)** |

Calendar (local GPU, serial):
* Week 1 -- code: shared head, holidays, local time, lean PLANET, PLANET-B, new baselines,
  foundation wrappers, MCS/Holm, `numbers.tex` macros; data D1-D5; pilot + smoke tests.
  GPU starts on the tuning grid as soon as the harness is stable (day 3).
* Week 2 -- main grid, regimes, ablations, mechanism, probabilistic, Electricity run
  continuously; figures and tables regenerate from JSON as results land.
* Week 3 -- manuscript rewrite; three simulated adversarial reviews (methodology, statistics,
  power-systems practitioner) run as independent agents against the PDF; fix; final build;
  reproducibility package (code + processed data + result JSONs + seeds).

## 12. Manuscript v2

* Working title: "Calendar-Locked Load Forecasting: Phase-Locked Seasonal Bases with
  Closed-Form Test-Time Estimation for National Grids".
* Contributions (three, each tied to an experiment that could have falsified it):
  (1) phase-locked calendar representation and where it pays (E5/E7);
  (2) closed-form per-window estimation and where *it* pays over a learned calendar map
      (E1/E2/E3) -- the honest replacement for v1's tie;
  (3) an operational, information-set-controlled benchmark on six national systems with
      operator forecasts, holidays, NWP-skill weather, MCS statistics, and calibrated
      intervals -- reusable by the community.
* Structure: Intro; Related work (add TSFMs, exogenous LTSF, load-forecasting practice);
  Problem and information sets; Method; Experimental protocol; Results (grids R2; regimes;
  day-ahead vs TSO; mechanism; probabilistic; Electricity); Ablations; Limitations (ETT
  appendix, contamination caveat for TSFMs, weather-regime assumptions); Conclusion.
* Target: IEEE Transactions on Smart Grid (IEEEtran, already in use). Alternates: Applied
  Energy; International Journal of Forecasting (if the statistical angle dominates).

## 13. Decisions for the author

1. Journal target (default: IEEE TSG).
2. Add ES/IT/NL national systems (default: yes, if coverage checks pass).
3. Include zero-shot foundation-model baselines (default: yes; ~3 GB downloads, inference only).
4. Compute: local GPU only (~5 days of runs) or rent a cloud GPU for the main grid (~1 day).
5. Keep MSE as the training loss of record for all models (default: yes, literature standard).

## 14. Definition of done

- [ ] Every headline number is a macro generated from result JSONs; grep finds no hard-coded result in `paper/sec/`.
- [ ] Every table cell has 3 seeds; every claim of superiority has a DM p-value with Holm correction or MCS membership.
- [ ] Information sets identical across all models in every table (P1), stated in the protocol section.
- [ ] Baseline list in the paper equals the list in Section 5 of this plan.
- [ ] Three independent adversarial reviews return no reject-level finding.
- [ ] Reproducibility package builds the PDF from raw downloads in one command.

## 15. Decision log

2026-09-27 (build day 1)
- Journal: IEEE Transactions on Smart Grid primary; manuscript written to move to Applied Energy or IJF with a format change only (author: any impact-factor journal, my choice).
- Systems: eight (DE, FR, GB, ES, IT, NL, PL, BE). OPSD covers all with 99.8-100% load and operator-forecast coverage. NL solar and IT wind dropped by a uniform rule (any raw gap > 720 h); PL has no usable solar. GB holidays use the England subdivision (Easter Monday, late-August bank holiday).
- ECL and ETT raw timestamps are local civil time, so local_stamps == stamps there; the UTC-vs-local ablation is a grid-only experiment.
- Foundation models: Chronos-2 and TimesFM-2.0-500m. Moirai-2 dropped: its package would downgrade pandas/scipy/einops in the shared environment. TimesFM-2.5 not used for the same reason (package pin).
- Baselines wrap the official neuralforecast implementations (TFT, TiDE, N-HiTS, N-BEATSx, TimesNet, TimeXer, TSMixerx, XLinear); xPatch and KAN are not in the library and are not added. TimeXer's native interface takes only the past covariates; the future part reaches it through the shared head, as for TimesNet.
- Tuning budget: size in {0.5x, 1x, 2x} selected on DE and FR and transferred; learning rate in {1e-4, 5e-4, 1e-3} per system. TFT and TimeXer (10-20x the cost of a PatchTST run) are tuned at published size with the learning-rate sweep on DE only; the paper states this.
- Under the MS setting, channel-independent models are trained on the load channel alone (identical target forecast, one-seventh of the compute); multivariate models keep all channels.
- Probabilistic study: v1-signature models, PLANET, and the univariate library models with a native multi-quantile head (TFT, TiDE, N-HiTS, N-BEATSx, TimesNet).
- Early zero-shot evidence: Chronos-2 without covariates reaches 2.28% WAPE on DE day-ahead (v1 PLANET 2.37%, N-BEATS 2.10%); it is reported with the pre-training-contamination caveat and raises the bar for every learned model.
- NWP noise for wind/solar generation is capacity-relative (8% / 5% of the training 99.9th percentile at 24 h); the absolute-MW rule inherited from v1 was wrong for every system but Germany. Twelve tuning results and all GBRT/foundation-model R2 cells produced before the fix are discarded and re-run.

- **2026-09-28 (audit, evening).** Wind-speed unit error in the NWP-error rule: the corpus stores
  Open-Meteo `wind_speed_10m` in km/h, and `NWP_ABS_24H["wspd"] = 1.1` was applied in the column's
  unit (1.1 km/h ~ 0.3 m/s, 3.6x below the intended 1.1 m/s). Fixed at the single source
  (`data_v2.WSPD_MS_24H * 3.6`; the fallback copy in `fm_zero_shot.py` matched). Only the degraded
  future wind-speed rows of regime R2 are affected, but every R2 record produced before the fix
  (1,193 neural tune/lookback/headline runs, 216 GBRT/Tao runs, dayahead.json) was moved to
  `results_v2_discarded_wspd/` (git-ignored) and the whole R2 protocol re-run, tuning included,
  so that no reported selection or result depends on the wrong constant. R0/R1 results and the
  R0 zero-shot runs are unaffected and kept. Also corrected in the same pass: the ERA5 point is
  one interior point per system near its geographic centre (not population-weighted); Holm
  adjustment is within each system (the code default), not "within each table"; seed counts
  per experiment are now stated explicitly (three for headline/day-ahead/ablation, one for
  R0/R1, mechanism, probabilistic and ETT); generation-series drop rule stated as both rules
  (30-day internal gap, <50% presence).

- **2026-09-30 08:10 (protocol change, validation-motivated).** The lookback for the headline
  models was selected on DE only and transferred. On FR the transferred lookbacks were clearly
  not validation-optimal: at H=96 the FR validation MSE at the DE choice vs at L=336 was 0.0332
  vs 0.0242 (N-BEATS, L=720), 0.0299 vs 0.0258 (PatchTST, L=512) and 0.0322 vs 0.0272 (PLANET,
  L=512); N-HiTS and DLinear were fine. The transfer rule therefore distorted the cross-system
  comparison in a model-dependent way. Replaced by per-system, per-horizon selection on
  validation for all nine headline models (the same rule the learning rate already follows);
  `pipeline_v2.sh` runs the lookback sweep on all eight grids before the headline stage (+756
  single-seed runs, ~25 GPU-hours). Headline runs already made at the transferred lookback are
  kept as records; `paper_numbers.select_headline` chooses among records by validation loss, so
  they never enter a table unless validation prefers them. Observation made on validation
  losses; test numbers were visible in the same listing, which is stated here for the record.

- **2026-09-30 09:00 (protocol finalised; user directive: solidity over time, "even a month").**
  Every time-saving asymmetry removed before more GPU time is sunk: (1) the five expensive models
  (TFT, TimeXer, TimesNet, Informer, Autoformer) receive the full tuning budget -- width sweep on
  DE/FR and learning rate per system -- instead of published width with lr on DE only; (2) the
  learning-rate grid gains 3e-3, because 1e-3 (the former upper edge) won for almost every model;
  (3) the lookback is selected per system and horizon for EVERY model, not only the nine headline
  models; (4) every reported cell is a three-seed mean: R0/R1, mechanism, probabilistic, ECL and
  ETT move from one seed to three; (5) a budget check (headline models on DE at 30 epochs,
  patience 5, three seeds) runs right after the DE lookback stage so that, if the ordering depends
  on the ten-epoch cap, the budget can be raised before the large stages run. The driver moves
  records made under a superseded selection (lr, width, backbone or lookback) to results_v2/stale/
  before each stage, so no stale record can enter a table. Estimated cost ~11 GPU-days at 10
  epochs; the 207 DE headline runs are kept where their selection is unchanged. This is the last
  protocol change; anything further is a bug fix or a reported deviation.

- **2026-09-30 19:10 (reported deviation from "no further changes": tuning grid extended).** With
  3e-3 added, the upper edge of the learning-rate grid again won on DE for 8 of the 16 models
  tuned so far (TCN, Informer, Autoformer, TimeMixer, iTransformer, PatchTST, PLANET, ...). A grid
  whose boundary wins is a standard reviewer objection, so one more decade, 1e-2, is added for
  every model on every system (~250 single-seed runs, ~10 GPU-hours). If 1e-2 wins for a model
  the grid is not extended again (Adam at 1e-2 is at the edge of stability for these
  architectures) and the paper states it. Selection rule unchanged; the running reference-system
  sweep was restarted so that width and learning rate are chosen over the full grid.

- **2026-10-01 04:40 (reported deviation: training budget raised before the large stages).** With
  the learning-rate grid extended to 1e-2, eleven of the nineteen models tuned so far on DE chose
  the new upper edge (NLinear, RLinear, FITS, N-BEATS, LSTM, Informer, Autoformer, TimeMixer,
  iTransformer, TFT, TiDE): under a binding ten-epoch cap the fastest-converging configuration
  wins, which is the under-training objection in its clearest form, independent of whether the
  ranking would survive a longer budget. The protocol therefore changes to training to early
  stopping (patience 5) under a cap of 30 epochs for every stage (tuning included), and the
  separate budget check is dropped; the paper reports the mean epochs run and the share of runs
  that reached the cap. Every neural record made at the ten-epoch budget (tuning, lookback, the
  207 DE headline runs, partial FR) is moved to results_v2_discarded_ep10/ and regenerated; the
  classical and zero-shot records are unaffected. Estimated cost ~2.2x the ten-epoch grid
  (~24 GPU-days); results expected around 25 October. Decided one day before the planned budget
  check because the learning-rate evidence made its outcome foreseeable and every day of
  ten-epoch runs would have been discarded.

- **2026-10-01 08:10 (budget check reinstated as a robustness experiment).** Under the 30-epoch
  cap with cosine annealing, 45 of the first 76 tuning runs still ran to the cap (median 30
  epochs) and several models still chose the top learning rate: with a cosine schedule the
  validation loss keeps improving slowly until the schedule ends, so "reached the cap" is expected
  and is not by itself evidence of under-training. The right evidence is sensitivity: the nine
  headline models are re-trained on DE at 90 epochs / patience 10 / three seeds (tag -ep90) as the
  last pipeline stage, and the discussion reports the WAPE change and the rank correlation between
  the two budgets. This adds an experiment; it changes no selection and no protocol rule.

- **2026-10-02 11:00 (bug fix: library models were the only ones without per-window
  normalisation; TiDE collapsed by a library defect).** Under the 30-epoch budget the five
  univariate library models fell far behind the in-house models on DE, and TiDE gave the same
  validation loss (0.129) at three widths. Diagnosis on CPU with the real DE windows: (1) NF's
  TiDE applies LayerNorm to the one-dimensional output of its temporal decoder, which turns the
  decoder output into a constant for point losses -- with layernorm=False TiDE fits as well as
  N-BEATS in three epochs (val 0.032 vs 0.037; 0.145 with the default). Fixed by setting
  layernorm=False for TiDE, the only deviation from a library default, stated in the paper.
  (2) The wrapper calls the library models' forward() directly, bypassing the library's own
  per-window temporal scaler, so they were the only models without instance normalisation while
  every re-implementation carries RevIN and PLANET its own; SETUP promised "the same
  normalisation". Per-window standardisation of the lookback (inverted on the forecast) is now
  applied inside the wrapper for every library model. It is neutral for TiDE and N-HiTS on
  validation; it is expected to matter for TFT and TimesNet across the 2020 level shift, where
  their test error was two to three times their 2019 error. All library-model tuning records made
  before the fix (DE: TFT, TiDE, N-HiTS, N-BEATSx, TimesNet) were moved to
  results_v2_discarded_nfnorm/ and are regenerated (~16 GPU-hours, mostly TimesNet). Gradient
  clipping, the width scaling and the covariate path were checked and are not at fault.

- **2026-10-02 20:25 (machine bug-checked; operational changes only).** After six days at
  87-91 C and full load (card target 83 C, maximum operating 92 C) a TimesNet tuning run raised
  a cuDNN internal error, the NVIDIA driver logged a burst of 19 errors (event 153) within two
  minutes, and Windows bug-checked (0x139) and rebooted. No hardware-error (WHEA) events; every
  result file on disk is readable and paired with its error array; the last finished record is
  from 20:07. A GPU compute check after the reboot reproduces the pre-crash numbers exactly.
  Changes: (1) a thermal guard in the training loops pauses at 88 C until 82 C (pause time
  excluded from the reported training time); (2) TimesNet, whose runs hold the card at full load
  for one to three hours each, is scheduled in a deferred pass at the end of the pipeline under
  the identical protocol (`--exclude TimesNet` in the main pass), so every other result is
  produced first and the decision to keep or drop it can be taken with complete information;
  (3) a reproducibility audit (re-running a random sample of finished runs) is to be added as a
  final stage, since a fault of this kind raises the question of silent errors. Lowering the
  card's power limit to 100 W needs an elevated shell (`nvidia-smi -pl 100`, reset at each boot)
  and is requested from the user. No selection rule or protocol setting changed.

- **2026-10-06 05:00 (pre-registered variant: PLANET-X, native covariate pathway). Written before any
  result of the variant exists.** Evidence that motivates it: on DE (daily load-temperature
  correlation -0.34, winter/summer load ratio 1.09, weather block worth -8% WAPE to GBRT at H=96)
  PLANET leads all 22 models at every horizon in the lookback sweep; on FR (correlation -0.83,
  ratio 1.51, weather worth -17%) TiDE leads at every horizon and PLANET is 2nd/4th/3rd. PLANET's
  only covariate pathway is the shared exogenous head (~6k parameters, a correction added to the
  forecast), whereas TiDE/N-BEATSx ingest the whole future covariate block in their main network.
  Hypothesis: the deficit on weather-driven systems is the pathway, not the calendar tier.
  Design, fixed now: PLANET-X = PLANET (tier, instance normalisation, same backbones and
  conditioning) + a covariate encoder (per-step linear E->4 + GELU, flatten over L+H, linear to a
  128-d embedding) whose output is concatenated to every residual block's input (basis backbone)
  or prepended as one token (patch backbone); no shared head. Tuned, lookback-selected and seeded
  under the identical protocol, added to the headline set so it appears in every experiment next to
  PLANET; PLANET is kept unchanged as the primary model and the ablation stays on PLANET. The paper
  states that the variant was designed after the first two systems' sweeps. Whatever its results,
  both rows are reported.
  Implementation (06 Oct 05:40): planet.CovEncoder; BasisResidual/PatchEncoder take cov_dim;
  PLANET(cov_native=True, E) sets needs_cov/native_cov so the harness passes the degraded block
  and does not attach the shared head; model name PLANETX in MODEL_CFG/SIZE_KEYS/CHANNEL_INDEPENDENT,
  V1_MODELS, HEADLINE, mech/prob/ECL/ETT lists; width sweep covers both backbones as for PLANET.
  Parameter counts at half width, L=336/H=96: PLANET-basis 1.36M, PLANET-X-basis 1.72M (encoder
  0.22M + block inputs); patch 0.37M vs 0.60M. Validation check before any GPU run: see below.
  Validation check (CPU, 3 epochs, lr 1e-3, half width, basis backbone, perfect future
  covariates): val MSE DE PLANET 0.0351 vs PLANET-X 0.0439; FR 0.0334 vs 0.0354 (PLANET-X ahead
  at epoch 1 on FR, behind at epoch 2). The variant trains correctly; three epochs cannot settle
  whether the larger covariate pathway pays, which is what the full protocol decides. Added to
  the queue 06 Oct 06:10; its results are unknown at the time of this entry.

- **2026-10-06 08:30 (PLANET-X tuning-stage outcome: negative).** On both reference systems the
  variant is clearly worse than PLANET under the identical sweep (H=96, L=336, single seed, best
  configuration on validation): DE val 0.0283 / WAPE 2.54 vs 0.0175 / 2.18; FR val 0.0251 / 3.02
  vs 0.0203 / 2.70 (TiDE: 0.0192 / 2.23 and 0.0141 / 2.26). The hypothesis "PLANET's deficit on
  weather-driven systems is covariate capacity" is not supported by this design: a dense encoder
  of the whole covariate block, concatenated to every residual block, is worse than the small
  per-step shared head. In hindsight the design dropped the per-step (horizon-aligned) pathway
  that the head provides and TiDE also has in its temporal decoder. No further variant is
  designed: iterating on architectures until one closes the gap would undo the value of the
  pre-registration. PLANET-X stays in the grid as pre-registered (cheap) and is reported as a
  negative result; the discussion will say that the French deficit is not a matter of covariate
  capacity alone.

- **2026-10-13 ~13:40 (second unexpected shutdown).** The machine stopped during the day-ahead
  stage (last record 13:39) and came back at 14:35 (Kernel-Power 41: no clean shutdown). Unlike
  2 Oct there is no bugcheck record and no NVIDIA driver error in the System log, and the thermal
  guard was holding the card at 80-82 C right before: the cause is not visible in the logs
  (power or a hard hang). Every result file is readable; the pipeline was relaunched at 20:00 and
  resumes the stage from its 25 finished runs. The GPU power limit reset to 125 W at boot and is to
  be re-applied by the user. Six hours of GPU time lost; main pass now ~21 October.

- **2026-10-14 05:3x (third unexpected stop).** The machine rebooted at 05:33 during the last
  runs of the day-ahead stage (last record ~05:27); again no bugcheck, no driver error, thermal
  guard active. Results intact; pipeline relaunched 08:30. Three unplanned stops in twelve days
  under sustained load, two of them without any software trace: hardware (power supply, cooling
  or memory) is the likely cause and is reported to the user. Main pass ~22 October.
  Correction (08:35): the System log shows this morning's restart was initiated by Windows Update
  (MoUsoCoreWorker / TrustedInstaller, "Operating System: Upgrade", 05:29 and 05:33), not a
  fault. Yesterday's 13:40 stop remains unexplained (Kernel-Power 41, no update event). Windows
  Update's automatic restarts must be paused by the user for the remaining weeks of the grid.

- **2026-10-18 08:30 (mechanism stage, preliminary outcome at job 309/504; DE complete, FR
  nearly, ES pending).** Recorded before the discussion text is revised so that the revision is
  traceable. Scarce data (last 10/25/50 % of the training split, H=96, 3 seeds): PLANET has the
  lowest WAPE of the seven methods at 25 % and 50 % on both reference systems (DE 2.69/2.32, FR
  3.20/2.97), but not at 10 % (about five months), where the lightest architectures degrade
  least (DE PatchTST 3.06 vs PLANET 3.23; FR N-BEATS 4.02 vs PLANET 4.34) and TiDE degrades most
  (DE 4.19). The scarce-data claim therefore holds at moderate scarcity only and the discussion
  must say so. Temporal drift (first 10/25/50 %): every method degrades by a comparable amount;
  PLANET is best at the longest gap on DE (3.49 vs PatchTST 3.57) and second on FR (CalTST 4.70
  vs 4.77); the drift-specific penalty (drift minus scarce at equal data) is smallest for the
  calendar-tier models at 10 % (+7-8 % vs +17-19 %) but not at larger fractions. Transfer
  without retraining: the PLANET family is best in three of four completed pairs (PLANET-X
  DE->FR 4.48 vs 5.58 next; ES->FR 3.90 vs 4.87; PLANET FR->DE 4.42 vs TiDE 4.71; TiDE ES->DE 3.43
  vs PLANET-X 3.52), N-BEATS worst everywhere (+166 to +206 %); seed sd 0.1-0.6 so the orderings
  with gaps above 0.5 are reliable. PLANET-X, negative in-domain, is the best transfer model in
  two pairs: to be reported as observed, not explained.

- **2026-10-18 19:00 (probabilistic stage: defect in the quantile wrapper for PatchTST+Cal).**
  Stage 7 finished 167 of 192 runs. All 24 PatchTST+Cal runs failed at the first training step:
  the quantile wrapper built the model with H*Q outputs by widening H, which for this model also
  sets the number of future calendar tokens, so the positional embedding (83 tokens) no longer
  matched the design matrix (65). Fix: CalTST gains an out_mult argument that widens only the
  output head; the wrapper passes the true H and out_mult=Q. The point-forecast path is
  unchanged (out_mult=1 gives the identical module, parameter order and shapes), so no record is
  affected. One further run (DLinear, GB, seed 2023) left no record and reproduces cleanly on CPU
  (transient). The 25 runs are re-run by src/prob_followup.sh under the identical command line
  once the pipeline reaches the ETT stage; the stage filter in pipeline_v2.sh hid the tracebacks
  and will keep the full output in a log once the script is no longer executing. Reproduced on
  CPU first so the running GPU pipeline was not touched.

- **2026-10-18 19:05 (mechanism stage complete, final outcome; supersedes the 08:30 entry).**
  Scarce data (H=96, 3 seeds): PLANET has the lowest WAPE in four of nine cells (DE 25 %, DE
  50 %, ES 10 %, ES 25 %), TiDE in four (FR at every fraction, ES 50 %), PatchTST in one (DE
  10 %). At 10 % PLANET degrades far less than TiDE on DE (+48 vs +86 %) and ES (+35 vs +86 %)
  and equally on FR (+56 vs +53 %). Drift (oldest 10 %): PLANET best on DE (3.49) and ES (3.29),
  second on FR behind PatchTST+Cal (4.77 vs 4.70); TiDE degrades most everywhere (+89, +116,
  +127 %); at 50 % the full-data ordering returns. Transfer (six pairs): mean penalty over the
  target's own model PLANET-X 55 %, PLANET 70 %, TiDE 74 %, PatchTST+Cal 92 %, PatchTST 112 %,
  TFT 135 %, N-BEATS 138 %; the PLANET family is best in three pairs and ties one, TiDE best in
  two and ties one. Controlled comparison solve vs learned calendar map (PLANET vs PatchTST+Cal,
  same inputs): PLANET better in 5/6 transfer pairs and in the DE and ES scarce and drift cells,
  equal or slightly worse on FR. The discussion will state: the mechanism pays on the two
  calendar-dominated systems and in transfer, not on the weather-dominated one; PLANET-X's
  transfer result is reported as observed.

- **2026-10-19 19:50 (budget-check stage never ran: argument-syntax defect).** The main pass
  reached stage 10 at 13:04 (ETT 480/480, foundation models 96/96 under R2, all stages before
  them complete) but the budget-check command passed "--suffix -ep90", which the argument parser
  reads as a second option flag; the stage exited with code 2 three times, was abandoned by
  design, and the script went on to the deferred TimesNet pass. Found at 19:38 on the next status
  check (TimesNet tuning at job 3/22, one run of 48 min lost). Fix: "--suffix=-ep90" in
  pipeline_v2.sh; dry run lists the expected 90 jobs. The pipeline was stopped (all processes
  verified gone) and resumed from stage 10 by src/pipeline_v2_resume.sh, which runs the budget
  check first (about one day) and then the TimesNet pass unchanged; the two finished TimesNet
  tuning runs are kept. No result record is affected. Main pass now complete ~21 October.

- **2026-10-19 20:35 (verification pass, first round: prose rewritten against the final
  records).** numbers.tex, every table and every figure regenerated from the complete result
  set (budget check and TimesNet still running). Three generator defects fixed on the way: the
  benchmark table read the headline regime (R2) and found no ECL/ETT records (they are R1); the
  grid table glued the MCS dagger to the Holm letter into an undefined command; the day-ahead
  block had no count over the unbiased systems (added, tolerance 3 %, GB and NL excluded).
  Results, discussion, conclusion, the contribution list and the abstract (198 words) were
  rewritten to what the records say: TiDE is the most accurate point and probabilistic
  forecaster (19/24 cells, MCS 24/24, mean rank 1.29); PLANET is second (rank 3.79, MCS 16/24),
  significantly behind TiDE in 12 cells and ahead of every other method; PLANET leads on DE at
  every horizon and is within 3 % on FR at one day and on GB; the deficit opens beyond one day
  and sits on the weather-driven systems; the tier is worth 6 % WAPE on average over the DE/FR
  ablation cells, 11-15 % on DE at 1-4 days, ~0 on FR at two weeks; calendar-locking is worth
  1 % on average (4 % on DE at four days) and UTC vs local time nothing measurable; optional
  components earn nothing; the mechanism holds in transfer and at moderate scarcity on the
  calendar-dominated systems; PLANET-X failed its pre-registered test (-2 % on FR, +11 % on DE
  relative to PLANET); operators beaten by PLANET on 3/6 unbiased systems, by TiDE on 4/6;
  Chronos-2 with a 1024-h context beats every trained model on 3/8 systems at one day. The
  ablation table and prose now use WAPE (the figure already did). Every "VERIFY" marker is
  resolved except the training-budget paragraph, which waits for the stage (its first four
  cells show +4 to +9 % WAPE with the 3x budget; early stopping does restore the best-validation
  weights, so this is a schedule effect, to be described when complete). Draft builds at 22
  pages, no undefined references; three budget macros still undefined until the stage ends.

- **2026-10-19 22:30 (verification pass, second round: rendering defects).** Proof-reading the
  rebuilt PDF found: relative-change macros printed without a percent sign (fpct now appends
  \%), negative numbers in text set with a hyphen (fnum wraps them in ensuremath), the MCS
  figure and the regimes figure included the day-ahead horizon (32 cells; restricted to the
  three grid horizons, denominators corrected), the MCS caption described a p-value where the
  cell shows the WAPE rank, the mechanism-figure caption listed its panels in the wrong order
  and its transfer panel had overlapping labels, the regimes figure was unreadable at column
  width (now text width, Hong's regression dropped from it because it cannot run R0), and the
  day-ahead table printed the operator bias in MW while the prose quotes percent (table now in
  percent). Numbers, tables and figures regenerated; draft builds at 23 pages with every macro
  defined and no hand-typed result. Three simulated adversarial reviews (statistician,
  power-systems practitioner, deep-learning referee) launched on this draft.

- **2026-10-19 21:30 (three simulated referee reviews; consolidated revision plan).** Three
  independent adversarial reviews (statistician, power-systems practitioner, deep-learning
  referee; scratchpad/review_*.md, to be archived under paper/reviews_v2/) all recommend major
  revision. Findings verified by me against code and data before being accepted:
  (1) Informer, Autoformer and iTransformer re-implementations carry no instance normalisation
  (SETUP says they do) -> add RevIN and re-run them under the full protocol. (2) Adaptive
  conformal update uses each window's full-horizon outcome one hour later (23 h look-ahead at
  H=24) -> delayed-feedback ACI, conformalise all seven levels, save calibration arrays, re-run
  the probabilistic stage; rename the score (quantile-weighted CRPS approximation). (3) Exactly
  two seeds diverged (val MSE >10x the other seeds: Autoformer DE H24 s2023, WAPE 33; LSTM ES
  H336 s2023, non-finite) -> exclusion rule stated as val MSE non-finite or >10x the median of
  the other seeds, applied uniformly. (4) ECL/ETT inherit DE-tuned lr/width; R0/R1 reuse
  R2-tuned settings -> disclose; re-select lr per dataset (ETT/ECL) and check lr re-selection
  per regime on DE. (5) NL load steps +17 % at 2016-01 (11.8 -> 13.8 GW, ENTSO-E perimeter);
  FR test target has one corrupt hour (158 GW at 2020-07-07 17:00); GB has 92 isolated spikes
  (the reviewer's "2,228 jumps" are mostly the 05-07 h morning ramp, also on IT/NL) -> document
  all three; exclude the windows covering the FR hour from FR scoring; soften GB claims.
  (6) Hong's regression carries a linear trend in hours since 2015 fitted once / on an expanding
  window and extrapolated across the NL step -> add rolling-window (24/36-month) and no-trend
  variants selected on validation (CPU). (7) DM variance: Bartlett at lag h-1 under-estimates
  the long-run variance of overlapping-window losses -> stationary-bootstrap DM consistent with
  the MCS, report effective n, set sizes, per-seed agreement, sensitivity counts. (8) Day-ahead
  comparison: add the 08:00 UTC origin (publication deadline), DM tests vs the operator on daily
  errors, seed-ensemble numbers, monthly / lockdown-excluded / holiday-day tallies, state the
  H=48 training and the post-hoc 3 % tolerance. (9) Transfer and short-history claims hold only
  against PatchTST+Cal (controlled pair), not against TiDE in absolute WAPE -> restate; the
  "calendar-dominated" mechanism does not track weather dependence across the eight systems
  (NL as calendar-driven as DE, TiDE wins there by 10 %) -> per-system table and honest
  statement. (10) The title claim (phase-locked at local civil time) is worth ~1 %/0 % and is
  largely a rotation of the per-window coefficients -> reframe title/abstract/contribution 1
  around the per-window closed-form calendar state + holiday masks + residual backbone; cite
  DeepTime, CycleNet, FAN/SAN/Dish-TS, ES-RNN, Prophet/dynamic harmonic regression, FSNet/OneNet;
  load-forecasting literature (Taylor & McSharry 2007, Lago 2021, Haben 2021, Obst/de Vilmarest
  & Goude, DeepAR, MQ-RNN, Hong Wang White 2015, GEFCom2017). (11) Weather proxy: white noise is
  kinder, not harsher (daily-mean error 0.25 K) -> fix text, rename "forecast skill" to
  "degraded reanalysis", add an AR(1)-correlated sensitivity (TiDE, PLANET; DE/FR/ES) and a
  population-weighted multi-point weather sensitivity (FR/ES/IT). (12) Covariate-pathway
  confound -> decisive experiment: TiDE fed through the shared head instead of its native
  pathway. (13) Four tables clipped in the PDF (Float too large) -> split; metric definitions
  paragraph; selected-configuration appendix table (backbone/width/lr/lookback, grid-edge
  fraction); counts (26 methods = 23 neural + naive + GBRT + Hong x2); library versions and
  determinism; TimesFM covariate port flagged (R0 as its headline); Fig. 2 y-label; sign
  convention unified; closed-form-alone row given its paragraph; budget check extended to FR.
  GPU order (TimesNet moved to the end; user to confirm): TiDE-through-head (~2 h), prob re-run
  with fixed ACI + TFT (~10 h), correlated-noise sensitivity (~2 h), lr-per-regime check on DE
  (~2 h), extra ablation seeds for the key variants (~5 h), Informer/Autoformer/iTransformer
  with RevIN under the full protocol (~4-5 days), multi-point weather (~2 h + data), monthly
  refit of day-ahead models (~6 h), ETT/ECL lr tuning + re-run (~1 day), budget check on FR
  (~6 h), then TimesNet (~7-14 days). The resume wrapper was stopped at 21:15 so that nothing
  starts after the budget stage; the stage's own process continues.

- **2026-10-19 22:50 (revision work started; GPU order confirmed by the user: revision stages first,
  TimesNet last).** Correction to the 21:30 entry, item (1): iTransformer normalises its inputs
  inline (verified), so only Informer and Autoformer lacked instance normalisation; the reviewer's
  "2,228 spurious jumps" on GB are mostly the 05-07 h morning ramp and are rejected as a defect
  (92 isolated spikes remain, documented). Code changes, all CPU-verified before use:
  (a) RevIN added to Informer and Autoformer (baselines.py); their 456 records (tuning, lookback,
  headline) will be moved to results_v2_discarded/norevin/ by the pipeline before the re-run.
  (b) Probabilistic calibration (prob.py, prob_v2.py): CQR now conformalises every symmetric level
  pair (90/80/50 %), the adaptive update is the delayed-feedback form (the level of step k is
  updated with the outcome of the most recent window whose step k is observed at the origin), the
  look-ahead form is kept as a diagnostic field (_aci0), corrupt target hours are masked, the
  calibration arrays and level trajectories are saved, and the score is named the quantile-weighted
  CRPS approximation; the 192 look-ahead records go to results_v2_discarded/prob_lookahead/ and the
  stage is re-run with TFT added (216 runs).
  (c) Corrupt hours: data_v2.BAD_HOURS = {FR: 2020-07-07 17:00 UTC}; the analysis drops the
  windows covering it from every FR point score (24 of ~10,000 windows at one day, 336 at two
  weeks; WAPE and MSE recomputed from the saved arrays, unmasked values kept), the probabilistic
  scores mask the hour, the day-ahead protocol drops the day.
  (d) Pathway control: TiDEh = TiDE with the library's covariate pathway switched off and the
  block routed through the shared head, run at TiDE's selected rate, width and lookback
  (run_all_v2.TUNE_ALIAS) so that only the pathway differs; 72 runs, tag -viahead.
  (e) Correlated weather noise: exo_mode noisy_ar, AR(1) error path along the horizon with
  phi = exp(-1/12) and the same marginal sd as the white proxy; TiDE and PLANET on DE/FR/ES,
  54 runs, tag -arnoise.
  (f) Hong's regression (tao_vanilla.py): the trend term is selected on validation for both
  variants; TaoRefit selects its fitting window (expanding / 36 / 24 months) and the trend by
  the validation loss of monthly-refit forecasts. Test on NL under R1: once-fitted 14.8 -> 8.3 %
  WAPE, refit 9.5 -> 6.6 % (24-month window). Old records moved to results_v2_discarded/tao_trend/;
  re-run started on CPU (logs/tao_rev.log).
  (g) Day-ahead protocol (dayahead_v2.py): second origin at 08:00 UTC (leads 16-39 h,
  results_v2/dayahead_08.json), seed ensembles as their own entries, Diebold-Mariano of every
  entry against the operator on daily MAE, monthly / lockdown (2020-03-10..05-31) / holiday-day
  WAPE tallies, the H=48 training stated in the protocol block. Re-run: DE PLANET ensemble 1.94 %
  vs operator 2.85 %, DM -6.95.
  (h) Statistics (stats.py, paper_numbers.py): stationary-bootstrap DM with the MCS block length
  (B = 1000) beside the HAC test, effective sample size per cell, per-seed agreement counts,
  the diverged-seed rule (val MSE non-finite or > 10x the median of the other seeds) applied
  uniformly wherever seeds are averaged and the excluded runs listed in a macro; the
  learning-rate-per-regime check and the two sensitivity experiments get their own macros.
  (i) Driver (run_all_v2.py): --exo_mode, --variants/--backbones for the ablation stage, the
  tuning stage under another regime (--regimes R0|R1), tuning alias for TiDEh.
  Pipeline: src/pipeline_rev.sh (launched 22:50, waits for the budget-check stage to free the
  GPU, then: budget gaps, TiDEh, probabilistic re-run, AR noise, lr per regime on DE, extra
  ablation seeds, Informer/Autoformer under the full protocol, ETT/ECL tuning and re-run, FR
  budget check, monthly-refit day-ahead if its script exists, TimesNet). Budget check at 44/90 at
  22:40, ~12 min per 90-epoch run, so the revision stages start around 07:00 on 20 October.

- **2026-10-19 23:30 (revision work, continued).** (j) Multi-point weather: src/weather_multipoint.py
  fetched ERA5 (Open-Meteo archive) at six metropolitan areas for FR, ES and IT and wrote
  OPSD-{FR,ES,IT}-mp, identical to the base corpora except for the four weather channels, which
  are the population-weighted mean (temperature rmsd to the single point 1.6-2.4 K, correlation
  0.98); run_all_v2 maps the variants to the base system's selections (DATA_ALIAS); the analysis
  keeps them out of every grid aggregate (mp.* macros only). Stage inserted after the AR-noise
  stage: TiDE and PLANET, R2, 54 runs. (k) Monthly-refit day-ahead protocol: src/refit_dayahead.py
  retrains PLANET and TiDE from scratch at the start of every test month (train = every row before
  the month, validation = the two preceding months, H=48, L=336, seed 2021), merges the monthly
  forecasts in the protocol's z-units and lets dayahead_v2.py score them as "<model>-refit";
  exp_v2/data_v2 gained a split override for it; 224 trainings, stage 9 of the pipeline. (l) Hong
  re-run complete (10 min on CPU). (m) Tables: the four clipped tables are split (grid table by
  horizon, regimes / day-ahead / probabilistic tables by systems); the day-ahead table shows the
  seed ensemble under both cut-offs with Diebold-Mariano marks against the operator and monthly-
  refit rows; a selected-configuration appendix table (width, backbone, lr per system, grid-edge
  counts); the per-lead figure caption corrected (it shows the two-week model's per-lead MAE per
  system; the label was right, the caption wrong); "Tao" relabelled "Hong" in figures and tables.
  (n) Text: corpus notes (NL step, FR corrupt hour, GB spikes, all as data.* macros computed from
  the corpus), forecaster counts, pathway control, inherited-selection disclosures, the uniform
  diverged-seed rule, metric and test definitions completed in PROBLEM.tex (effective sample size,
  bootstrap DM, per-seed confirmation, set sizes), extended day-ahead protocol, corrected
  probabilistic protocol (delayed-feedback ACI, qCRPS), versions and determinism; a "Sensitivity
  analyses" subsection (correlated noise, weather representation, lr per regime, budget on DE and
  FR) and a pathway-control paragraph in the mechanism section, with macros that fill in as the
  stages finish. Draft builds at 23 pages without float warnings. Regenerated numbers with the
  revised analysis: two diverged seeds excluded under the uniform rule, 1,445 French records
  masked. Pipeline relaunched 23:05 (budget check at 45/90).

- **2026-10-19 23:30 (text revision, first pass).** Title changed to "Closed-Form Calendar
  Estimation in Deep Load Forecasting: An Information-Controlled Benchmark on Eight National
  Grids" (the phase-locking / local-time detail is now reported as a measured component worth
  about one per cent and nothing, respectively); abstract and contributions 1-2 reframed
  accordingly, the short-history and transfer claims tied to the learned-calendar control, the
  seed-ensemble day-ahead result added. Related work extended with the load-forecasting
  literature (Taylor & McSharry 2007; Hong, Wang & White 2015; GEFCom2017; Haben et al. 2021;
  Lago et al. 2021; Obst, de Vilmarest & Goude 2021; DeepAR; MQ-RNN), the nearest relatives
  (DeepTime, CycleNet, ES-RNN, dynamic harmonic regression, Prophet), the normalisation and
  online-adaptation families (SAN, FAN, Dish-TS, FSNet, OneNet) and the calibration references
  (Zaffran et al. 2022; Chernozhukov et al. 2010; Gneiting & Ranjan 2011); every entry checked
  against the proceedings / journal record (paper/refs_rev.bib; the volume and pages of Obst et
  al. could not be confirmed and are omitted, arXiv id kept). A per-system weather-dependence
  table (R^2 of a quadratic in daily temperature, winter/summer ratio, PLANET vs TiDE and vs
  PatchTST+Cal) replaces the "calendar-dominated" explanation: it does not hold as a rule (NL and
  PL are as weakly weather-dependent as DE and TiDE leads there by 8-23 %; IT the least and TiDE
  leads by 13 %); every sentence that tied the tier's value or PLANET's margin to
  calendar-dominated systems was restated (German system; against the learned-calendar control;
  not predicted by weather dependence). Bootstrap DM: the empirical p-value floored at 1/B is
  useless under Holm over ~70 tests (zero significant results at B=1000), so the statistic is
  referred to the normal distribution with the bootstrap standard error (text updated). Hand-
  typed-number check: version numbers in the reproducibility paragraph treated as context.
  Draft builds at 25 pages, no undefined references or float warnings; 44 macros of the pending
  stages print as ?? until their runs finish.

- **2026-10-19 23:40 (numbers regenerated with the revised analysis; installed in the draft).**
  Bootstrap-standard-error DM: 389 significant wins of the reference after Holm against 410 with
  the HAC variance, 16 losses against 19 (TiDE 11 against 12), so the HAC test overstates
  slightly and the text quotes both. Data notes from the corpus: NL mean load 11.2 GW in 2015 and
  13.0 GW in 2016 (+16.3 %); FR corrupt hour 158 GW between 46 and 50 GW at 2020-07-07 17:00 UTC
  (test split); isolated single-hour spikes (deviation > 15 % of mean load from the mean of the
  neighbours, neighbours within 5 %): GB 36, ES 4, NL 1, others 0 -- the earlier "92" used a
  looser definition and is superseded by the stated one. Hong's regression after the
  trend/window selection: mean grid WAPE 7.40 % (once) and 6.10 % (refit) against 8.47 / 6.69
  before; NL day-ahead refit 5.56 % (was 8.81). Budget check at 16 cells: +3.1 % WAPE on average.
  Hand-typed-number check clean (0 literals); 38 macros undefined, all from the pending stages
  (budget FR, exo.ar, lrsel, mp, path, prob look-ahead) plus the four weather-dependence macros
  added after this run started. Figures regenerated (Hong relabelled; day-ahead figure shows seed
  entries only; ensembles and refits live in the table).

- **2026-10-20 05:10 (numbers regenerated while the budget stage finishes; budget conclusion drafted).**
  Regeneration with 76 of the 90 German 90-epoch records (TimeXer H336 and DLinear pending; the
  main pass is at run 77/90, expected to finish 08:00-09:00, after which pipeline_rev.sh starts
  by itself). Only the budget.* macros and the four new weather-dependence macros changed
  against the 23:40 file (checked by diff). Budget check on 26 German cells: mean WAPE change
  +2.9 % (range -2.1 to +12.0; 21 of 26 cells worse under the longer budget), Spearman 0.93
  (0.90-0.95 per horizon), best headline model changed in 1 of 3 horizons -- at two weeks TiDE
  overtakes PLANET under the longer budget, i.e. the change runs against the proposed model.
  Discussion budget paragraph completed on that basis (the VERIFY marker now refers to the
  French check, stage 8 of the revision pipeline). Weather-dependence macros now in numbers.tex
  (R^2 of daily load on a quadratic in daily temperature, weekdays, training split: FR 0.84,
  BE 0.59, ES 0.42, DE 0.27; winter/summer FR 1.51, BE 1.17, DE 1.09, ES 1.02). Draft: 25
  pages, 0 undefined references, 0 overfull boxes, 0 hand-typed literals; 26 distinct number
  keys undefined, all belonging to pending stages (budget FR 5, exo.ar 4, lrsel 7, mp 4, path 5,
  prob look-ahead gap 1).

- **2026-10-20 05:40 (prior work of the author cited).** Salam & El Hibaoui (2021), Mathematics
  and Computers in Simulation 190, 97-109 (inception-residual + LSTM consumption model; record
  checked on RePEc, DOI 10.1016/j.matcom.2021.05.006), added to refs_rev.bib and cited in the
  introduction with the recurrent load-specific deep models (one self-citation; no claim of
  continuity with this work). Draft rebuilt: 25 pages, 0 undefined citations.

- **2026-10-20 05:50 (venue decided; submission preparation added to the plan).** The user
  delegated the venue choice ("go for the best from your side") after discussing acceptance
  rates (no legitimate journal reaches the 95 % asked about; parallel submission ruled out as
  misconduct). Decision: Energy and AI (Elsevier, OA) first, arXiv preprint the same day; on
  rejection, transfer with reviews in the order Sustainable Energy, Grids and Networks ->
  Mathematics and Computers in Simulation (the author's 2021 venue) -> IEEE Access. IJF kept as
  the alternative first step if the user later accepts a year-long review. Preparation items
  (all strengthen the paper for every venue in the cascade):
  (a) a classical benchmark with autoregressive error structure -- Hong's regression with AR
      errors on the residual (regression with ARMA errors / dynamic-harmonic-regression family),
      order selected on validation; CPU only, implemented in tao_vanilla.py; a protocol addition
      (new model, no change to existing records), logged here when run;
  (b) a cross-model mean combination where predictions are stored (day-ahead stage); the
      headline grid stores per-window losses only, so a grid-wide combination would need every
      member re-run and is not done (recorded as a limitation, not a protocol change);
  (c) a positioning paragraph on regression with time-varying coefficients / dynamic harmonic
      regression as the classical ancestors of the calendar tier;
  (d) at release: Elsevier elsarticle single-column version with ETT/ECL and the long tables
      moved to supplementary material, highlights, declarations, data-availability statement,
      cover letter, suggested reviewers, arXiv package; (e) a simulated review in the style of
      the target journal before release.

- **2026-10-20 07:00 (pre-submission additions implemented: classical benchmark with AR errors;
  forecast combinations).** Protocol additions, no change to any existing record.
  (1) TaoRefitAR (tao_vanilla.py): Hong's regression, monthly refit, plus an AR(p) model of the
  residual -- coefficients by ridge least squares on the training residuals of the train-only
  fit at the selected recency/trend; order chosen on validation (monthly-refit validation
  forecasts with the correction); forecast = regression at the H target steps + recursive AR
  forecast from the p residuals ending at the origin (coefficients of the origin's month, true
  past temperatures; p + 72 h of recency lags inside L = 336). First run with orders
  {0,24,48,168}: 96 records in 17 min, 168 selected in all 32 selections (boundary), so the set
  was extended to {0,24,48,168,240} and the run repeated with --force (logs/tao_ar2.log).
  Preliminary H=24 R2 WAPE (seed 2021, orders up to 168): DE 2.27, FR 2.22, GB 4.45, ES 2.13,
  IT 3.42, NL 3.10, PL 2.09, BE 2.37 % -- against 5.1-7.0 % for the refit alone; still behind
  PLANET/TiDE in every cell. Registered in make_tables_v2 (Hong (refit)+AR), fig_results_v2,
  paper_numbers (BUILTIN_CAL; counted as its own learned model). Consequence to log: the Holm
  family per system and every MCS gain one member, so the DM/MCS counts shift slightly.
  (2) Combinations: day-ahead stage (dayahead_v2.py COMBOS: Comb2 = PLANET+TiDE, CombAll = ten
  headline models; per seed and of the seed ensembles; table rows "Combination (...)"; figure
  skips them); grid level (combo_v2.py -> results_v2/combo.json; seed 2021, the stored seed;
  windows target-aligned on idx+L because the stored idx depends on the model's L; corrupt
  windows dropped; DM vs TiDE/PLANET with the bootstrap SE): Comb2 beats TiDE in 23/24 cells
  (exception PL H96) and PLANET in 24/24, by ~5-8 %; CombTop3 (three lowest validation MSE)
  and CombAll reported; comb.* macros (paper_numbers.emit_combo). Not entered in the headline
  table (single seed vs three-seed means). Text: SETUP (forecaster list, AR scheme), RELATED
  (dynamic regression / time-varying-coefficient positioning; refs hyndman2021fpp3,
  dordonnat2008load verified on RePEc/catalogue), RESULTS (headline paragraph, transfer
  sentence, fifth sensitivity check, day-ahead combination sentence), DISCUSSION (fifth
  implication). Numbers/tables/figures to regenerate when the repeated AR run finishes.

- **2026-10-20 06:50 (AR benchmark and combinations: final numbers installed; draft verified).**
  Repeated TaoRefitAR run with orders {0,24,48,168,240}: 240 selected in 25 of 32 selections,
  168 in 7 (240 is the bound the lookback admits; the setup says so; validation loss changes
  little between 168 and 240). Grid-mean WAPE 4.01 % (refit alone 6.10, once 7.40, PLANET
  2.87); mean rank 19.42 of 27; behind both PLANET and TiDE in every one of the 24 cells
  (checked cell by cell), closest on GB H24 (4.42 vs 4.06). Combinations: Comb2 (PLANET+TiDE,
  seed 2021) beats TiDE in 23/24 cells (16 significantly, bootstrap-SE DM) by -5.2 % on
  average and PLANET in 24/24; CombTop3 -5.3 % vs TiDE, PLANET among the three in 16 cells;
  CombAll -3.6 %. Day-ahead: the combined seed ensemble beats the operator on 5 of 6 unbiased
  systems (4 significantly; PLANET-ens 3, TiDE-ens 5), improves on PLANET's ensemble on all
  8 systems and on TiDE's on 5, tying it on NL/PL/BE. Consequence of the added model on the
  inferential counts (Holm family and MCS gain one member): DM wins 410 -> 432 of 600 -> 624
  tests (HAC), losses 19 -> 18, TiDE losses 12 -> 11; bootstrap-SE wins 389 -> 410, losses 16;
  MCS membership of PLANET unchanged (16/24); PLANET second mean rank unchanged (3.79; TiDE
  1.29); median-method gap -13.5 -> -13.7 %. All quoted through macros. Draft: 27 pages (+2),
  0 undefined citations/references, 0 overfull boxes, 0 hand-typed literals, 26 undefined keys
  (pending GPU stages only). Figures regenerated (no change in content; combinations and the
  AR row are table-only). Budget stage at run 84/90 (DLinear), revision pipeline starts next.

- **2026-10-20 09:30 (deadline; TimesNet dropped; pipeline re-ordered).** The user cannot wait
  until mid-November for the submission. Serial on the Quadro RTX 4000 the remaining stages of
  pipeline_rev.sh were due to finish around 28 October and the deferred TimesNet pass (7-14 days)
  between 4 and 11 November, so the submission would have fallen in the second week of November.
  Decision (venue cascade unchanged): (1) TimesNet leaves the forecaster set. It had no headline,
  lookback or probabilistic record; its ten DE tuning records (H=96, two widths, five learning
  rates) are moved to results_v2_discarded/timesnet_dropped/. run_all_v2.NF_MODELS and EXPENSIVE
  lose the entry (proto.n_nf_models 8 -> 7, proto.n_neural 24 -> 23 with the seasonal-naive reference in the list, proto.n_expensive 5 -> 4; the SETUP count sentence, 26 learned = neural + GBRT + Hong + Hong-AR, was off by one while TimesNet was in the list and is consistent now);
  SETUP lists seven library models, names four expensive models, says that TimeXer alone reads
  only the past covariates natively, and states the exclusion with its reason (runs of one to
  three hours each at full card load; the full schedule beyond the compute budget). The citation
  stays. (2) The remaining stages run from src/pipeline_rev2.sh in a new order: probabilistic
  (resumable, 18 of 216 cells done by the first script), AR noise, multi-point weather, lr per
  regime, extra ablation seeds, FR budget check, monthly-refit day-ahead -- every stage the
  pending macros depend on -- then the two corrections of existing results, ETT/ECL learning
  rates and Informer/Autoformer with instance normalisation, last, so that an overrun can be cut
  without touching a claim. The first script's shell was stopped at 09:08 between two
  probabilistic runs (its running Python job finished as an orphan and wrote its record); the
  second script started at 09:09:48 after the GPU was free. Expected: probabilistic stage
  ~19:00 on 20 October, the macro stages by the morning of 21 October, FR budget check evening of
  21 October, monthly refit early 22 October, ETT/ECL by 23 October, Informer/Autoformer 27-28
  October. Submission target: 29-30 October (Energy and AI, arXiv the same day). Cut rule: if
  Informer/Autoformer is not complete by the morning of 28 October, the discarded norevin records
  are restored and the missing instance normalisation of the two models is disclosed in SETUP.
  Release preparation (elsarticle, supplement, highlights, declarations, cover letter, suggested
  reviewers, simulated review) proceeds on the CPU while the GPU stages run.

- **2026-10-20 09:45 (submission package started on the CPU).** paper/main_els.tex: the Elsevier
  version (elsarticle, preprint option, single column, line numbers) that inputs the same section
  files, numbers, tables and figures as main_v2.tex; front matter with keywords, CRediT,
  competing-interest, funding, data-availability and generative-AI declarations in Elsevier's
  wording (the IEEE acknowledgment wording stays in main_v2.tex); column-width figures take a
  fixed share of the text width. paper/build_els.sh mirrors build_v2.sh (with a placeholder check:
  ELSCOUNTRY and ELSREPOSITORY must be filled by the author). paper/highlights_els.tex (five
  bullets, checked by check_highlights.py against the 85-character rule), paper/cover_letter_els.tex
  (numbers from numbers.tex), paper/make_package.sh (source packages for Elsevier and arXiv with a
  test build), paper/SUBMISSION_NOTES.md (fields to fill, guide-for-authors items to verify in a
  browser since ScienceDirect blocks automated access, candidate reviewers, arXiv steps, release-day
  order). Rebuild after the TimesNet removal running (tables and figures regenerated 09:13).

- **2026-10-20 09:55 (simulated referee pass, part 1, on the 06:48 PDF).** Read as an Energy and
  AI referee would. Fixed: (a) DISCUSSION referred to "the first version of this paper" (not
  permitted; the explanation is now introduced as the natural hypothesis); (b) the day-ahead
  figure caption described a relative change with a sign convention while the figure plots
  absolute WAPE per system with the operator's forecast as a reference bar (caption rewritten);
  (c) the batch-size sentence still counted five expensive models (four since TimesNet left);
  (d) the regimes table shows Hong's regression getting worse with the weather block on Germany
  (once-fitted 4.85 -> 7.20 WAPE at one day; the AR variant 2.05 -> 2.27; Italy similar; France
  nearly halved) -- a genuine property of the protocol, since the regression is fitted on observed
  temperatures and forecasts with the degraded ones while the neural models train on the degraded
  block; the value-of-information paragraph now says so with the cells quoted through the
  R1./R2. macros. Noted, no action: 26 learned forecasters versus 27 ranked rows (Hong once and
  refit are one model in the count and two rows in the tables; the SETUP sentence states it).

- **2026-10-20 10:00 (Elsevier build, table overflow).** (a) main_els.tex: lmodern added (this MiKTeX
  has no cm-super, so the class's T1 EC fonts came out as fourteen bitmapped Type-3 fonts; Latin
  Modern is Type 1); the elsarticle preprint text block is 548 x 390 pt against the IEEE page's
  666 x 516 pt the generated tables are sized for, so geometry sets A4 with 2.5 cm margins
  (702 x 455 pt) and \tabv maps \footnotesize to \scriptsize while a table is input (8 pt at the
  12 pt base, the IEEE size); build_els.sh now also counts Type-3 fonts and "??" in the PDF.
  (b) The day-ahead table had grown to fourteen rows per system (combination and monthly-refit
  rows) and overflowed the page in both builds (IEEE: Float too large by 337 pt, so the table
  was clipped in the 06:48 PDF); make_tables_v2 now writes three system blocks per table
  (tab_dayahead_v2, _b, _c; RESULTS inputs the third). (c) dayahead.json carries operator-only
  entries for the multi-point corpora (OPSD-{FR,ES,IT}-mp); they appeared as blocks of one row
  in the table and as operator-only bars in the day-ahead figure. Both consumers now skip a
  system without a learned entry (paper_numbers already excluded them through is_grid). After
  the fixes: IEEE draft 25 pages, Elsevier draft 42 pages, no float overflow, no overfull box,
  no Type-3 font. Source packages test-built (paper/make_package.sh, release/<stamp>/).

- **2026-10-20 10:20 (table widths; silent check repaired).** The overfull-box count of build_v2.sh
  had always been 0 because its grep pattern ("Overfull \hbox" in double quotes) never matched;
  the 06:48 PDF in fact had seven overfull boxes: the solver equation 83 pt wider than the column,
  the ablation table 40 pt wider than the page, the three day-ahead tables 36 pt wider than the
  column (their last column ran into the neighbouring column), and two tables by 3 pt. Fixes:
  the equation is set on two lines (split); make_tables_v2 wraps every tabular in an adjustbox
  of at most the line width (a table a few points too wide is scaled slightly, never clipped),
  shortens "Combination" to "Comb." in the day-ahead rows (2.5 pt column separation) and sets
  the ablation tables at 2 pt; both main files load adjustbox. Elsevier version: 2 cm side
  margins (482 pt), tables at IEEE's exact 8/7 pt sizes, emergency stretch for lines with
  unbreakable inline mathematics, the probabilistic-protocol model list in parentheses rather
  than between em-dashes (no break point). Both build scripts now count overfull boxes
  correctly, floats too large for the page and "??" in the PDF, and the release mode refuses them.

- **2026-10-20 18:15 (timeline from measured run times).** Probabilistic stage at 185/216 cells,
  no error (TFT 14-20 min per run, the rest 1.5-2.5 min; the record files give a 22 s start-up
  overhead per run). Remaining GPU time of pipeline_rev2.sh estimated from the recorded
  train_time_s + infer_time_s of comparable cells (scratch script eta_rev2.py): prob remainder
  1.6 h; AR-noise 1.8 h; multi-point 1.7 h; lr selection R0/R1 3.0 + 3.0 h; ablation seeds 3.0 h;
  FR budget check 7.2 h; monthly refits 7.8 h (220 monthly fits of PLANET and TiDE at H = 48) ->
  the macro-dependent stages complete about Wed 21 Oct 23:00; ETT/ECL (tune 245 + main 534
  cells) 27 h -> Fri 23 Oct ~02:00; Informer/Autoformer under the full protocol (60 + 60 tuning,
  192 lookback, 144 headline cells; the lookback sweep alone 34 h at L = 720 / H = 336, batch 16)
  76 h -> Mon 26 Oct ~06:00. Two days of margin before the 28 Oct cut rule. Plan: numbers, prose
  and both builds refreshed after each stage; second simulated referee pass and release checks
  on 26-27 Oct; submission to Energy and AI and the arXiv preprint on Tue 27 or Wed 28 Oct
  (latest 30 Oct). Release README (README_v2.md updated for the revision stages, no local
  absolute paths, new title) to be written with the package on 26-27 Oct.

- **2026-10-20 18:30 (hardware; Informer/Autoformer re-run cancelled).** The air conditioning of
  the room failed today; the card reads 85-88 C at 48-97 W under the 100 W cap (the cap's
  minimum), fan at 90 %, SW thermal slowdown active. Decisions: (a) the Informer/Autoformer
  re-run under the full protocol (stage 6 of pipeline_rev2.sh, 76 GPU-hours of sustained full
  load at batch 16, the longest and hottest stage, for two baselines at the foot of every table
  on which no claim rests) is cancelled; their 456 existing records (made without instance
  normalisation) stay in results_v2 and are reported with the deviation disclosed in SETUP (the
  cut rule of the 09:30 entry, applied early). The stage is neutralised without touching the
  running script: results_v2_discarded/norevin/.moved exists, so the discard is skipped and every
  job of the stage is found done. baselines.py: Informer and Autoformer take revin=False by
  default (pass-through in place of RevIN), so the released code reproduces the records as run;
  the RevIN variant stays available through the flag. (b) Thermal guard (exp_v2.thermal_guard)
  tightened: pause at 86 C until 81 C, no forced resume after five minutes (cap two hours), every
  pause appended to logs/thermal.log because the stage filters hide the console line. Takes effect
  from the next run's process. (c) ETT/ECL re-run (27 h of short runs) kept. Revised end of the
  GPU work: about Fri 23 Oct 02:00 plus thermal pauses; submission moves forward to 24-26 Oct.
  If the room stays above about 35 C the pipeline is to be stopped (kill the bash and python
  processes; every stage resumes) until the cooling is repaired.

- **2026-10-20 19:55 (probabilistic stage complete; four silent losses).** The stage ended 19:31
  with 212 of 216 records: four runs (IT PLANET s2021, NL PatchTST s2023, PL CalTST s2022, BE
  PatchTST s2021) printed their start line and then died without a traceback or a record (the
  pipeline's positive filter on the prob stage would have shown a traceback; a hard process
  death under the hot card is the likely cause, as noted on 2026-10-02). Re-run one by one with
  the identical command while the AR-noise stage shared the card (2.4 GB headroom; thermal
  guard active in both processes); all four completed, 216/216 records verified by name
  (prob_v2.py skips an existing record, so nothing was repeated). numbers, tables, figures and
  both builds regenerated (logs/*_rev7.log). Thermal guard with the tightened thresholds is
  pausing 30-50 s about every 100 s at 81-86 C (logs/thermal.log); the stage estimates of the
  18:15 entry grow by roughly a third while the room stays warm.
  Release preparation: README_release.md drafted (replaces README.md in the public copy; layout,
  environment, the three pipeline scripts and their stages, regeneration, reviewer checks).
  Hosting measured: 8,908 JSON records 49 MB and corpus 41 MB (GitHub); results_v2/errors 4.0 GB
  and results_v2/prob 4.4 GB (Zenodo, two zips, DOI placeholder ZENODO_DOI in the README);
  details in paper/SUBMISSION_NOTES.md.

- **2026-10-20 20:30 (probabilistic section verified against the completed stage; patch 43).**
  All 216 cells, three seeds each. Claims re-checked: ACI brings every method to 89.7-90.5 %
  coverage; TiDE lowest qCRPS on 6 of 8 systems, PLANET on DE and FR; grid-mean qCRPS TiDE 466,
  PLANET 488, N-BEATS 500, PatchTST+Cal 522, PLANET-X 527, N-HiTS 532, PatchTST 560, DLinear 579,
  TFT 580; calibrated width TiDE 2793 < PLANET 3468 < N-BEATS 3594 < PatchTST+Cal 3736 MW and
  Winkler in the same order; raw heads under-covered for every method. Two prose corrections:
  (a) SETUP: the immediate-feedback ACI reaches the same coverage as the delayed form (gap
  0.0 points), its optimism is in width (-13.4 %) and Winkler (-12.9 %), so the sentence now
  quotes those two macros instead of the coverage gap; (b) RESULTS: widths listed in ascending
  order so that "the Winkler scores order them the same way" reads correctly; one sentence on
  TFT (highest qCRPS with DLinear, widest intervals with PatchTST). Pages viewed (IEEE 18 and
  20): figure with nine models, two tables of four systems each, no overflow. Pace note: the
  AR-noise stage runs at about 2.5x the 18:15 estimate (TiDE 8 min at H = 336, PLANET 4 min at
  H = 24; the recorded train_time_s excludes thermal pauses and the card sits at 22 % utilisation
  on these small models), so the macro-dependent stages are now expected around Thu 23 Oct and
  ETT/ECL around Sat 24 Oct; re-estimated tomorrow from the measured rates.

- **2026-10-21 05:40 (AR-noise, multi-point and R0 learning-rate stages complete; pace measured;
  per-regime learning-rate re-run added as a third script).** Stages 3, 3b and 4 (R0) ended at
  23:07, 00:59 and 03:54 with 54, 54 and 50 records and no traceback; the R1 sweep is at 27 of 50.
  Pace: wall-clock over the recorded GPU time of a stage is 1.0 (AR-noise, cool evening) to 1.5
  (R0 sweep at 02-03 h, 73-78 thermal pauses of 20 s per hour, logs/thermal.log, 223 pauses so
  far), so the 20:30 note was too pessimistic. From the measured rates: R1 sweep ~06:40, ablation
  seeds ~10:00 (60 runs at 2.8 GPU min), FR budget ~22:00 Wed 21 Oct (the 90 DE ep90 records took
  10.0 GPU h), refit day-ahead at most 7 h, so the "pending macros" marker ~Thu 22 Oct 06:00; the
  ETT/ECL re-run ~26 h (old sets: ETT 480 records 7.1 GPU h, ECL 54 records 7.2 GPU h, plus the
  sweeps), so ~Fri 23 Oct 08:00.
  Results from the records (seed-mean WAPE; the macros are verified against these when
  paper_numbers finishes): AR(1) weather noise raises WAPE by +4.2 % (PLANET) and +7.9 % (TiDE)
  over the nine DE/FR/ES cells, largest TiDE on FR at H = 336 (+37 %); PLANET's deficit against
  TiDE on these cells narrows from +7.9 % to +4.3 %. Multi-point weather changes WAPE by -1.5 %
  (PLANET) and -3.5 % (TiDE) on FR/ES/IT; the deficit widens from +13.1 % to +15.5 %. Learning
  rate re-selected under R0 (DE, H = 96): 6 of 10 headline models change (CalTST 1e-3 -> 3e-3,
  N-BEATS 1e-2 -> 1e-3, N-HiTS 1e-4 -> 5e-4, PLANET-X 3e-3 -> 5e-4, TiDE 1e-3 -> 3e-3, TimeXer
  1e-3 -> 5e-4), validation MSE -7.0 % on average and -19.6 % at most (CalTST); under R1 so far
  5 of 6 (PLANET 3e-3 -> 1e-3 for -0.1 %).
  Decision (protocol addition, logged here): the margin is material for single cells (up to a
  fifth of the validation MSE, against regime steps of -16 % and -11 % WAPE), so the Germany
  R0 and R1 cells of the models whose selection changed are re-run with their own rate:
  run_all_v2.py gained --lr_regime own (best_tune_regime: the rate with the lowest validation
  loss in the regime's own sweep at the R2 size and backbone; a cell whose own rate equals the
  R2 rate keeps its headline record, so nothing is repeated), tag -ownlr, lookback and width as
  in the headline; by the current selection 99 runs, 5.2 GPU h, about 7 h wall. src/pipeline_rev3.sh
  waits for the second script's COMPLETE marker and runs the stage (log logs/pipeline_rev3.log;
  started 05:28). The headline R0/R1 cells are unchanged (the protocol's selection under R2
  stands); the re-run is reported as a sensitivity next to the margin sentence (regime steps
  under own rates; emitter to be added to paper_numbers.py, records are kind "other", variant
  -ownlr). Cut-off: if the stage has not finished by Sun 25 Oct 12:00 the manuscript goes with
  the margin sentence alone.

- **2026-10-21 05:50 (sensitivity prose written; emitter for the own-rate cells; builds).**
  paper_numbers rev8 (logs/numbers_rev8.log, 86,327 macros): exo.ar.* and mp.* agree with the
  record-level check of 05:40 (TiDE +8.0 %, PLANET +4.3 %, gap +8.0 % -> +4.3 %; multi-point
  TiDE -3.5 %, PLANET -1.5 %, gap +13.1 % -> +15.5 %; the small differences are the bad-hour
  masking). The lrsel.* block compared the regime sweeps at the width selected on Germany alone,
  whereas the driver runs them at the width selected on the mean over DE and FR, so six models
  were dropped (n_models 4); corrected to the width and backbone the sweep ran at (patch in
  emit_revision); verified on the records: R0 10 models, 6 changed, validation MSE -7.0 % mean,
  -19.6 % min, as at 05:40. paper_numbers.py also gained the own-rate emitter (patch 45;
  lrsel.<R>.<ds>.<m>.H<H>.wape_own | dwape_own_pct, lrsel.<R>.grid.*_own, and the regime steps
  lrsel.own.<ds>.<Ra>_vs_<Rb>.mean_dwape_pct against lrsel.r2.* over the same cells; unit-tested
  on synthetic records; emit_revision takes aggs). Patch 46: the three sensitivity paragraphs
  (RESULTS, Section sec:sensitivity) written against the signs of the records (AR noise: TiDE
  loses more, deficit narrows, the white-noise proxy favours the baseline if anyone; multi-point:
  both gain, deficit widens, the single point understates the baseline's margin; learning rate:
  "changes their validation loss by" with the signed macro, as the paper writes every signed
  change, plus the own-rate sentence with its ten macros and a VERIFY comment for the closing
  clause); SETUP points to the re-run. Tables, figures and both builds regenerated (rev8 logs):
  IEEE 26 pages, Elsevier 41 pages, 0 overfull, 0 undefined references; 15 macros still
  undefined, all pending stages (budget.OPSD-FR.* 5, lrsel own-rate 10). Next numbers run after
  the ablation-seed stage (watcher fires at the FR budget start, ~10:00), which also completes
  lrsel.R1.*.

- **2026-10-21 08:05 (R1 learning-rate sweep complete; ablation-seed stage running).** The R1
  sweep ended 07:29 (50 records, 3 h 34 min for 50 runs; 69-84 thermal pauses per hour at
  05-06 h). Final per-regime selection on Germany at H = 96: R0 6 of 10 models change (as at
  05:40); R1 7 of 10 (CalTST 1e-3 -> 3e-3, DLinear 3e-3 -> 1e-2, N-BEATS 1e-2 -> 1e-3, N-HiTS
  1e-4 -> 5e-4, PLANET 3e-3 -> 1e-3, PLANET-X 3e-3 -> 1e-3, TFT 3e-3 -> 5e-4), validation MSE
  -4.9 % on average and -17.8 % at most (DLinear); PatchTST, TiDE and TimeXer keep their rate.
  The own-rate stage of pipeline_rev3.sh is therefore 117 runs (54 under R0, 63 under R1; dry
  run, no side effects), about 5.6 GPU h. Ablation seeds started 07:29, 10 of 60 records at
  08:00 (3.1 min per run), so ~10:35; FR budget ~13 h -> ~23:30; refit -> Thu 22 Oct ~06:30;
  ETT/ECL -> Fri 23 Oct ~09:00; own-rate -> Fri 23 Oct ~17:00. Macros for lrsel.R1.* are
  regenerated with the ablation seeds at the next cycle.

- **2026-10-21 10:15 (ablation seeds complete; numbers rev9; patch 47).** The ablation-seed
  stage ended 09:47 (60 records: five variants x DE/FR x three horizons x seeds 2024, 2025, on
  the patch backbone), the FR budget stage started 09:47. Record-level check of the new seeds
  against the old three (mean dWAPE vs the full model over the six cells): no_pl +5.5 % new vs
  +6.4 % old, no_resid +10.6 % vs +12.2 %, relative -0.0 % vs +1.1 %, utc -2.0 % vs -0.3 %; the
  full model's five-seed sd is 0.03-0.14 WAPE points per cell. paper_numbers rev9 (86,390
  macros; the ablation block pairs each variant run with the full-model run of the same seed, so
  variants at three seeds pair with three): abl.n_runs 510, abl.patch no_pl +6.2 %, no_resid
  +11.7 %, relative +0.8 %, utc -1.0 %; lrsel final R0 10/6/-7.0 %/-19.6 %, R1 10/7/-4.9 %/
  -17.8 %. Prose: the ablation paragraph now says five seeds for the full model and the four
  principal variants on the patch backbone, seed-paired. Patch 47: with five seeds the no_pl
  cells FR H24 on the basis stack (-0.3 %) and FR H336 on the patch backbone (-0.6 %) carry the
  opposite sign to the other backbone, so "moves both in the same direction in the same cells"
  (RESULTS, DISCUSSION, CONCLUSION) is qualified to "in every cell where it moves either by more
  than a point", with the two exceptions quoted by macro in RESULTS; the horizon statement is
  restricted to the patch backbone (on the basis stack the French value grows with the horizon);
  CONCLUSION's "least at two weeks on the French one" becomes "least on the French one, where at
  one horizon on each backbone it is worth nothing". Builds (rev9b): IEEE 26 pages, Elsevier 41,
  0 overfull, 0 undefined references; 13 macros undefined (budget.OPSD-FR 3, own-rate 10).
  Note: budget.OPSD-FR.mean_dwape_pct, n_cells, min/max are currently emitted from the partial
  FR budget stage (1 cell at the time of the run) and are refreshed at the next cycle; the
  DISCUSSION sentence keeps its VERIFY marker until then.

- **2026-10-21 10:20 (multi-point copies leaked into the grid tables and figures; fixed).** The
  OPSD-FR/ES/IT-mp records are headline-kind records of a variant dataset, so after the stage
  completed they appeared as three extra systems in the MCS figure (columns with PLANET and TiDE
  only, "in set" denominators of 33 against 24) and in tab_grids_v2 / tab_grids_v2_b. The
  macros were never affected: scopes_of() keeps variants out of the all/grid scopes and the
  DM/MCS loops run over cell_keys, which excludes them (mcs.grid.n_cells = 24, dm.n_tests = 624
  unchanged). Fix: make_tables_v2.systems_in() and fig_results_v2.cells() drop variant datasets
  (pn.is_variant); the -mp results stay in the sensitivity prose (mp.* macros). Tables and
  figures regenerated (rev9b logs), both builds rebuilt (rev9c): IEEE 26 pages, Elsevier 42
  (reflow of the grid table), 0 overfull, 0 undefined references, MCS page 12 inspected (eight
  systems, x/24). Lesson for the release checks: any new dataset name that reaches results_v2
  must be classified by is_variant() or it becomes a system.

- **2026-10-21 20:25 (French budget stage complete; budget macros final; patch 48; numbers rev10,
  builds rev10).** The FR budget stage (main --datasets OPSD-FR --epochs 90 --patience 10
  --suffix=-ep90, 90 runs) completed at 19:19, 9.5 h after its 09:47 start against the 13 h
  estimated, with 60-90 thermal pauses per hour through the afternoon (908 pauses in
  logs/thermal.log at 19:48). The monthly-refit day-ahead stage (refit_dayahead.sh: 8 systems x
  PLANET, TiDE x 14 months = 224 refits) started at 19:19; 32 refits done at 20:11 (~1.6 min
  each), so the stage and the "pending macros ... complete" marker are due ~Thu 22 Oct 01:30.
  Numbers rev10 (9,435 records, 86,489 macros): budget.OPSD-FR final over 30 cells, mean
  +1.3 %, min -8.3 % (TFT H336), max +15.3 % (NBEATS H24); per-horizon Spearman 0.55 / 1.00 /
  0.88, mean 0.81; best model changes at one of three horizons, at one day, PLANET -> CalTST
  (PLANET +8.0 % under the longer budget, CalTST -0.5 %); TiDE is worse under the longer
  budget on all three French horizons (+6.1 / +10.4 / +4.5 %). 17 of the 30 French cells are
  less accurate under the longer budget (24 of 30 on Germany). Germany unchanged: mean +3.2 %,
  Spearman 0.95, one change at two weeks, PLANET -> TiDE. Reading: on both systems the one
  change at the top runs against the proposed model, so the DISCUSSION sentence "the budget
  does not shape the ordering in the proposed model's favour" holds and its VERIFY marker is
  removed. Patch 48 (DISCUSSION, RESULTS): the German change names its models through
  budget.OPSD-DE.H336.best_small / best_large instead of hand-typed names; the French clause is
  added (best model changes at one horizon, again away from PLANET, budget.OPSD-FR.H24.best_large
  overtaking budget.OPSD-FR.H24.best_small at one day); RESULTS' French figures say where the
  change is. Builds rev10: IEEE 26 pages, Elsevier 42, 0 overfull, 0 undefined references;
  10 macros undefined, all own-rate (lrsel.*own*, lrsel.r2.*), the "??" lines in the build logs
  down from 3/4 to 2/3 (IEEE/Elsevier). Timeline brought forward: macro stages ~Thu 01:30,
  ETT/ECL ~Fri 23 Oct 04:00, own-rate ~Fri 12:00, release checks Friday, submission Sat 24 /
  Sun 25 Oct.

- **2026-10-22 01:05 (monthly-refit stage complete; every macro-bearing stage in; patch 49;
  numbers rev11/rev12, builds rev11/rev12).** refit_dayahead.sh finished at 00:29 (5 h 10 min for
  224 refits), dayahead_v2.py rescored both cut-offs (dayahead.json, dayahead_08.json rewritten
  00:29), and the "all stages the pending macros depend on are complete" marker was written;
  the ETT lr sweep (200 tune jobs, stale_moved=0: the old ETT/ECL records are R1 main records
  and are pruned by _prune_stale only when the main stage finds their lr superseded) started
  at 00:29. The "[skip] ... only 24/418 windows saved" lines of the scorer are the per-month
  partial prediction files the refit script leaves in results_v2/errors; the merged
  <ds>_<model>_L336_H48_s2021_v2-R2-refit_pred.npz files (16, one per system and model) are
  what it scored, and all 16 "<model>-refit" entries are present under both cut-offs. Thermal:
  no pauses were logged during the refit stage (short jobs, card cooler); the guard paused
  again at 00:34 under the ETT sweep (86 C, cap 86/81). Refit rows (11:00 cut-off, WAPE):
  PLANET once -> refit BE 2.11 -> 2.06, DE 2.03 -> 2.02, ES 2.32 -> 2.15, FR 2.18 -> 2.19,
  GB 4.48 -> 4.36, IT 2.57 -> 2.73, NL 2.76 -> 2.49, PL 1.81 -> 1.78; TiDE BE 1.89 -> 1.89,
  DE 2.01 -> 2.04, ES 1.90 -> 1.90, FR 1.87 -> 2.02, GB 4.43 -> 4.29, IT 2.27 -> 2.37,
  NL 2.50 -> 2.37, PL 1.61 -> 1.61. Both models beat the operator on exactly the systems the
  once-trained runs do (PLANET: BE, DE, PL; TiDE: BE, DE, IT, PL). Numbers rev11 (9,648
  records) from the rescored day-ahead files; tables and figures rev11; builds rev11 clean
  (IEEE 26 pages, Elsevier 42, 0 overfull, 0 undefined references, the only "??" the own-rate
  sentence). The refit rows render in tab_dayahead_v2/_b/_c with their DM marks and "--" in
  the ensemble columns. Gap found: SETUP promises two complements that measure the handicap of
  training once (seed ensemble, monthly refit) and RESULTS quoted only the ensemble. Patch 49:
  paper_numbers.py emits da.<ds>.<m>-refit.dwape_vs_once_pct (refit against the once-trained
  run of the same seed, 2021; seed mean as fallback) and da.grid.<m>-refit.mean|min|max_
  dwape_vs_once_unbiased_pct, n_systems_vs_once_unbiased, n_better_than_once_unbiased over the
  six unbiased systems; RESULTS gains one sentence after the ensemble sentence of the day-ahead
  paragraph. Numbers rev12 (87,245 macros): PLANET-refit mean -0.8 % (min -7.3 % ES, max +6.2 %
  IT), TiDE-refit mean +2.4 % (min +0.0 %, max +8.2 % FR); n_beats_tso_unbiased 3 and 4, the
  same counts as once-trained. Reading in the text: training once is not what separates the
  learned models from the operators. Builds rev12 clean, same page counts, "??" lines 2/3
  (own-rate only). Side effect noted: the refit rows take part in "best per system" bolding
  of the day-ahead tables (NL: TiDE refit bold at 2.37; PL/BE/NL 08:00 column) and in
  da.grid.any.n_beats_tso_unbiased (value unchanged); the caption already states what the
  refit rows are. Remaining: ETT/ECL re-run (watcher bfgemesv2), own-rate stage (watcher
  bm5xqzk3m, RESULTS VERIFY marker), release checks.

- **2026-10-22 20:15 (ETT re-run complete; ETT reading rewritten against the re-tuned records;
  patch 50; numbers rev13, tables rev13, builds rev13).** The ETT lr sweep (200 tune runs at
  H96, width fixed at the reference-system choice, tag _v2-R2-k*-lr*-tune) ran 00:29-04:14; the
  ETT main stage (480 jobs, R1) ran 04:14-12:58: _prune_stale moved 768 files (384 records +
  their error files) of the cells whose selected rate changed to results_v2/stale/ and re-ran
  them; the 96 cells whose rate held keep their records. Recorded GPU time 5.9 h over 8.7 h wall
  (duty 68 %). ECL lr sweep (45 runs) started 12:58, 25/45 at 20:00, ~18 min wall per run
  against 9.8 min GPU (duty 54 %: 80-88 thermal pauses per hour through the afternoon, 2,081
  pauses in logs/thermal.log). Numbers rev13 (9,864 records): the re-selected rates reorder the
  ETT grid. Mean MSE rank over 16 station-horizon cells, old -> new: FITS 1.69 -> 4.25 (best in
  10 -> 3 cells), RLinear 6.88 -> 2.44 (0 -> 7), N-BEATS 4.75 -> 2.88 (0 -> 5), PatchTST 4.19
  -> 4.25, PLANET 3.75 -> 4.62 (1 -> 0), PLANETX 5.75 -> 5.69, DLinear 7.69 -> 6.75, CalTST
  6.00 -> 7.12, TiDE 4.88 -> 7.38, NHITS 9.44 -> 9.62. Mean MSE: N-BEATS 0.3553, RLinear
  0.3579, FITS 0.3606, PatchTST 0.3618, PLANET 0.3686, PLANETX 0.3717, TiDE 0.3759. The RESULTS
  sentence named FITS as the best and placed PLANET with PatchTST, N-BEATS and TiDE, which the
  records no longer say. Patch 50 (RESULTS only; no other section describes the ETT result):
  "the best mean ranks ... belong to the linear RLinear (macro) and to N-BEATS (macro), and
  PLANET's (macro) sits with PatchTST (macro) and the frequency-domain FITS (macro), ahead of
  TiDE (macro)"; the paragraph's reading (the tier is expected to add little on an industrial
  load and does not) is unchanged. SETUP's sentence on the Electricity/ETT selection (width
  from the reference systems, learning rate re-selected on each dataset's own validation
  split) matches what ran. tab_bench_v2 regenerated (N-BEATS best on ETTh1/ETTh2, FITS on
  ETTm1, PLANET-X on ETTm2). Builds rev13 clean: IEEE 26 pages, Elsevier 42, 0 overfull,
  0 undefined references, "??" lines 2/3 (own-rate sentence only). ECL macros (R1.ECL.*) still
  come from the old records until the ECL main stage replaces them; the Electricity sentence of
  RESULTS is re-read at that cycle. Timeline: ECL sweep ~Fri 23 Oct 02:00, ECL main ~Fri
  14:00-16:00, own-rate stage ~Sat 24 Oct 00:00-02:00, final numbers/tables/builds and release
  checks Saturday, submission Sat 24 / Sun 25 Oct (own-rate cut-off Sun 25 Oct 12:00 holds).

- **2026-10-22 23:20 (ECL sweep complete 22:33, ECL main stage running; thermal-guard timing
  tightened, patch 51; no protocol effect).** The ECL lr sweep (45 runs) finished at 22:33, three
  hours ahead of the 02:00 estimate (the later runs were the short models). The ECL main stage
  (54 jobs, R1, H96/H336) started 22:33: _prune_stale moved 48 files (24 records + their error
  files) of the cells whose selected rate changed to results_v2/stale/, todo=24; the 30 cells
  whose rate held keep their records. Throughput audit (user request 23:05): the card runs at
  the driver's minimum settable power limit (100 W; min_limit 100, max 125), draws ~92 W at
  1425 MHz under the cap, fan 66-69 %, and in the afternoon room cycles 81->86 C in ~20 s of
  training; the guard then pauses ~20 s, so the duty is ~50 % (80-88 pauses/h, 14:00-22:00)
  against 65-75 % at night (24-49/h). No elevation in this session: nvidia-smi -pl / -lgc refuse
  ("Insufficient Permissions"), so the one lever that raises work per joule, a locked SM clock
  (nvidia-smi -lgc 300,1100, elevated prompt), is handed to the author. Patch 51 (src/exp_v2.py,
  thermal_guard): poll interval 20 -> 10 s (less overshoot above 86 C) and in-pause re-check
  10 -> 3 s (every logged pause lasted exactly 20 s = two 10 s checks, i.e. several seconds per
  cycle below 81 C before resuming). Expected gain ~10 % of wall time while pauses continue.
  Paused time is excluded from train_s as before; training time is quoted in the paper only for
  the German H96 runs (complete), and SETUP's hardware sentence names neither a power cap nor a
  clock, so neither the guard change nor a clock lock on the remaining stages (ECL main, own-rate)
  touches a reported number. The ECL main stage process imported the old guard (started 22:33,
  before the patch); the patched guard applies from the own-rate stage (pipeline_rev3.sh). Not
  restarted: the run in progress (ep24 at 2019 s) is worth more than the ~25 min the new guard
  would save over the stage. Timeline: ECL main ~Fri 23 Oct 03:00-05:00, own-rate ~Fri
  14:00-16:00 at the current duty (about Fri 10:00 with a clock lock), final numbers/tables/
  builds and release checks Fri evening/Sat, submission Sat 24 / Sun 25 Oct.

- **2026-10-22 23:25 (clock lock tested: no effect; advice corrected).** The author locked the SM
  clock (nvidia-smi -lgc 300,1100) at ~23:17. Measured over the following ten minutes: 1095 MHz
  at 81-85 W (before: 1125-1155 MHz at 83-89 W on this ECL job, i.e. the power cap already held
  the clock there), pauses 6-8 per 5 min before and after, epoch wall time 81-85 s before and
  after. Energy per unit of work is worse at the lock (82 W / 1095 MHz = 0.075 W per MHz against
  92 W / 1425 MHz = 0.065), because the 31 W base draw (memory at 6500 MHz + board, paid during
  pauses too) is amortised over less work; the GPU runs in the fixed-voltage part of its V/F
  curve below ~1150 MHz, so a lower lock only trades work for heat one for one. Recommendation
  handed to the author: remove the lock (nvidia-smi -rgc); optional second test, lock the memory
  clock to the next supported step (nvidia-smi -lmc 5001; steps 405/810/5001/6501) to cut the
  base draw; optional guard ceiling 88/84 C (card max operating 92, slowdown 94, driver target
  83), expected ~+10 % duty, author's decision. Paper unaffected (no timing quoted for these
  stages). Timeline unchanged from the 23:20 entry.

- **2026-10-23 05:00 (ETT/ECL re-run complete; second script COMPLETE 04:12; own-rate stage
  running; numbers rev14, tables rev14, figs rev14, builds rev14; Electricity sentence verified,
  no prose change).** ECL main stage 22:33-04:10 (24 cells re-run at the re-selected rates; the
  three "CUDA context unusable after OOM -- restarting the stage (batch hint 8)" lines are the
  pipeline's own OOM handling on ECL TiDE H336, seeds 2021-2023, all three records present).
  Records: ECL 99 (54 main + 45 tune), ETT 680 (480 + 200), results_v2/stale 816 files (768 ETT +
  48 ECL). The cancelled Informer/Autoformer stages passed through with todo=0 (04:10-04:12).
  pipeline_rev3.sh picked up the COMPLETE marker at 04:12 and started the own-rate stage (117
  runs, Germany R0/R1 cells, --lr_regime own, suffix -ownlr); 16/117 at 04:47, 0 tracebacks.
  Numbers rev14: 9,890 records, 87,282 macros. ECL mean MSE rank over the 2 horizon cells of 9
  methods, old -> new: PatchTST 1.50 (=), PatchTST+Cal 2.50 (=), DLinear 4.00 (=), PLANET 4.00
  (=), iTransformer 5.00 (=), N-BEATS 6.00 -> 5.50, TiDE 5.00 -> 5.50, PLANET-X 9.00 -> 8.00,
  N-HiTS 8.00 -> 9.00; MCS membership PLANET 0/2 (=), set sizes H96 6 -> 5, H336 7. H96 MSE
  spread above the best (PatchTST+Cal 0.1212): PatchTST 1.3 %, PLANET 2.1 %, TiDE 2.5 %, N-BEATS
  3.7 %, DLinear 3.8 %, iTransformer 4.5 %, PLANET-X 5.0 %, N-HiTS 6.4 %; the calendar variant of
  PatchTST is 1.3 % better at H96 and 2.5 % worse at H336. The RESULTS sentence quotes only
  macros and its two prose claims ("compact models within a few percent of one another at four
  days"; "calendar tier neither help nor harm on this panel") hold on the new records. Builds
  rev14 clean: IEEE 26 pages, Elsevier 42, 0 overfull, 0 undefined references, "??" 2/2 (own-rate
  sentence only). GPU: the author set the memory clock to 5001 MHz (observed 5000 MHz at 04:47);
  the current own-rate NBEATS runs draw 25-40 W and pause ~11/h, so the effect of the memory lock
  is not measurable until the heavier models of the stage run. Next: own-rate stage done ~Fri
  10:00-12:00 at this pace; then numbers rev15, fill the own-rate clause at the RESULTS VERIFY
  marker, --release builds, second simulated referee pass, highlights/cover letter, package;
  submission Sat 24 Oct (Sun 25 at the latest; cut-off Sun 12:00 holds).

- **2026-10-23 13:40 (own-rate stage complete 12:52; numbers rev15; closing clause written,
  patch 52; last VERIFY marker removed; tables/figs/builds rev15 launched).** pipeline_rev3.sh:
  117 runs 04:14-12:52 (8.6 h wall for ~5.6 GPU h; 0 tracebacks; 117 records + 117 pred files;
  3,075 thermal pauses over the whole revision run). Numbers rev15: 10,004 records, 87,556 macros.
  Own-rate effect on test WAPE (re-run cell vs the headline record at the R2 rate): R0 mean -3.9 %
  over 18 cells (12 improve; TimeXer H24 -34.1 %, H336 -15.1 %; N-BEATS -5.6..-10.9 %; N-HiTS
  -1.2..-5.2 %; TiDE -0.9..-1.5 %; PatchTST+Cal -2.7..+3.9 %; PLANET-X +1.6..+6.3 %); R1 mean
  +1.4 % over 21 cells (11 improve; DLinear +12.7..+26.7 % worse at its validation-chosen rate
  0.01; PatchTST+Cal +2.5..+6.3 %; PLANET +0.8..+1.9 %; N-BEATS -4.0..-7.5 %; PLANET-X
  -1.4..-7.4 %; N-HiTS -1.9..-5.7 %; TFT within 0.4 %). Regime steps on Germany over the re-run
  models without a built-in calendar (18 cells), R2 rate -> own rate: R1 vs R0 -26.3 % -> -21.1 %,
  R2 vs R1 -2.2 % -> -3.4 %, R2 vs R0 -28.1 % -> -24.0 %; over all 27 re-run cells -20.2 -> -17.1,
  -3.4 -> -4.3, -23.2 -> -20.9. The reading of sec:info therefore survives: the calendar step
  shrinks but stays ~6x the weather step on Germany, and the R0->R2 change (-24.0 %) still exceeds
  the PLANET-vs-median gap within R2 (-13.7 % over the six systems). Patch 52 (RESULTS, learning-
  rate-per-regime paragraph): appended "The reading of Section sec:info survives: the calendar
  step shrinks once every model trains at its own rate but remains several times the weather
  step on this calendar-driven system, and the combined change from R0 to R2 over the same cells,
  (macro own R2_vs_R0) against (macro r2 R2_vs_R0) at the R2 rate, still exceeds the gap between
  PLANET and the median learned method within R2 ((macro grid.median_gap_wape_pct) over the six
  systems)." and removed the VERIFY comment; no VERIFY marker remains in paper/sec_v2. SETUP's
  promise (cells re-run with their own rate as a sensitivity, Section sec:sensitivity) is met.
  Next: builds rev15 -> rendered check -> --release builds of both versions -> second simulated
  referee pass on main_els.pdf -> highlights/cover letter -> make_package -> README_release ->
  submission Sat 24 Oct.

- **2026-10-23 13:15 (builds rev15 clean; release builds; cover letter and highlights; patch 53
  wording; second simulated referee pass launched).** Tables/figs rev15 and builds rev15: IEEE 26
  pages, Elsevier 42, 0 overfull, 0 undefined references, 0 unresolved macros (the own-rate
  sentence now renders: "...move from -26.3% and -2.2% at the R2 rate to -21.1% and -3.4% with the
  own rates. The reading of Section 6.2 survives: ... -24.0% against -28.1% at the R2 rate, still
  exceeds the gap ... (-13.7% over the six systems)"). Release builds: build_v2.sh --release OK;
  build_els.sh --release refused only for the two author placeholders (ELSCOUNTRY, ELSREPOSITORY),
  everything else passes. cover_letter_els.pdf and highlights_els.pdf compiled (1 page each;
  check_highlights.py OK, 5 bullets of 74-84 characters). paper_numbers.py --check: 0 hand-typed
  result literals in sec_v2/main_v2; the 6 flagged in main_els.tex are layout dimensions (9.5 pt,
  0.62 textwidth). Highlights and abstract agree with the final day-ahead macros (Comb2 = PLANET +
  TiDE seed ensemble beats the operator on 5 of 6 unbiased systems = da.grid.any). Patch 53
  (wording only): abl.*.no_pl.mean_dwape_pct is the WAPE change when the tier is removed (+6.2 %,
  +5.8 %); RESULTS says "raises WAPE by" but the abstract (main_els, main_v2), the cover letter,
  DISCUSSION and CONCLUSION read "lowers/improves ... by +6.2%". Reworded: abstract and cover
  letter "Removing the tier raises the error of two unrelated backbones by ..."; DISCUSSION "it is
  worth ... in WAPE to two residual backbones ..."; CONCLUSION "is worth ... on average to two
  unrelated residual backbones, which it moves in the same direction ...". Rebuild rev16 launched
  (IEEE --release, Elsevier draft, cover letter). Second simulated referee pass: three reviewer
  agents (ML, practitioner, statistician) on the frozen rev15 Elsevier text, each re-checking its
  19 Oct report's major issues; reports to paper/reviews_v2/review2_*.md. Author e-mail in the
  manuscript and cover letter: salllam.ye@gmail.com (three l's) -- to be confirmed by the author.

- **2026-10-23 14:50 (second simulated referee pass: three reports, all "minor revision";
  patches 54 and 55).** Reports in paper/reviews_v2/review2_{ml,practitioner,statistician}.md.
  Verdicts on the 19 Oct major issues: ML 4 resolved / 8 partly of 12; practitioner 9 resolved /
  8 partly / 1 not (the 5.4 split promise) of 18; statistician 6 resolved / 7 partly of 13. The
  must-fix findings are wording contradictions between sections and three unkept promises; none
  requires a new run. Fixed in patch 54 (45 prose edits, every number still a macro):
  abstract (both main files): "least on the French one" (the basis stack is least at FR one
  day), the lock worth both backbones' values, "the seed ensemble of the two leading models is
  ahead on 5 of 6, significantly on 4" (was "seed ensembles ... on 5"), "more than the gap
  between the proposed model and the median learned method" (was "any architecture gap";
  Informer-to-TiDE gaps are larger); cover letter and highlight 4 the same; highlight 5 names
  the PLANET+TiDE ensemble; INTRO: effects "on the German one, shrink with the horizon";
  METHOD: "capacity-matched at its published width" replaced by the truth (backbones at the
  width selected for PLANET; the ablation pair at that same width); PROBLEM 3.3: the DM
  bootstrap uses dm.B=1000 resamples with the MCS block scheme (mcs.B=2000), both counts and
  the per-seed confirmation are now quoted in 6.1, set sizes by horizon reported in 6.1;
  SETUP: six-point check "on three of the larger systems" (IT has the lowest temperature share
  in Table 15, so "where the weather carries the most load" was false), "every reported cell"
  seed-sd claim restricted to the tables that print it, "48-hour point models trained under the
  headline protocol", the month/lockdown/holiday split promise replaced by what is reported
  (lockdown and holiday counts in 6.3; monthly splits in the released summaries); RESULTS:
  seed-stability "TFT the least of the six" (TFT is not the largest Transformer), classical
  ordering corrected (Hong(refit)+AR 19.42 ranks above GBRT 21.96; once/refit forms below the
  naive on several systems because of the trend term), Informer/Autoformer "under the
  full-information regime, the only one they were run under", DM counts with bootstrap
  (410/198/16) and per-seed (413/17) variants, MCS membership by horizon with set sizes, grid
  means labelled as such in the Germany example, "one model is harmed" -> DLinear worse with
  weather on DE at every horizon and N-HiTS on NL, plus Hong harmed outright, same-direction
  claim restated as "every cell but two, where one backbone's effect is within a point of
  zero" (also DISCUSSION and CONCLUSION), short-history claim restated with the DE pair quoted
  (2.69 vs 2.71 at a quarter is thin; PLANET passes TiDE on ES at a quarter only; it already
  leads TiDE on DE), transfer "about as well in relative terms (+74.8 vs +71.0) and to a lower
  absolute error (3.99 vs 4.32)", pathway control "sits +5.6% behind PLANET (worse in 17 of 24
  cells)" (sign was hidden), own-rate clause "over the eight systems" (I had written six),
  TimesFM R2 > R0 in every cell flagged as the NumPy port, cost overhead base made explicit
  (standalone N-BEATS row, -22.5% parameters for PLANET; tier parameters from the ablation
  pair), day-ahead counts with significance (Comb2 5 by point / 4 by test; TiDE-ens 5/3;
  PLANET-ens 3/3), "at least one learned model or seed ensemble", holiday/lockdown/outside
  counts sentence; DISCUSSION: NL/PL/IT margins claim dropped, GB "within a point",
  white-noise proxy "kinder in every respect that matters here" priced with the AR(1) check
  (was "harsher"), budget "edges past ... by a small margin (2.67 vs 2.69)", practice
  paragraph: TiDE or the PLANET+TiDE mean is the default, the tier for short history and
  transfer, inference cost +24.2% stated (was "costs nothing"), short history "keeps PLANET
  ahead on DE, past TiDE on ES with a quarter"; CONCLUSION: MCS counts by horizon. Appendix
  cross-reference: new ppref macro (Elsevier ef already prints "Appendix A"; IEEE needs
  the word), fixes "Appendix Appendix A". Patch 55 (paper_numbers.py emitters): prose names for
  budget.best_small|best_large ("CalTST" leaked into the PDF), MCS per-horizon membership and
  set sizes (mcs.<scope>.<m>.H<H>.n_in, mcs.<scope>.H<H>.size_*), day-ahead holiday/lockdown/
  outside-lockdown win counts on the unbiased systems (da.grid.<m>.n_beats_tso_{hol,lock,
  xlock}_unbiased). Not done (would strengthen, needs runs or scope): Informer/Autoformer with
  RevIN; FR/24 ablation seed re-run; GB error floor; refit with three seeds; TimesFM port
  check; sd for the short-history curves; de Vilmarest & Goude 2022 / Ziel 2018 citations;
  ramp/hour-of-day operational metrics. Numbers rev16 running; then tables/figs/builds rev17,
  rendered checks, release builds, package.

- **2026-10-23 15:40 (numbers rev16; patch 56 holiday reading; builds rev17 clean; patch 57
  table/figure model names; rev18 launched).** Numbers rev16: 10,004 records, 89,340 macros;
  budget.OPSD-FR.H24.best_large now renders "PatchTST+Cal". New macro values: MCS membership of
  PLANET by horizon 3 / 5 / 8 (set sizes 1-8 at one day, 1-14 at four days, 2-23 at two weeks of
  27 methods); day-ahead splits on the six unbiased systems, ensembles ahead of the operator:
  holiday days PLANET-ens 2, TiDE-ens 2, Comb2 3; inside the lockdown window 3 / 4 / 4; outside
  it 4 / 5 / 5; mean WAPE over the six on holiday days operator 3.58 vs 4.31 (PLANET-ens) and
  3.67 (TiDE-ens), outside the lockdown 2.14 vs 1.89 and 1.71. Patch 56 (RESULTS, day-ahead):
  "what the operators keep is concentrated on the holiday days" with those macros. Builds
  rev17: IEEE 26 pages, Elsevier 42, 0 overfull / undefined / unresolved; 31 of 33 rendered
  checks matched, the other two only split by elsarticle line numbers; "Appendix Appendix" gone.
  The check also found raw model keys in the rendered tables and figures: every table printed
  "NHITS" (make_tables_v2 PRETTY had "NHiTS" but the records use "NHITS") and the figure
  legends printed "PLANETX" (fig_results_v2 PRETTY lacked it). Patch 57: both maps and the
  table ORDER list extended; tables/figs/builds rev18 launched with the same checks. Next:
  release builds, cover letter/highlights, make_package, memory, status.

- **2026-10-23 16:05 (rev18 clean; release builds; package 20261023-1415; manuscript ready
  for the author's placeholders).** Builds rev18 (after patch 57): IEEE 26 pages, Elsevier 42,
  0 overfull / undefined / unresolved; rendered checks: no raw model key left in any table or
  figure (N-HiTS, PLANET-X), all rewritten sentences present. paper_numbers.py --check: 0
  hand-typed result literals, 0 undefined keys (the 6 flagged lines in main_els.tex are layout
  dimensions). No VERIFY/TODO marker in any source. build_v2.sh --release: OK. build_els.sh
  --release: refused only by ELSCOUNTRY / ELSREPOSITORY (author); everything else passes.
  cover_letter_els.pdf and highlights_els.pdf recompiled with the reworded claims (highlights 5
  bullets, 74-85 characters). make_package.sh: paper/release/20261023-1415/{els,ieee}-*.zip,
  both test-built in place (42 / 26 pages). Elsevier PDF text ~26,000 words including tables,
  captions, references and the review-mode line numbers (author to check the journal's limit).
  Remaining for the author: ELSCOUNTRY, ELSREPOSITORY (main_els.tex), <anonymised> URL
  (main_v2.tex), GitHub repository + Zenodo record (ZENODO_DOI in README_release.md), e-mail
  confirmation, title veto; then build_els.sh --release and make_package.sh once more (two
  minutes) and submit to Energy and AI + arXiv (Sat 24 / Sun 25 Oct).
