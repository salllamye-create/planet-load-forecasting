# data/proc_v2 audit

Generated 2026-09-27 20:03 by src/audit_data_v2.py over 13 datasets.

Definitions: *DST trans.* = number of changes of `local_stamps - stamps` inside the span; *holidays/yr* = distinct local holiday dates in span divided by span length in years (total distinct dates in brackets); *bridge days* / *xmas days* = distinct local dates flagged; *missing cells interpolated* is taken from the `notes` field written by preprocess_v2.py; *rows target<=0* is on the target channel (system load for grids, MT_369 for ECL, oil temperature OT for ETT); *dup/non-uniform stamps* counts duplicated stamps plus steps != freq (must be 0); *v1 identity* = bit-identity of `data`, `stamps`, `aux` and equality of `splits` with the frozen v1 file in data/proc/ (n/a for systems that are new in v2). *stamps basis* `local(raw)` marks datasets whose raw timestamps are already local civil time (ECL, ETT), for which `local_stamps == stamps`. *holiday source* = `holidays` package version and country[/subdivision] used (GB uses England: Easter Monday and the late-August bank holiday are England/Wales-level in the package).

| dataset | T | C | channels (target last) | span (stamps basis) | tz | DST trans. | holiday source | holidays/yr (total days) | bridge days | xmas days | missing cells interpolated | NaN cells | rows target<=0 | dup/non-uniform stamps | TSO fc coverage | v1 identity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OPSD-DE | 50393 | 7 | solar wind temp rhum rad wspd load | 2015-01-01 07:00 -> 2020-09-30 23:00 (UTC) | Europe/Berlin | 11 | holidays 0.96 DE | 9.0 (52) | 14 | 76 | 170 | 0 | 0.00% | 0 | 100.0% | PASS (data,stamps,aux,splits) |
| OPSD-FR | 50391 | 7 | solar wind temp rhum rad wspd load | 2015-01-01 08:00 -> 2020-09-30 22:00 (UTC) | Europe/Paris | 11 | holidays 0.96 FR | 11.0 (63) | 18 | 76 | 51 | 0 | 0.00% | 0 | 100.0% | PASS (data,stamps,aux,splits) |
| OPSD-GB | 50391 | 7 | solar wind temp rhum rad wspd load | 2015-01-01 09:00 -> 2020-09-30 23:00 (UTC) | Europe/London | 11 | holidays 0.96 GB/ENG | 8.5 (49) | 4 | 76 | 90 | 0 | 0.00% | 0 | 99.8% | PASS (data,stamps,aux,splits) |
| OPSD-ES | 50398 | 7 | solar wind temp rhum rad wspd load | 2015-01-01 01:00 -> 2020-09-30 22:00 (UTC) | Europe/Madrid | 11 | holidays 0.96 ES | 8.3 (48) | 14 | 76 | 48 | 0 | 0.00% | 0 | 100.0% | n/a (new) |
| OPSD-IT | 43817 | 6 | solar temp rhum rad wspd load | 2015-10-02 06:00 -> 2020-09-30 22:00 (UTC) | Europe/Rome | 10 | holidays 0.96 IT | 12.8 (64) | 14 | 70 | 16 | 0 | 0.00% | 0 | 99.9% | n/a (new) |
| OPSD-NL | 50400 | 6 | wind temp rhum rad wspd load | 2015-01-01 00:00 -> 2020-09-30 23:00 (UTC) | Europe/Amsterdam | 11 | holidays 0.96 NL | 10.4 (60) | 13 | 76 | 174 | 0 | 0.00% | 0 | 100.0% | n/a (new) |
| OPSD-PL | 50398 | 6 | wind temp rhum rad wspd load | 2015-01-01 01:00 -> 2020-09-30 22:00 (UTC) | Europe/Warsaw | 11 | holidays 0.96 PL | 13.0 (75) | 18 | 76 | 0 | 0 | 0.00% | 0 | 99.9% | n/a (new) |
| OPSD-BE | 50392 | 7 | solar wind temp rhum rad wspd load | 2015-01-01 08:00 -> 2020-09-30 23:00 (UTC) | Europe/Brussels | 11 | holidays 0.96 BE | 12.0 (69) | 17 | 76 | 8 | 0 | 0.00% | 0 | 100.0% | n/a (new) |
| ECL | 26304 | 321 | MT_001 MT_002 ... MT_369 | 2012-01-01 00:00 -> 2014-12-31 23:00 (local(raw)) | Europe/Lisbon | 0 | holidays 0.96: country_holidays('PT', subdiv=None) evaluated on the local (Europe/Lisbon) civil date | 10.3 (31) | 10 | 42 | 0 | 0 | 0.00% | 0 | - | PASS (data,stamps,splits) |
| ETTh1 | 14400 | 7 | HUFL HULL MUFL MULL LUFL LULL OT | 2016-07-01 00:00 -> 2018-02-20 23:00 (local(raw)) | Asia/Shanghai | 0 | holidays 0.96: country_holidays('CN', subdiv=None) evaluated on the local (Asia/Shanghai) civil date | 21.9 (36) | 1 | 28 | 0 | 0 | 1.69% | 0 | - | PASS (data,stamps,splits) |
| ETTh2 | 14400 | 7 | HUFL HULL MUFL MULL LUFL LULL OT | 2016-07-01 00:00 -> 2018-02-20 23:00 (local(raw)) | Asia/Shanghai | 0 | holidays 0.96: country_holidays('CN', subdiv=None) evaluated on the local (Asia/Shanghai) civil date | 21.9 (36) | 1 | 28 | 0 | 0 | 0.12% | 0 | - | PASS (data,stamps,splits) |
| ETTm1 | 57600 | 7 | HUFL HULL MUFL MULL LUFL LULL OT | 2016-07-01 00:00 -> 2018-02-20 23:45 (local(raw)) | Asia/Shanghai | 0 | holidays 0.96: country_holidays('CN', subdiv=None) evaluated on the local (Asia/Shanghai) civil date | 21.9 (36) | 1 | 28 | 0 | 0 | 1.68% | 0 | - | PASS (data,stamps,splits) |
| ETTm2 | 57600 | 7 | HUFL HULL MUFL MULL LUFL LULL OT | 2016-07-01 00:00 -> 2018-02-20 23:45 (local(raw)) | Asia/Shanghai | 0 | holidays 0.96: country_holidays('CN', subdiv=None) evaluated on the local (Asia/Shanghai) civil date | 21.9 (36) | 1 | 28 | 0 | 0 | 0.12% | 0 | - | PASS (data,stamps,splits) |

## Exogenous-series rules (grids)

1. A solar/wind generation series is dropped if its raw data contains any single internal gap (NaN run between its first and last observation) longer than 720 h (30 days).
2. Then (v1 rule) it is dropped if less than 50% of the load span is present.
Leading NaN blocks are not gaps: the span is trimmed to the first fully observed row (v1 rule).

| dataset | dropped series (reason) |
|---|---|
| OPSD-DE | none |
| OPSD-FR | none |
| OPSD-GB | none |
| OPSD-ES | none |
| OPSD-IT | wind=IT_wind_onshore_generation_actual (internal gap of 1608 h > 720 h) |
| OPSD-NL | solar=NL_solar_generation_actual (internal gap of 2904 h > 720 h) |
| OPSD-PL | solar=PL_solar_generation_actual (8.3% present < 50%) |
| OPSD-BE | none |

**Result: all checks passed.**
