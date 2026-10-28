# Public aggregate data only

The assessment supplied 50 smartwatch patient-summary records. Their original repository, redistribution licence, sampling mechanism, device validation and health-score formula were not supplied. The raw CSV is therefore NOT redistributed. This repository contains educational aggregate outputs only, not a clinical dataset or anonymisation guarantee. Patient IDs, individual records, row-level predictions, fitted models and original PBIX files are excluded.

Group summaries contain at least eleven source records and do not include individual trigger lists. Indicator counts overlap. Review allocation is an illustrative policy, not disease risk. The supplied health score is not a confirmed outcome.

src/analyze.py and src/model.py reproduce the original methods only when an authorised local smartwatch_health_data.csv is supplied in 04_Data at repository root. Such input and generated row-level outputs must remain local. The independent random split is 40 training / 10 testing, seed 42. No time backtest is possible because dates are absent. The public dashboard instead uses only the included aggregate snapshots.
