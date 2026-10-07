# AGENTS.md — StreetLight Anomaly

This file defines source-of-truth rules for AI coding agents working on this repository.

## 1. Product identity

StreetLight Anomaly is an **energy-anomaly prioritization prototype**.

Current product flow:

```text
hourly consumption history
→ causal feature engineering
→ anomaly probability / priority score
→ technician review queue
→ human verification
```

Do not describe the current repository as a live municipal IoT deployment.

## 2. Dataset grounding

Training benchmark:

```text
LEAD — Large-scale Energy Anomaly Detection
```

Critical rule:

```text
LEAD building smart-meter data ≠ real street-light telemetry
```

Never call `building_id` a real lamp ID. The Streamlit demo may display fictional `SL-###` identifiers, but they must remain explicitly labeled simulation IDs.

## 3. Verified temporal benchmark

Base rate: 2.0201%.

```text
Gradient Boosting
Precision  : 51.95%
Recall     : 64.82%
F1         : 57.67%
PR-AUC     : 0.6693
ROC-AUC    : 0.9090
Event recall: 50.92%
```

```text
Gradient Boosting + IF
Precision  : 50.53%
Recall     : 64.76%
F1         : 56.77%
PR-AUC     : 0.6675
ROC-AUC    : 0.9062
Event recall: 51.54%
```

The exported v6 bundle is the validation-selected `GradientBoosting+IF` artifact. Do not silently claim the exported artifact is plain Gradient Boosting.

## 4. Verified grouped benchmark

Base rate: 2.2016%.

```text
Gradient Boosting
Precision  : 90.41%
Recall     : 60.15%
F1         : 72.24%
PR-AUC     : 0.6276
ROC-AUC    : 0.8466
Event recall: 33.96%
False alerts/building/week ≈ 0.23
```

This protocol evaluates unseen-building generalization on historical LEAD data, not deployment performance.

## 5. Score semantics

Critical rule:

```text
anomaly probability / score ≠ physical fault severity
```

A high score means the observed pattern is prioritized by the model. It does not prove:

- a lamp is broken;
- electrical danger exists;
- maintenance urgency or severity;
- a physical fault category.

Use `Review` or `Needs review`, not `Broken` or `Fault confirmed`.

## 6. Feature rules

Production-facing inference must preserve causal features. Never introduce future-looking centered windows without explicitly changing the product to retrospective analysis.

Current feature families:

- calendar;
- lags 1/2/24/168h;
- trailing rolling statistics;
- zero/flatline counts;
- differences and ratios;
- train-derived hour-of-week baseline;
- robust deviation;
- optional Isolation Forest score.

Do not normalize using full-test/full-year statistics.

## 7. Human in the loop

All flagged items are review candidates. Preserve technician verification in UI, README, and product copy.

## 8. Simulation app

The app is a simulation/prototype. It may:

- generate fictional `SL-###` histories;
- upload historical CSV;
- rank current units;
- show temporal charts and model scores;
- show verified benchmark context.

It must not fabricate:

- real GPS locations;
- live sensor connections;
- municipal deployment;
- confirmed repair tickets;
- physical fault severity.

## 9. Ground truth

Ground-truth `anomaly` labels are used for offline benchmark evaluation only. Operational inference must not require or display hidden truth labels as if available in production.

## 10. Model changes

If the model, feature list, threshold, or benchmark changes, update together:

- training notebook;
- exported bundle;
- `src/features.py`;
- `src/predict.py`;
- README;
- AGENTS.md;
- report artifacts.

Never change one silently.
