# 💡 StreetLight Anomaly

**4% — AI/ML Portfolio Progress**

StreetLight Anomaly is a machine-learning MVP for **energy-anomaly prioritization**. The repository demonstrates how historical smart-meter patterns can be converted into a ranked technician-review queue.

> **Important:** the benchmark uses **LEAD — Large-scale Energy Anomaly Detection**, a building smart-meter dataset. It is **not** real street-light telemetry. The Streamlit interface is an operational **simulation/prototype**, not a deployed monitoring network.

## Problem

Energy-monitoring teams can face too many readings to inspect manually. A useful system should not pretend that an anomaly score proves equipment failure; it should rank unusual behavior so a technician can decide what deserves inspection first.

```text
hourly meter readings
        ↓
causal temporal features
        ↓
Gradient Boosting benchmark
        ↓
anomaly probability / priority
        ↓
technician review queue
        ↓
human verification
```

## What changed during the experiment

The project began with an unsupervised Isolation Forest baseline. That model carried real ranking signal, but its precision and PR-AUC were limited. The evaluation was then redesigned around causal features, temporal holdout with a purge gap, an unseen-building holdout, capacity-aware Top-K metrics, and event-level metrics. A supervised gradient-boosting model substantially improved the benchmark.

An LSTM Autoencoder was explored but was not retained as the primary path because the simpler tabular approach produced a stronger cost/benefit trade-off for the current data.

## Verified benchmark results

### Temporal holdout

Base anomaly rate: **2.02%**.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Event Recall |
|---|---:|---:|---:|---:|---:|---:|
| Robust-Z | 10.23% | 43.62% | 16.58% | 0.0831 | 0.7468 | 42.15% |
| Isolation Forest | 18.86% | 56.22% | 28.24% | 0.1447 | 0.8401 | 46.15% |
| Gradient Boosting | **51.95%** | **64.82%** | **57.67%** | **0.6693** | **0.9090** | 50.92% |
| Gradient Boosting + IF | 50.53% | 64.76% | 56.77% | 0.6675 | 0.9062 | **51.54%** |

On this temporal test, plain Gradient Boosting is slightly stronger on F1, PR-AUC, ROC-AUC, precision, and false-alert burden. The exported Kaggle bundle is the validation-selected **GradientBoosting+IF** variant; both are documented instead of hiding that distinction.

### Unseen-building holdout

Base anomaly rate: **2.20%**.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Event Recall |
|---|---:|---:|---:|---:|---:|---:|
| Robust-Z | 1.29% | 1.98% | 1.56% | 0.0324 | 0.6650 | 4.36% |
| Isolation Forest | 11.19% | 21.47% | 14.71% | 0.0810 | 0.7970 | 20.52% |
| Gradient Boosting | **90.41%** | **60.15%** | **72.24%** | **0.6276** | **0.8466** | **33.96%** |

False alerts for grouped Gradient Boosting were approximately **0.23 per building per week** on this historical test protocol.

## Capacity-aware ranking

The operational question is not only “is this anomalous?” but also:

> If a team can inspect only the highest-ranked fraction of readings, how many true anomalies are concentrated there?

The repository includes Precision@K, Recall@K, and Lift@K reports for 0.5%, 1%, 2%, 5%, and 10% inspection budgets.

## Causal features

The classifier uses only current and past information, including:

- hour/day/month calendar features;
- lags at 1, 2, 24, and 168 hours;
- trailing rolling mean/std/min/max;
- recent zero and flatline counts;
- differences and ratios to previous readings;
- training-derived hour-of-week median/MAD baseline;
- robust deviation features;
- optional Isolation Forest score.

No SMOTE is used.

## Evaluation protocols

Two protocols answer different questions:

1. **Temporal holdout + 7-day purge** — forward-in-time behavior for buildings represented in training.
2. **Building-grouped holdout** — generalization to buildings never observed during training.

Primary ranking metric: **PR-AUC**. Operational metrics include Precision@K, Recall@K, Lift@K, event recall, and false alerts per building per week. Accuracy is intentionally not the headline metric because anomalies are rare.

## Streamlit simulation

Run:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The demo can either generate fictional `SL-001`, `SL-002`, ... histories or accept a CSV.

CSV input:

```text
timestamp,meter_reading,unit_id
```

or:

```text
timestamp,meter_reading,building_id
```

Provide at least **169 hourly readings per unit** so the 168-hour lag can be computed.

## Simulation rules

- `SL-001` etc. are fictional UI identifiers.
- The app does not claim GPS, live IoT connectivity, or real municipal deployment.
- Model probability is **not fault severity**.
- A flagged row means **review recommended**, not “lamp confirmed broken”.
- Ground-truth anomaly labels are not required for app inference.

## Project structure

```text
streetlight-anomaly/
├── app.py
├── requirements.txt
├── README.md
├── AGENTS.md
├── models/
│   └── streetlight_anomaly_v6_bundle.joblib
├── reports/
│   ├── temporal_model_comparison.csv
│   ├── temporal_topk_comparison.csv
│   ├── grouped_model_comparison.csv
│   ├── grouped_topk_comparison.csv
│   ├── feature_importance.csv
│   └── final_summary.json
├── assets/
│   └── feature_importance.png
├── notebooks/
│   └── StreetLight_Anomaly_v6_LightGBM_Benchmark.ipynb
└── src/
    ├── __init__.py
    ├── features.py
    ├── predict.py
    └── simulation.py
```

## Limitations

- LEAD represents building energy use, not street-light telemetry.
- Historical benchmark performance does not prove production readiness.
- Building behavior, sensors, climates, maintenance patterns, and faults can differ in a real street-light network.
- A field pilot with genuine street-light telemetry is required before operational claims.
- Human verification remains part of the intended workflow.

## Portfolio takeaway

The core lesson is not “deep learning always wins.” It is to **verify** the evaluation and choose the model that best fits the evidence, data, operational budget, and deployment constraints.
