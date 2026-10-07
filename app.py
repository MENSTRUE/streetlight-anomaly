from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from src.predict import StreetLightAnomalyModel
from src.simulation import generate_demo_data

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "streetlight_anomaly_v6_bundle.joblib"
SUMMARY_PATH = ROOT / "reports" / "final_summary.json"
TEMPORAL_TOPK_PATH = ROOT / "reports" / "temporal_topk_comparison.csv"

st.set_page_config(
    page_title="StreetLight Anomaly",
    page_icon="💡",
    layout="wide",
)


@st.cache_resource
def load_model() -> StreetLightAnomalyModel:
    return StreetLightAnomalyModel(MODEL_PATH)


@st.cache_data
def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def status_label(probability: float, threshold: float) -> str:
    return "REVIEW" if probability >= threshold else "NORMAL"


def explanation(row: pd.Series) -> str:
    reasons = []
    if abs(float(row.get("robust_z", 0))) >= 3:
        reasons.append("large deviation from hour-of-week baseline")
    if float(row.get("zero_count_24", 0)) >= 6:
        reasons.append("many zero readings in recent history")
    if float(row.get("flat_count_24", 0)) >= 8:
        reasons.append("extended near-constant readings")
    if abs(float(row.get("diff_24", 0))) > max(20.0, abs(float(row.get("lag_24", 0))) * 0.5):
        reasons.append("strong change versus 24 hours ago")
    if not reasons:
        reasons.append("combined temporal pattern differs from learned normal behavior")
    return "; ".join(reasons[:2])


st.title("💡 StreetLight Anomaly")
st.caption("Simulation dashboard for energy-anomaly prioritization — human verification required.")

st.info(
    "Simulation Mode: the interface uses fictional street-light IDs and demo telemetry. "
    "The trained model comes from the LEAD building smart-meter benchmark, not real street-light telemetry."
)

model = load_model()
summary = load_summary()

with st.sidebar:
    st.header("Simulation")
    source = st.radio("Data source", ["Generate demo", "Upload CSV"], index=0)

    if source == "Generate demo":
        n_units = st.slider("Number of simulated units", 6, 40, 24, 2)
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
        raw = generate_demo_data(n_units=n_units, hours=240, seed=int(seed))
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        st.caption("Required: timestamp, meter_reading, plus building_id or unit_id. At least 169 hourly rows per unit.")
        if uploaded is None:
            st.stop()
        raw = pd.read_csv(uploaded)

try:
    scored = model.score_history(raw)
except Exception as exc:
    st.error(f"Could not score the data: {exc}")
    st.stop()

# Preserve fictional UI IDs safely.
# score_history() may already preserve unit_id, so avoid creating unit_id_x / unit_id_y.
if "unit_id" not in scored.columns:
    if "unit_id" in raw.columns:
        if "building_id" in raw.columns:
            id_map = (
                raw[["building_id", "unit_id"]]
                .drop_duplicates("building_id")
                .copy()
            )
        else:
            tmp = raw[["unit_id"]].copy()
            tmp["building_id"] = pd.factorize(
                tmp["unit_id"].astype(str),
                sort=True,
            )[0].astype("int32") + 100_000
            id_map = tmp[["building_id", "unit_id"]].drop_duplicates("building_id")

        scored = scored.merge(
            id_map,
            on="building_id",
            how="left",
            validate="many_to_one",
        )
    else:
        scored["unit_id"] = scored["building_id"].map(lambda x: f"SL-{int(x):03d}")

latest = (
    scored.sort_values(["building_id", "timestamp"])
    .groupby("building_id", as_index=False)
    .tail(1)
    .sort_values("anomaly_probability", ascending=False)
    .reset_index(drop=True)
)
latest["status"] = latest["anomaly_probability"].apply(lambda p: status_label(float(p), model.threshold))
latest["reason"] = latest.apply(explanation, axis=1)
latest["priority_rank"] = range(1, len(latest) + 1)

review_count = int(latest["review_required"].sum())
normal_count = int((~latest["review_required"]).sum())
mean_score = float(latest["anomaly_probability"].mean())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Units scored", f"{len(latest):,}")
c2.metric("Needs review", f"{review_count:,}")
c3.metric("Normal", f"{normal_count:,}")
c4.metric("Mean model score", f"{mean_score:.1%}")

st.subheader("Technician Priority Queue")
queue = latest[[
    "priority_rank",
    "unit_id",
    "timestamp",
    "meter_reading",
    "anomaly_probability",
    "status",
    "reason",
]].copy()
queue["anomaly_probability"] = queue["anomaly_probability"].map(lambda x: f"{x:.1%}")
st.dataframe(queue, use_container_width=True, hide_index=True)

selected_unit = st.selectbox("Inspect unit history", latest["unit_id"].tolist())
selected_building = int(latest.loc[latest["unit_id"] == selected_unit, "building_id"].iloc[0])
unit_history = scored[scored["building_id"] == selected_building].copy().sort_values("timestamp")

left, right = st.columns([2, 1])
with left:
    st.subheader(f"Energy history — {selected_unit}")
    chart_df = unit_history.set_index("timestamp")[["meter_reading"]].tail(168)
    st.line_chart(chart_df)

with right:
    row = latest[latest["building_id"] == selected_building].iloc[0]
    st.subheader("Latest assessment")
    st.write(f"**Status:** {row['status']}")
    st.write(f"**Model score:** {float(row['anomaly_probability']):.1%}")
    st.write(f"**Review threshold:** {model.threshold:.1%}")
    st.write(f"**Reading:** {float(row['meter_reading']):.2f}")
    st.write(f"**Why surfaced:** {row['reason']}")
    st.caption("The score is model probability/priority, not physical fault severity.")

st.divider()
st.subheader("Benchmark context")

# Report verified benchmark values from the generated experiment artifacts.
temporal = pd.DataFrame(summary["temporal_comparison"])
grouped = pd.DataFrame(summary["grouped_comparison"])

bt1, bt2 = st.tabs(["Temporal holdout", "Unseen-building holdout"])
with bt1:
    st.dataframe(
        temporal[["model", "base_rate", "precision", "recall", "f1", "pr_auc", "roc_auc", "event_recall"]],
        use_container_width=True,
        hide_index=True,
    )
with bt2:
    st.dataframe(
        grouped[["model", "base_rate", "precision", "recall", "f1", "pr_auc", "roc_auc", "event_recall"]],
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Benchmark results describe historical LEAD experiments only. They do not establish real-world street-light performance."
)
