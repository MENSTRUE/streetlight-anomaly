from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .features import build_model_frame


class StreetLightAnomalyModel:
    def __init__(self, bundle_path: str | Path):
        self.bundle_path = Path(bundle_path)
        self.bundle = joblib.load(self.bundle_path)
        self.model = self.bundle["model"]
        self.feature_columns = list(self.bundle["feature_columns"])
        self.threshold = float(self.bundle["threshold"])
        self.model_name = str(self.bundle["model_name"])

    def score_history(self, raw: pd.DataFrame) -> pd.DataFrame:
        frame = build_model_frame(raw, self.bundle)
        valid = frame[self.feature_columns].notna().all(axis=1)

        scored = frame.loc[valid].copy()
        if scored.empty:
            raise ValueError(
                "Not enough valid history. Provide at least 169 hourly readings per unit."
            )

        x = scored[self.feature_columns].astype("float32")
        scores = self.model.predict_proba(x)[:, 1]

        scored["anomaly_probability"] = np.clip(scores, 0.0, 1.0)
        scored["review_required"] = scored["anomaly_probability"] >= self.threshold
        return scored

    def latest_per_unit(self, raw: pd.DataFrame) -> pd.DataFrame:
        scored = self.score_history(raw)
        latest = (
            scored.sort_values(["building_id", "timestamp"])
            .groupby("building_id", as_index=False)
            .tail(1)
            .copy()
        )
        return latest.sort_values("anomaly_probability", ascending=False).reset_index(drop=True)
