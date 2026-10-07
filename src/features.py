from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize app input into building_id, timestamp, meter_reading columns."""
    out = df.copy()

    if "unit_id" in out.columns and "building_id" not in out.columns:
        codes, _ = pd.factorize(out["unit_id"].astype(str), sort=True)
        out["building_id"] = codes.astype("int32") + 100_000

    required = {"building_id", "timestamp", "meter_reading"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["building_id"] = pd.to_numeric(out["building_id"], errors="coerce")
    out["meter_reading"] = pd.to_numeric(out["meter_reading"], errors="coerce")

    out = (
        out.dropna(subset=["building_id", "timestamp", "meter_reading"])
        .sort_values(["building_id", "timestamp"])
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError("No valid rows remain after parsing the input.")

    out["building_id"] = out["building_id"].astype("int32")
    return out


def add_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    out["hour"] = out["timestamp"].dt.hour.astype("int8")
    out["dow"] = out["timestamp"].dt.dayofweek.astype("int8")
    out["month"] = out["timestamp"].dt.month.astype("int8")
    out["weekend"] = (out["dow"] >= 5).astype("int8")
    out["hour_of_week"] = (out["dow"] * 24 + out["hour"]).astype("int16")

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)

    g = out.groupby("building_id", group_keys=False)

    for lag in [1, 2, 24, 168]:
        out[f"lag_{lag}"] = g["meter_reading"].shift(lag)

    shifted = g["meter_reading"].shift(1)
    group_ids = out["building_id"]

    out["rolling_mean_6"] = (
        shifted.groupby(group_ids).rolling(6, min_periods=3).mean().reset_index(level=0, drop=True)
    )
    out["rolling_mean_24"] = (
        shifted.groupby(group_ids).rolling(24, min_periods=8).mean().reset_index(level=0, drop=True)
    )
    out["rolling_std_24"] = (
        shifted.groupby(group_ids).rolling(24, min_periods=8).std().reset_index(level=0, drop=True)
    )
    out["rolling_min_24"] = (
        shifted.groupby(group_ids).rolling(24, min_periods=8).min().reset_index(level=0, drop=True)
    )
    out["rolling_max_24"] = (
        shifted.groupby(group_ids).rolling(24, min_periods=8).max().reset_index(level=0, drop=True)
    )

    zero_flag = (out["meter_reading"] == 0).astype("float32")
    zero_shifted = zero_flag.groupby(group_ids).shift(1)
    out["zero_count_24"] = (
        zero_shifted.groupby(group_ids).rolling(24, min_periods=8).sum().reset_index(level=0, drop=True)
    )

    eps = 1.0
    out["diff_1"] = out["meter_reading"] - out["lag_1"]
    out["diff_24"] = out["meter_reading"] - out["lag_24"]
    out["diff_168"] = out["meter_reading"] - out["lag_168"]

    out["ratio_lag_1"] = out["meter_reading"] / (out["lag_1"].abs() + eps)
    out["ratio_lag_24"] = out["meter_reading"] / (out["lag_24"].abs() + eps)
    out["ratio_lag_168"] = out["meter_reading"] / (out["lag_168"].abs() + eps)

    out["abs_diff_1"] = out["diff_1"].abs()
    flat_flag = (out["abs_diff_1"] < 1e-6).astype("float32")
    flat_shifted = flat_flag.groupby(group_ids).shift(1)
    out["flat_count_24"] = (
        flat_shifted.groupby(group_ids).rolling(24, min_periods=8).sum().reset_index(level=0, drop=True)
    )

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def apply_baseline_features(frame: pd.DataFrame, tables: dict) -> pd.DataFrame:
    out = frame.copy()

    out = out.merge(
        tables["baseline"],
        on=["building_id", "hour_of_week"],
        how="left",
    )
    out = out.merge(
        tables["global_how_median"],
        on="hour_of_week",
        how="left",
    )
    out = out.merge(
        tables["building_median"],
        on="building_id",
        how="left",
    )

    out["baseline_median"] = (
        out["baseline_median"]
        .fillna(out["global_how_median"])
        .fillna(out["building_median"])
        .fillna(tables["global_median"])
    )
    out["baseline_mad"] = (
        out["baseline_mad"]
        .where(out["baseline_mad"] > 0)
        .fillna(tables["global_mad"])
    )

    eps = 1.0
    out["robust_z"] = (
        0.6745
        * (out["meter_reading"] - out["baseline_median"])
        / (out["baseline_mad"] + eps)
    )
    out["relative_to_baseline"] = (
        (out["meter_reading"] - out["baseline_median"])
        / (out["baseline_median"].abs() + eps)
    )

    building_scale = out["building_median"].fillna(tables["global_median"])
    out["relative_to_building_scale"] = out["meter_reading"] / (building_scale.abs() + eps)

    clip_limits = {
        "robust_z": 30.0,
        "relative_to_baseline": 15.0,
        "relative_to_building_scale": 20.0,
        "ratio_lag_1": 20.0,
        "ratio_lag_24": 20.0,
        "ratio_lag_168": 20.0,
    }
    for column, limit in clip_limits.items():
        if column in out.columns:
            out[column] = out[column].clip(-limit, limit)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def build_model_frame(raw: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    out = normalize_input(raw)
    out = add_causal_features(out)
    out = apply_baseline_features(out, bundle["baseline_tables"])

    if bundle.get("if_model") is not None:
        if_features = bundle["if_features"]
        mask = out[if_features].notna().all(axis=1)
        out["if_score"] = np.nan
        if mask.any():
            x_if = bundle["if_scaler"].transform(out.loc[mask, if_features].astype("float32"))
            out.loc[mask, "if_score"] = -bundle["if_model"].score_samples(x_if)

    return out
