from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demo_data(
    n_units: int = 24,
    hours: int = 240,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate fictional hourly smart-meter-like histories for UI simulation."""
    if n_units < 3:
        raise ValueError("n_units must be at least 3")
    if hours < 169:
        raise ValueError("hours must be at least 169")

    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now().floor("h")
    timestamps = pd.date_range(end=end, periods=hours, freq="h")

    frames = []
    selected = set(rng.choice(np.arange(n_units), size=max(2, n_units // 5), replace=False).tolist())

    for idx in range(n_units):
        unit_id = f"SL-{idx + 1:03d}"
        internal_id = 100_000 + idx

        base = rng.uniform(55, 125)
        phase = rng.uniform(0, 2 * np.pi)
        daily = 12 * np.sin(2 * np.pi * np.arange(hours) / 24 + phase)
        weekly = 5 * np.sin(2 * np.pi * np.arange(hours) / 168 + phase / 2)
        noise = rng.normal(0, 3.0, size=hours)

        readings = np.maximum(0, base + daily + weekly + noise)
        demo_event = np.zeros(hours, dtype="int8")

        if idx in selected:
            event_kind = idx % 3
            start = int(rng.integers(180, max(181, hours - 8)))
            length = int(rng.integers(3, 8))
            stop = min(hours, start + length)

            if event_kind == 0:
                readings[start:stop] *= rng.uniform(0.02, 0.15)
            elif event_kind == 1:
                readings[start:stop] *= rng.uniform(1.8, 2.5)
            else:
                readings[start:stop] = readings[start]

            demo_event[start:stop] = 1

        frame = pd.DataFrame({
            "unit_id": unit_id,
            "building_id": internal_id,
            "timestamp": timestamps,
            "meter_reading": readings.astype("float32"),
            "demo_injected_event": demo_event,
        })
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)
