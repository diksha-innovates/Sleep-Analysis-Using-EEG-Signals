"""Heuristic band-power screening.

IMPORTANT: this is NOT a validated diagnostic tool. Real diagnoses of sleep
apnea, insomnia, narcolepsy or RLS require clinical PSG scoring, oximetry,
patient history etc. This module only flags nights where a frequency band
spends an unusually large fraction of time elevated *relative to that same
recording's own baseline* — a starting point for a human to look closer at,
nothing more.

The original version of this heuristic compared each band's power to a
fixed 15% of the four bands' combined total. Because that total is just the
sum of the same four numbers, any one band contributing >15% of it is the
typical case, not an anomaly — so the old check fired "detected" on almost
every recording. This version instead uses each band's own mean/std across
the night (a self-referenced z-score), which actually distinguishes unusual
nights from typical ones.
"""

import numpy as np
import pandas as pd

import config

# Loose, illustrative mapping only — not clinically validated.
_BAND_TO_HEURISTIC_LABEL = {
    "delta": "elevated_delta (possible apnea-related proxy)",
    "beta": "elevated_beta (possible insomnia/arousal proxy)",
    "theta": "elevated_theta (possible narcolepsy/drowsiness proxy)",
    "alpha": "elevated_alpha (possible RLS/wake-intrusion proxy)",
}


def _night_band_series(features_df: pd.DataFrame, band: str) -> pd.Series:
    """Averages one band's power across EEG channels, per epoch."""
    cols = [c for c in features_df.columns if c.endswith(f"_{band}")]
    return features_df[cols].mean(axis=1)


def screen_for_disorders(features_df: pd.DataFrame, z_threshold: float = config.DISORDER_ZSCORE_THRESHOLD):
    """Returns a dict per band: {mean, std, pct_epochs_elevated, flagged, label}."""
    results = {}
    for band in ("delta", "theta", "alpha", "beta"):
        series = _night_band_series(features_df, band)
        mean, std = series.mean(), series.std() or 1e-12
        z_scores = (series - mean) / std
        pct_elevated = float((z_scores > z_threshold).mean() * 100)

        results[band] = {
            "mean_power": float(mean),
            "std_power": float(std),
            "pct_epochs_elevated": round(pct_elevated, 2),
            # Flag only if a meaningful chunk of the night (>10%) is elevated,
            # not just a couple of noisy epochs.
            "flagged": pct_elevated > 10.0,
            "label": _BAND_TO_HEURISTIC_LABEL[band],
        }
    return results


def summarize_flags(screening_result: dict) -> str:
    flagged = [r["label"] for r in screening_result.values() if r["flagged"]]
    if not flagged:
        return "No bands showed sustained elevation vs. this recording's own baseline."
    return "Heuristic flags (not a diagnosis): " + "; ".join(flagged)
