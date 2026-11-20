"""Turns a continuous, annotated Raw recording into (epoch, label) pairs and
per-epoch EEG band-power features suitable for sleep-stage classification.
"""

import mne
import numpy as np
import pandas as pd

import config

_STAGE_TO_INT = {stage: i for i, stage in enumerate(config.STAGE_LABELS)}
_INT_TO_STAGE = {i: stage for stage, i in _STAGE_TO_INT.items()}


def epochs_from_raw(raw, epoch_duration=config.EPOCH_DURATION_SEC):
    """Splits raw into fixed-length epochs aligned to the hypnogram labels.

    Only epochs whose annotation maps to a known stage (config.ANNOTATION_TO_STAGE)
    are kept; unscored/movement epochs are dropped.
    """
    event_id = {desc: i + 1 for i, desc in enumerate(config.ANNOTATION_TO_STAGE)}
    events, _ = mne.events_from_annotations(
        raw, event_id=event_id, chunk_duration=epoch_duration, verbose=False
    )
    if len(events) == 0:
        return None, None

    # Not every recording contains every stage (e.g. a night with no scored
    # N3), so restrict event_id to codes that actually occurred -- passing
    # the full dict makes mne.Epochs raise on the missing ones.
    present_codes = set(events[:, 2])
    event_id = {desc: code for desc, code in event_id.items() if code in present_codes}

    tmax = epoch_duration - 1.0 / raw.info["sfreq"]
    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=0.0, tmax=tmax,
        baseline=None, preload=True, verbose=False,
    )

    desc_by_code = {v: k for k, v in event_id.items()}
    stage_ints = np.array([
        _STAGE_TO_INT[config.ANNOTATION_TO_STAGE[desc_by_code[code]]]
        for code in epochs.events[:, 2]
    ])
    return epochs, stage_ints


_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz


def _band_power(psd_row, freqs, band):
    low, high = band
    mask = (freqs >= low) & (freqs < high)
    return _trapezoid(psd_row[mask], freqs[mask]) if mask.any() else 0.0


def band_power_features(epochs, bands=config.BANDS):
    """Computes per-epoch, per-EEG-channel band power (Welch PSD) and a few
    cross-band ratios that are informative for stage discrimination
    (e.g. delta/beta is high in deep sleep, alpha is prominent in wake).
    """
    eeg_epochs = epochs.copy().pick(picks="eeg")
    psd = eeg_epochs.compute_psd(method="welch", fmin=0.5, fmax=30.0, verbose=False)
    psd_data, freqs = psd.get_data(return_freqs=True)  # (n_epochs, n_channels, n_freqs)

    rows = []
    for epoch_idx in range(psd_data.shape[0]):
        row = {}
        for ch_idx, ch_name in enumerate(eeg_epochs.ch_names):
            powers = {}
            for band_name, band_range in bands.items():
                p = _band_power(psd_data[epoch_idx, ch_idx], freqs, band_range)
                powers[band_name] = p
                row[f"{ch_name}_{band_name}"] = p
            total = sum(powers.values()) or 1e-12
            row[f"{ch_name}_delta_ratio"] = powers["delta"] / total
            row[f"{ch_name}_alpha_beta_ratio"] = (powers["alpha"] + 1e-12) / (powers["beta"] + 1e-12)
        rows.append(row)

    return pd.DataFrame(rows)


def extract_features_and_labels(raw):
    """End-to-end: raw -> (feature DataFrame X, integer label array y)."""
    epochs, stage_ints = epochs_from_raw(raw)
    if epochs is None:
        return None, None
    X = band_power_features(epochs)
    return X, stage_ints


def stage_name(stage_int):
    return _INT_TO_STAGE[stage_int]
