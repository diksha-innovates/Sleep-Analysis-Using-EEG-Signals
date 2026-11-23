"""Shared fixtures. Real EDF files aren't available in CI, so these build
synthetic mne Raw objects / feature DataFrames with the same shape the real
pipeline produces, which is enough to exercise every code path."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def make_synthetic_raw(n_epochs=40, sfreq=100.0, stages=None, include_flat_channel=True):
    """A synthetic multi-channel Raw with hypnogram-style annotations,
    mirroring what data_loader.load_raw_with_annotations hands to the rest
    of the pipeline for a real PSG/Hypnogram pair."""
    import mne

    stages = stages or ["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage 3", "Sleep stage R"]

    ch_names = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]
    ch_types = ["eeg", "eeg", "eog", "emg"]
    if include_flat_channel:
        ch_names.append("Flatline")
        ch_types.append("eeg")

    n_samples = int(sfreq * 30 * n_epochs)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(len(ch_names), n_samples)) * 1e-5
    if include_flat_channel:
        data[-1, :] = 0.0

    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)

    onsets = [i * 30.0 for i in range(n_epochs)]
    descriptions = [stages[i % len(stages)] for i in range(n_epochs)]
    raw.set_annotations(mne.Annotations(onset=onsets, duration=[30.0] * n_epochs, description=descriptions))
    return raw


@pytest.fixture
def synthetic_raw():
    return make_synthetic_raw()


@pytest.fixture
def synthetic_features_and_labels():
    """A band-power feature DataFrame + integer label array shaped like
    features.extract_features_and_labels's real output, with delta power
    made to depend on stage so the classifier has a learnable signal."""
    rng = np.random.default_rng(0)
    n_epochs = 600
    channels = ["EEG Fpz-Cz", "EEG Pz-Oz"]
    bands = list(config.BANDS.keys())

    y = rng.choice(len(config.STAGE_LABELS), size=n_epochs, p=[0.2, 0.05, 0.4, 0.15, 0.2])
    cols = {}
    n3_idx = config.STAGE_LABELS.index("N3")
    for ch in channels:
        for band in bands:
            base = rng.random(n_epochs) * 10
            if band == "delta":
                base = base + (y == n3_idx) * 20
            cols[f"{ch}_{band}"] = base
        total = sum(cols[f"{ch}_{b}"] for b in bands)
        cols[f"{ch}_delta_ratio"] = cols[f"{ch}_delta"] / total
        cols[f"{ch}_alpha_beta_ratio"] = cols[f"{ch}_alpha"] / (cols[f"{ch}_beta"] + 1e-9)

    return pd.DataFrame(cols), y


@pytest.fixture
def synthetic_multi_subject_dataset(synthetic_features_and_labels):
    """Same features, but split into 3 fake subjects so group-based
    splitting has something real to group on."""
    X, y = synthetic_features_and_labels
    n = len(X)
    groups = pd.Series(["S1"] * (n // 3) + ["S2"] * (n // 3) + ["S3"] * (n - 2 * (n // 3)))
    return X, y, groups
