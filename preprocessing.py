"""Signal cleaning: flat-channel removal, band-pass filtering, optional ICA."""

import numpy as np
from mne.preprocessing import ICA

import config


def remove_flat_channels(raw, std_threshold=config.FLAT_SIGNAL_STD_THRESHOLD):
    """Drops channels whose signal is effectively constant (dead electrode)."""
    bad_channels = []
    for ch in raw.ch_names:
        data, _ = raw[ch, :]
        if np.std(data) < std_threshold:
            bad_channels.append(ch)

    if bad_channels:
        raw.info["bads"] = bad_channels
        raw.drop_channels(bad_channels)

    return raw, bad_channels


def bandpass_filter(raw, l_freq=config.BANDPASS_LOW_HZ, h_freq=config.BANDPASS_HIGH_HZ):
    raw.load_data()
    raw.set_eeg_reference()
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", verbose=False)
    return raw


def apply_ica(raw, n_components=config.ICA_N_COMPONENTS, random_state=config.ICA_RANDOM_STATE):
    """Runs ICA and returns a copy of raw with artifact components removed.

    Skipped by default (see config.APPLY_ICA) — on a full-night recording
    this is the slowest step in the pipeline for a marginal cleanup gain on
    this already-fairly-clean dataset.
    """
    ica = ICA(n_components=n_components, random_state=random_state)
    ica.fit(raw)
    raw_cleaned = raw.copy()
    ica.apply(raw_cleaned)
    return raw_cleaned


def preprocess(raw, apply_ica_step=config.APPLY_ICA):
    raw, dropped = remove_flat_channels(raw)
    raw = bandpass_filter(raw)
    if apply_ica_step:
        raw = apply_ica(raw)
    return raw, dropped
