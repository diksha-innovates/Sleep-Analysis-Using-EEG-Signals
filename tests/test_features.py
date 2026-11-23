import numpy as np

import features
import preprocessing
from conftest import make_synthetic_raw


def test_extract_features_and_labels_shapes(synthetic_raw):
    raw, dropped = preprocessing.preprocess(synthetic_raw)
    assert dropped == ["Flatline"]

    X, y = features.extract_features_and_labels(raw)
    assert X is not None
    assert len(X) == len(y) == 40
    assert set(features.stage_name(i) for i in set(y)) == {"Wake", "N1", "N2", "N3", "REM"}


def test_extract_features_handles_missing_stage():
    """Regression test: mne.Epochs used to raise if a stage (e.g. N3) never
    occurs in a given recording, which is common in real hypnograms."""
    raw = make_synthetic_raw(n_epochs=30, stages=["Sleep stage W", "Sleep stage R"])
    raw, _ = preprocessing.preprocess(raw)

    X, y = features.extract_features_and_labels(raw)
    assert X is not None
    assert set(features.stage_name(i) for i in set(y)) == {"Wake", "REM"}


def test_band_power_features_are_finite(synthetic_raw):
    raw, _ = preprocessing.preprocess(synthetic_raw)
    X, _ = features.extract_features_and_labels(raw)
    assert np.isfinite(X.to_numpy()).all()
