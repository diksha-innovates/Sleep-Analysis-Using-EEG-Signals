import numpy as np
import pandas as pd

import disorder_detection


def test_flat_recording_flags_nothing():
    """If every band is constant all night, no band is a self-referenced
    outlier, so nothing should be flagged."""
    n = 100
    X = pd.DataFrame({
        "EEG Fpz-Cz_delta": np.full(n, 5.0),
        "EEG Fpz-Cz_theta": np.full(n, 5.0),
        "EEG Fpz-Cz_alpha": np.full(n, 5.0),
        "EEG Fpz-Cz_beta": np.full(n, 5.0),
    })
    result = disorder_detection.screen_for_disorders(X)
    assert all(not r["flagged"] for r in result.values())
    assert "No bands showed sustained elevation" in disorder_detection.summarize_flags(result)


def test_sustained_spike_in_one_band_gets_flagged():
    """A band that's elevated for a meaningful chunk of the night (not just
    one noisy epoch) should be flagged; the others shouldn't be. The spike
    covers a minority of epochs so it reads as an outlier against the
    night's own baseline rather than dominating/resetting that baseline."""
    n = 200
    rng = np.random.default_rng(0)
    delta = rng.normal(5.0, 0.5, size=n)
    delta[: n // 5] += 30.0  # sustained elevation for 20% of the night

    X = pd.DataFrame({
        "EEG Fpz-Cz_delta": delta,
        "EEG Fpz-Cz_theta": rng.normal(5.0, 0.5, size=n),
        "EEG Fpz-Cz_alpha": rng.normal(5.0, 0.5, size=n),
        "EEG Fpz-Cz_beta": rng.normal(5.0, 0.5, size=n),
    })
    result = disorder_detection.screen_for_disorders(X)
    assert result["delta"]["flagged"]
    assert not result["beta"]["flagged"]
    assert "elevated_delta" in disorder_detection.summarize_flags(result)
