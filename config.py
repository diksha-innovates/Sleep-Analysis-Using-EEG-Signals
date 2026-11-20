"""Central configuration for the Sleep-EEG pipeline. Edit values here instead of
scattering magic numbers through the modules."""

import os

# --- Paths -------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

for _d in (DATA_DIR, RAW_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

# --- Kaggle dataset ------------------------------------------------------
# https://www.kaggle.com/datasets/iamommpatel/physiobank-database-sleep-edfx-cassette
KAGGLE_DATASET = "iamommpatel/physiobank-database-sleep-edfx-cassette"

# --- Pipeline scale -------------------------------------------------------
# Each PSG file is a ~24h recording; ICA + per-epoch PSD on all 197 subjects
# is impractical on a laptop. Process a handful by default; raise this once
# you've confirmed the pipeline runs end to end.
N_SUBJECTS = 3

# --- Preprocessing --------------------------------------------------------
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 40.0
FLAT_SIGNAL_STD_THRESHOLD = 1e-10

# ICA on a full-night recording is slow (minutes per subject) and the
# Sleep-EDF cassette data is already fairly clean, so it's opt-in.
APPLY_ICA = False
ICA_N_COMPONENTS = 0.9999
ICA_RANDOM_STATE = 42

# --- Epoching & frequency bands --------------------------------------------
EPOCH_DURATION_SEC = 30.0  # matches the hypnogram's native scoring window

# Hypnogram annotation text -> unified stage label used throughout the code
ANNOTATION_TO_STAGE = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",  # R&K stages 3/4 both map to AASM N3
    "Sleep stage R": "REM",
}
STAGE_LABELS = ["Wake", "N1", "N2", "N3", "REM"]

# EEG frequency bands (Hz) used for feature extraction / disorder screening
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 15.0),  # spindle band, useful for N2
    "beta": (15.0, 30.0),
}

# --- Disorder screening heuristic -------------------------------------------
# NOT a clinical diagnostic. Flags a band as "elevated" if its power across
# the night is more than this many standard deviations above that *same
# recording's* own mean for that band (self-referenced z-score), rather than
# an arbitrary fixed fraction of total power.
DISORDER_ZSCORE_THRESHOLD = 1.5

# --- Classifier --------------------------------------------------------
TEST_SIZE = 0.25
RANDOM_STATE = 42
RF_N_ESTIMATORS = 200

# --- Time-series models --------------------------------------------------
TS_TEST_SIZE = 0.2
ARIMA_ORDER = (2, 1, 2)
SARIMA_ORDER = (1, 1, 1)
# A full sleep cycle is ~90 min = 180 epochs at 30s/epoch, which is the
# physiologically meaningful seasonal period -- but SARIMAX's Kalman filter
# cost grows sharply with the seasonal period, so 180 can take minutes to
# fit locally. Default to a faster, shorter period and raise it if you have
# the time/CPU budget.
SARIMA_SEASONAL_PERIOD = 20
LSTM_LOOKBACK = 10
LSTM_EPOCHS = 15
