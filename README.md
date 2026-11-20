# Sleep-Analysis-Using-EEG-Signals

Analyzes overnight EEG from the Sleep-EDF Database Expanded to: classify
sleep stages against the real hypnogram labels, screen band power for
unusual nights (heuristic, not diagnostic), and model/forecast how EEG
band power trends across the night using six different time-series models.

Originally a single Kaggle notebook (kept for reference under
`notebooks/`); now a modular, locally-runnable pipeline.

## What changed from the original notebook

The original notebook's later cells looked like a working pipeline but
weren't wired to real data:
- Sleep-stage "classification" metrics (accuracy, F1, confusion matrix)
  were computed on a **hardcoded 10-element example array**, not any model's
  predictions.
- The ARIMA/SARIMA cell fit on `np.random.randn(500)` -- **random noise**,
  not the preprocessed EEG signal from earlier cells -- and reported
  `random.uniform(85, 95)` as a fabricated "model accuracy."
- The disorder-detection heuristic compared each band's power to 15% of
  the four bands' own total, which is close to a tautology (one band
  usually exceeds 15% of a sum of four bands by default), so it flagged
  "detected" on almost every run.

This version fixes all three: real hypnogram-based labels, a real
classifier, a real EEG-derived time series for the forecasting models, and
a disorder heuristic referenced against each recording's own baseline
instead of an arbitrary constant.

## Architecture

```
config.py             central constants (paths, bands, model params)
download_data.py      Kaggle API downloader
data_loader.py        finds & pairs PSG/Hypnogram files, loads with mne
preprocessing.py       flat-channel removal, band-pass filter, optional ICA
features.py           epochs raw signal to 30s windows, extracts band-power features + labels
disorder_detection.py  heuristic band-power screening (self-referenced, not diagnostic)
classifier.py          trains/evaluates a real RandomForest sleep-stage classifier
timeseries_models.py  ARIMA, SARIMA, Holt-Winters, Auto-ARIMA, Prophet, LSTM on real band-power trend
evaluation.py          classification + regression metrics shared by the above
visualization.py       saves all plots to figures/
main.py                orchestrates the full pipeline end to end
interview_questions.txt  50 interview questions covering this project's domain
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`prophet` and `tensorflow` are heavier, optional dependencies -- if either
fails to install, `timeseries_models.py` detects that at import time and
skips that model (reported as "skipped" in the comparison table) instead of
crashing the run. Likewise `pmdarima` (Auto-ARIMA) can have numpy version
conflicts on some setups; same graceful-skip behavior applies.

### Getting the data (Kaggle API)

1. Go to kaggle.com -> Settings -> API -> "Create New Token" and save the
   downloaded `kaggle.json` to `~/.kaggle/kaggle.json` (or set
   `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars instead).
2. Run `python download_data.py` (or `python main.py --download`).

This downloads the [Sleep-EDF cassette dataset mirror on
Kaggle](https://www.kaggle.com/datasets/iamommpatel/physiobank-database-sleep-edfx-cassette)
into `data/raw/`. The same data is also available directly, without any
account, from [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)
if you'd rather not use Kaggle.

## Running

```bash
python main.py --n-subjects 3
```

`--n-subjects` controls how many subjects are processed (default in
`config.N_SUBJECTS`). Each PSG file is a ~24h recording, so processing all
197 subjects is a lot of compute for a laptop -- start small and raise it
once you've confirmed everything runs.

Outputs:
- `results/classification_metrics.csv`, `results/classification_per_class.csv`
- `results/disorder_screening.csv`
- `results/ts_model_comparison.csv`
- `figures/*.png` (confusion matrix, band-power chart, forecast comparison, model RMSE comparison)

## Metrics reported

**Classification:** accuracy, balanced accuracy (corrects for the heavy
N2-majority class imbalance in sleep data), weighted/macro precision,
recall, F1, Cohen's Kappa (standard in sleep-staging literature for
agreement scoring), and Matthews Correlation Coefficient (robust single
summary metric under imbalance) -- plus the full per-class report and
confusion matrix.

**Time-series forecasting:** MAE, RMSE, MAPE, R^2, fit time, and AIC where
applicable (ARIMA/SARIMA/Auto-ARIMA/Holt-Winters), compared across all six
models in one table.

## Dataset Information

- Format: European Data Format (EDF), a standard for storing biosignals.
- File types:
  - `*PSG.edf`: whole-night polysomnographic recordings (EEG from Fpz-Cz
    and Pz-Oz, EOG, chin EMG, event marker).
  - `SC*PSG.edf`: "Sleep cassette study", may also contain oro-nasal
    respiration and rectal body temperature.
  - `*Hypnogram.edf`: sleep stages annotated in 30-second epochs (Wake, N1,
    N2, N3, REM).
- Duration: ~24 hours of data per participant.
- Channels: EEG (Fpz-Cz, Pz-Oz), EOG (horizontal), EMG (chin), respiration,
  derived heart rate.
- File naming:
  - Prefix `SC`: healthy subjects (e.g. `SC4001E0.edf`).
  - Prefix `ST`: subjects with sleep disorders (e.g. `ST7002J0.edf`).
  - Suffix `E0`/`E1`: first/second night.
  - Suffix `PSG`/`Hyp`: PSG signals vs. hypnogram annotations.

## Dataset Citations

[1] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE-BME 47(9):1185–1194 (2000)

[2] MS Mourtazaev, B Kemp, AH Zwinderman, HAC Kamphuisen. Age and gender affect different characteristics of slow waves in the sleep EEG. Sleep 18(7):557–564 (1995).

[3] A Rechtschaffen, AE Kales. A manual of standardized terminology, techniques and scoring systems for sleep stages of human subjects. Los Angeles, CA: UCLA Brain Information Service. Brain Research Institute 10 (1968).

[4] B van Sweden, B Kemp, HAC Kamphuisen, EA van der Velde. Alternative electrode placement in (automatic) sleep scoring (Fpz-Cz/Pz-Oz versus C4-A1/C3-A2). Sleep 13(3):279–283 (1990).

[5] B Kemp, A Värri, AC Rosa, KD Nielsen, J Gade. A simple format for exchange of digitized polygraphic recordings. Electroencephalography and Clinical Neurophysiology 82:391–393 (1992).

[6] B Kemp, J Olivan. European data format 'plus' (EDF+), an EDF alike standard format for the exchange of physiological data. Clinical Neurophysiology 114:1755–1761 (2002).

[7] B Kemp. Model-based monitoring of human sleep stages. Ph.D. thesis, Twente University (1987).

[8] B Kemp, AJMW Janssen, MJ Roessen. A digital telemetry system for ambulatory sleep recording. In: Sleep-Wake Research in The Netherlands 4, eds ML Coenen and J Arends, Dutch Society for Sleep Wake Research, pp 129–132 (1993).
