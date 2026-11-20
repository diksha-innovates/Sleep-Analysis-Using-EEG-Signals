"""End-to-end pipeline: load Sleep-EDF recordings -> preprocess -> extract
band-power features -> train a real sleep-stage classifier -> heuristic
disorder screening -> compare 6 time-series models on the night's band-power
trend. Run `python main.py --help` for options.
"""

import argparse
import os

import pandas as pd

import classifier
import config
import data_loader
import disorder_detection
import evaluation
import features
import preprocessing
import timeseries_models
import visualization


def process_subject(recording):
    print(f"[{recording.subject_id}] loading ...")
    raw = data_loader.load_raw_with_annotations(recording)

    raw, dropped = preprocessing.preprocess(raw)
    if dropped:
        print(f"[{recording.subject_id}] dropped flat channels: {dropped}")

    X, y = features.extract_features_and_labels(raw)
    if X is None:
        print(f"[{recording.subject_id}] no scoreable epochs found, skipping")
        return None, None
    print(f"[{recording.subject_id}] {len(X)} epochs extracted")
    return X, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the dataset via Kaggle API first")
    parser.add_argument("--n-subjects", type=int, default=config.N_SUBJECTS, help="How many subjects to process")
    args = parser.parse_args()

    if args.download:
        import download_data
        download_data.main()

    pairs = data_loader.find_subject_pairs()[: args.n_subjects]
    if not pairs:
        raise SystemExit(
            f"No PSG/Hypnogram pairs found under {config.RAW_DIR}. "
            "Run with --download, or point config.RAW_DIR at your local copy of the dataset."
        )

    all_X, all_y = [], []
    representative_features = None  # used for disorder screening + time-series modeling

    for recording in pairs:
        X, y = process_subject(recording)
        if X is None:
            continue
        all_X.append(X)
        all_y.append(y)
        if representative_features is None:
            representative_features = X

    if not all_X:
        raise SystemExit("No usable epochs across any processed subject.")

    X = pd.concat(all_X, ignore_index=True)
    y = pd.Series(pd.concat([pd.Series(arr) for arr in all_y], ignore_index=True))

    # ---- 1. Sleep-stage classification (real labels, real predictions) ----
    print("\n=== Sleep stage classification ===")
    present_labels = sorted(y.unique())
    class_names = [features.stage_name(i) for i in present_labels]

    model, scaler, X_test, y_test, y_pred = classifier.train_sleep_stage_classifier(X, y)
    metrics = evaluation.classification_metrics(y_test, y_pred, class_names)

    for k, v in metrics.items():
        if k not in ("per_class", "confusion_matrix"):
            print(f"  {k}: {v:.3f}")

    cm_path = visualization.plot_confusion_matrix(metrics["confusion_matrix"], class_names)
    print(f"  confusion matrix -> {cm_path}")

    scalar_metrics = {k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix")}
    pd.DataFrame([scalar_metrics]).to_csv(os.path.join(config.RESULTS_DIR, "classification_metrics.csv"), index=False)
    metrics["per_class"].to_csv(os.path.join(config.RESULTS_DIR, "classification_per_class.csv"))

    # ---- 2. Heuristic disorder screening (first subject) -------------------
    print("\n=== Heuristic disorder screening (not a diagnosis) ===")
    screening = disorder_detection.screen_for_disorders(representative_features)
    print(" ", disorder_detection.summarize_flags(screening))
    pd.DataFrame(screening).T.to_csv(os.path.join(config.RESULTS_DIR, "disorder_screening.csv"))
    visualization.plot_band_power_bar(screening)

    # ---- 3. Time-series model comparison on the night's delta-power trend --
    print("\n=== Time-series model comparison (delta-band power trend) ===")
    delta_cols = [c for c in representative_features.columns if c.endswith("_delta")]
    delta_series = representative_features[delta_cols].mean(axis=1)

    train, test, ts_results = timeseries_models.run_all_models(delta_series)
    comparison_df = timeseries_models.compare_models(ts_results)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(os.path.join(config.RESULTS_DIR, "ts_model_comparison.csv"), index=False)

    visualization.plot_ts_forecast_comparison(train, test, ts_results)
    visualization.plot_model_metrics_comparison(comparison_df)

    print(f"\nAll results saved under {config.RESULTS_DIR}, figures under {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
