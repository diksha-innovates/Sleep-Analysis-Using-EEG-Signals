"""Metrics for both the classifier (sleep stages) and the time-series models
(band-power forecasts)."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)


def classification_metrics(y_true, y_pred, class_labels):
    """Accuracy/F1/kappa plus two additions over the original notebook:

    - balanced accuracy: unlike plain accuracy, not inflated by the
      majority class (sleep-stage datasets are heavily skewed toward N2).
    - Matthews correlation coefficient: a single balanced summary metric
      that behaves sensibly even under class imbalance, unlike F1/accuracy.
    """
    report = classification_report(
        y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "per_class": pd.DataFrame(report).T,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def regression_metrics(y_true, y_pred):
    """Forecast-accuracy metrics for the time-series model comparison."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else np.nan
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2}
