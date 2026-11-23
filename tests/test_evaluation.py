import numpy as np

import evaluation


def test_classification_metrics_basic():
    y_true = np.array([0, 1, 2, 2, 1, 0, 3, 3, 2, 1])
    y_pred = np.array([0, 1, 2, 1, 1, 0, 3, 2, 2, 1])
    class_labels = ["Wake", "N1", "N2", "REM"]

    metrics = evaluation.classification_metrics(y_true, y_pred, class_labels)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert -1.0 <= metrics["matthews_corrcoef"] <= 1.0
    assert metrics["confusion_matrix"].shape == (4, 4)
    assert list(metrics["per_class"].index[:4]) == class_labels


def test_classification_metrics_perfect_prediction():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    metrics = evaluation.classification_metrics(y_true, y_true, ["Wake", "N1", "N2", "REM"])
    assert metrics["accuracy"] == 1.0
    assert metrics["cohen_kappa"] == 1.0


def test_regression_metrics_zero_error():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = evaluation.regression_metrics(y_true, y_true)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["MAPE_%"] == 0.0


def test_regression_metrics_known_error():
    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    y_pred = np.array([12.0, 12.0, 12.0, 12.0])
    metrics = evaluation.regression_metrics(y_true, y_pred)
    assert metrics["MAE"] == 2.0
    assert metrics["RMSE"] == 2.0
    assert metrics["MAPE_%"] == 20.0
