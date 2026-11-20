"""Plotting helpers. Every function saves a PNG into config.FIGURES_DIR and
returns the path, so main.py doesn't need to juggle matplotlib state."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config


def _savefig(fig, name):
    path = os.path.join(config.FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(conf_matrix, class_labels, name="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("True stage")
    ax.set_title("Sleep Stage Classification - Confusion Matrix")
    return _savefig(fig, name)


def plot_band_power_bar(band_results: dict, name="disorder_screening_band_power.png"):
    bands = list(band_results.keys())
    means = [band_results[b]["mean_power"] for b in bands]
    flagged = [band_results[b]["flagged"] for b in bands]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if f else "#1f77b4" for f in flagged]
    ax.bar(bands, means, color=colors)
    ax.set_xlabel("EEG frequency band")
    ax.set_ylabel("Mean power (this recording)")
    ax.set_title("Whole-night band power (red = sustained elevation flag)")
    return _savefig(fig, name)


def plot_ts_forecast_comparison(train, test, results: dict, name="ts_model_forecasts.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    history_x = np.arange(len(train))
    future_x = np.arange(len(train), len(train) + len(test))

    ax.plot(history_x, train.values, label="train (history)", color="gray", alpha=0.6)
    ax.plot(future_x, test.values, label="actual", color="black", linewidth=2)

    for model_name, res in results.items():
        if res.get("status") == "ok":
            ax.plot(future_x, res["forecast"], label=model_name, linestyle="--")

    ax.set_xlabel("Epoch (30s each)")
    ax.set_ylabel("Delta-band power")
    ax.set_title("Sleep Trend Forecast: Model Comparison")
    ax.legend(fontsize=8)
    return _savefig(fig, name)


def plot_model_metrics_comparison(comparison_df, name="ts_model_metrics_comparison.png"):
    ok_df = comparison_df[comparison_df["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ok_df["model"], ok_df["RMSE"], color="#2ca02c")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Time-Series Model Comparison")
    return _savefig(fig, name)
