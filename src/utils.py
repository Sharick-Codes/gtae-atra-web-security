"""
src/utils.py
Shared utility functions used across the IDS-Project pipeline.

Covers:
  - Logging setup
  - Metric calculation (TPR, FPR, precision, recall, F1)
  - Per-attack-type metric breakdown
  - JSON results persistence
  - Plotting helpers (confusion matrix, ROC curve)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


import json
import logging
import logging.handlers
import os
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")   # Headless -- safe in server/subprocess environments
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None
from pathlib import Path
from datetime import datetime
from config import LOGGING_CONFIG, EVALUATION_CONFIG

# ============================================================
# LOGGING SETUP
# ============================================================

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with both console and rotating-file handlers.
    Call this at the top of each module:
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger   # Already configured -- avoid duplicate handlers

    logger.setLevel(getattr(logging, LOGGING_CONFIG["log_level"]))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler
    log_path = Path(LOGGING_CONFIG["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=LOGGING_CONFIG["max_bytes"],
        backupCount=LOGGING_CONFIG["backup_count"],
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================
# METRIC CALCULATION
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> dict:
    """
    Compute binary classification metrics.

    Args:
        y_true: Ground-truth labels (0 = benign, 1 = attack)
        y_pred: Predicted labels (0 = benign, 1 = attack)
        name:   Optional name for printing

    Returns:
        dict with keys: tp, fp, tn, fn, tpr, fpr, precision, recall, f1
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    tpr       = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tpr
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    metrics = {
        "name": name,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr":       round(tpr, 4),
        "fpr":       round(fpr, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }

    if name:
        print(f"{name:<25} | TPR:{tpr*100:6.2f}% | FPR:{fpr*100:6.2f}% | "
              f"F1:{f1:.3f} | TP={tp} FP={fp} TN={tn} FN={fn}")

    return metrics


def compute_per_attack_metrics(
    detection_results: list,
    pred_key: str = "ensemble_verdict",
) -> dict:
    """
    Break down TPR / FPR by attack type.

    Args:
        detection_results: List of per-flow dicts from anomaly_detector
        pred_key: Which verdict field to use

    Returns:
        dict: {attack_type: {tpr, fpr, count, detected}}
    """
    from collections import defaultdict

    # Group by attack type (attack flows only)
    by_type = defaultdict(lambda: {"total": 0, "detected": 0})

    for d in detection_results:
        if d["true_label"] == "attack":
            atype = d.get("attack_type", "Unknown")
            by_type[atype]["total"] += 1
            if d[pred_key] == "anomaly":
                by_type[atype]["detected"] += 1

    results = {}
    print("\n--- Per-Attack-Type Detection Rate ---")
    for atype, counts in sorted(by_type.items()):
        tpr = counts["detected"] / counts["total"] if counts["total"] > 0 else 0.0
        results[atype] = {
            "total":    counts["total"],
            "detected": counts["detected"],
            "tpr":      round(tpr, 4),
        }
        print(f"  {atype:<30} | TPR: {tpr*100:6.2f}% "
              f"({counts['detected']}/{counts['total']})")

    return results


# ============================================================
# RESULTS PERSISTENCE
# ============================================================

def save_results(results: dict, path: str = None):
    """
    Save evaluation results as a JSON file.

    Args:
        results: Dictionary of results
        path:    Output path (defaults to EVALUATION_CONFIG["results_path"])
    """
    path = path or EVALUATION_CONFIG["results_path"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Add run timestamp
    results["run_timestamp"] = datetime.now().isoformat()

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved evaluation results -> {path}")


def load_results(path: str = None) -> dict:
    """Load previously saved evaluation results."""
    path = path or EVALUATION_CONFIG["results_path"]
    with open(path) as f:
        return json.load(f)


# ============================================================
# PLOTTING HELPERS
# ============================================================

def plot_confusion_matrix(
    tp: int, fp: int, tn: int, fn: int,
    title: str = "GTAE-IDS Ensemble: Confusion Matrix",
    path: str = None,
):
    """
    Plot and save a 2x2 confusion matrix heatmap.
    """
    path = path or EVALUATION_CONFIG["confusion_matrix_path"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    cm = np.array([[tn, fp], [fn, tp]])
    labels = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Benign", "Predicted Attack"], fontsize=11)
    ax.set_yticklabels(["Actually Benign", "Actually Attack"], fontsize=11)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")

    ax.set_title(title, fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {path}")


def plot_roc_curve(fpr_list: list, tpr_list: list, auc: float, path: str = None):
    """
    Plot and save a ROC curve.

    Args:
        fpr_list: List of FPR values (x-axis)
        tpr_list: List of TPR values (y-axis)
        auc:      Area Under the Curve value
        path:     Output path
    """
    path = path or EVALUATION_CONFIG["roc_curve_path"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_list, tpr_list, color="steelblue", lw=2,
             label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve -- Ensemble Anomaly Detector", fontsize=13)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved ROC curve -> {path}")


def plot_risk_distribution(atra_results: list, risk_thresholds: dict, path: str = None):
    """
    Plot histogram of risk scores with threshold annotations.
    """
    path = path or EVALUATION_CONFIG["risk_dist_path"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    flagged = [r for r in atra_results if r.get("risk_level") not in (None, "None")]
    if not flagged:
        print("No flagged flows -- skipping risk distribution plot.")
        return

    scores = [r["risk_score"] for r in flagged]

    plt.figure(figsize=(9, 5))
    plt.hist(scores, bins=20, color="#55A868", edgecolor="black", alpha=0.85)

    colors = {"Medium": "orange", "High": "red", "Critical": "darkred"}
    for level, color in colors.items():
        thresh = risk_thresholds[level]
        plt.axvline(thresh, color=color, linestyle="--",
                    label=f"{level} threshold ({thresh})")

    plt.title("Risk Score Distribution (ATRA)", fontsize=13)
    plt.xlabel("Risk Score (0-100)", fontsize=12)
    plt.ylabel("Number of Flows", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved risk score distribution -> {path}")


def plot_action_distribution(atra_results: list, path: str = None):
    """
    Bar chart of ATRA response actions taken.
    """
    path = path or EVALUATION_CONFIG["action_dist_path"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    flagged = [r for r in atra_results if r.get("risk_level") not in (None, "None")]
    if not flagged:
        print("No flagged flows -- skipping action distribution plot.")
        return

    action_counts: dict = {}
    for r in flagged:
        act = r.get("action", "Unknown")
        action_counts[act] = action_counts.get(act, 0) + 1

    plt.figure(figsize=(8, 5))
    plt.bar(action_counts.keys(), action_counts.values(), color="#4C72B0")
    plt.title("ATRA Response Actions Taken", fontsize=13)
    plt.ylabel("Number of Flows", fontsize=12)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved action distribution -> {path}")

