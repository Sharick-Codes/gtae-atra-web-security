"""
src/anomaly_detector.py
Ensemble anomaly detection using GTAE bottleneck embeddings.

Detectors (config-driven):
  - IF   : Isolation Forest (sklearn)
  - OCSVM: One-Class SVM (sklearn)
  - HBOS : Histogram-Based Outlier Score (pure NumPy -- no pyod/numba dep)

CRITICAL: StandardScaler is fit on benign-only embeddings to prevent
data leakage from attack distribution statistics.
"""

import pickle
import numpy as np
import os
import sys
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from config import ANOMALY_CONFIG, DATA_CONFIG
from utils import get_logger, compute_metrics

logger = get_logger(__name__)


# ============================================================
# PURE NUMPY HBOS (avoids pyod/numba/llvmlite chain)
# ============================================================

class HBOSDetector:
    """
    Histogram-Based Outlier Score -- pure NumPy implementation.

    Algorithm: for each feature, build a histogram on training data.
    Anomaly score = sum of -log(bin_density) across features.
    Threshold calibrated from training data contamination percentile.
    """

    def __init__(self, n_bins: int = 40, contamination: float = 0.05):
        self.n_bins = n_bins
        self.contamination = contamination
        self._bin_edges = []
        self._bin_densities = []
        self._threshold = 0.0

    def fit(self, X: np.ndarray) -> "HBOSDetector":
        """Fit histograms on training (benign) data."""
        self._bin_edges = []
        self._bin_densities = []

        for j in range(X.shape[1]):
            counts, edges = np.histogram(X[:, j], bins=self.n_bins, density=True)
            counts = np.where(counts == 0, 1e-10, counts)  # avoid log(0)
            self._bin_edges.append(edges)
            self._bin_densities.append(counts)

        train_scores = self._score(X)
        self._threshold = np.percentile(train_scores, 100 * (1 - self.contamination))
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score per sample (higher = more anomalous)."""
        scores = np.zeros(X.shape[0])
        for j in range(X.shape[1]):
            edges = self._bin_edges[j]
            densities = self._bin_densities[j]
            bin_idx = np.digitize(X[:, j], edges[:-1]) - 1
            bin_idx = np.clip(bin_idx, 0, len(densities) - 1)
            scores += -np.log(densities[bin_idx])
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for anomaly, 0 for normal."""
        return (self._score(X) >= self._threshold).astype(int)


# ============================================================
# TRAINING
# ============================================================

def train_detectors(embeddings_benign: np.ndarray) -> dict:
    """
    Train anomaly detectors on benign-only embeddings.

    Which detectors to train is driven by ANOMALY_CONFIG['ensemble'].
    Returns dict mapping detector name -> trained model.
    """
    detectors = {}
    methods = ANOMALY_CONFIG.get("ensemble", ["IF", "OCSVM", "HBOS"])
    contamination = ANOMALY_CONFIG.get("contamination", 0.05)

    if "IF" in methods:
        n_est = ANOMALY_CONFIG.get("if_n_estimators", 150)
        iso = IsolationForest(n_estimators=n_est, contamination=contamination, random_state=42)
        iso.fit(embeddings_benign)
        detectors["IF"] = iso
        logger.info(f"Trained Isolation Forest ({n_est} trees)")

    if "OCSVM" in methods:
        nu = ANOMALY_CONFIG.get("ocsvm_nu", contamination)
        kernel = ANOMALY_CONFIG.get("ocsvm_kernel", "rbf")
        ocsvm = OneClassSVM(kernel=kernel, nu=nu, gamma="scale")
        ocsvm.fit(embeddings_benign)
        detectors["OCSVM"] = ocsvm
        logger.info("Trained One-Class SVM")

    if "HBOS" in methods:
        n_bins = ANOMALY_CONFIG.get("hbos_n_bins", 40)
        hbos = HBOSDetector(n_bins=n_bins, contamination=contamination)
        hbos.fit(embeddings_benign)
        detectors["HBOS"] = hbos
        logger.info(f"Trained HBOS ({n_bins} bins)")

    return detectors


# ============================================================
# PREDICTION
# ============================================================

def predict_all(detectors: dict, embeddings_scaled: np.ndarray) -> dict:
    """
    Predict anomalies with all detectors + produce ensemble verdict.
    Convention: 1 = anomaly, 0 = normal.

    Voting strategy from ANOMALY_CONFIG['voting']:
      'union'    -> anomaly if ANY detector flags (max recall)
      'majority' -> anomaly if >= ceil(N/2) detectors flag
    """
    predictions = {}

    for name, model in detectors.items():
        if name in ("IF", "OCSVM"):
            # sklearn: -1 = outlier, +1 = inlier
            raw = model.predict(embeddings_scaled)
            predictions[name] = (raw == -1).astype(int)
        elif name == "HBOS":
            # Our HBOSDetector: 1 = anomaly, 0 = normal
            predictions[name] = model.predict(embeddings_scaled)

    # Ensemble voting
    voting = ANOMALY_CONFIG.get("voting", "union")
    pred_sum = np.zeros(len(embeddings_scaled), dtype=int)
    for preds in predictions.values():
        pred_sum += preds

    if voting == "majority":
        threshold = max(2, (len(detectors) + 1) // 2)
        ensemble_pred = (pred_sum >= threshold).astype(int)
    else:  # "union" or fallback
        ensemble_pred = (pred_sum >= 1).astype(int)

    predictions["ensemble"] = ensemble_pred
    return predictions


# ============================================================
# MAIN
# ============================================================

def main():
    """Full anomaly detection pipeline: load embeddings -> train -> evaluate -> save."""
    results_path = DATA_CONFIG["gtae_results_path"]
    if not Path(results_path).exists():
        logger.error(f"GTAE results not found: {results_path}\n"
                     "Run 'python src/gtae_model.py' first.")
        return

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    embeddings   = results["embeddings"]
    recon_errors = results["reconstruction_errors"]
    edge_labels  = results["edge_labels"]
    attack_types = results["attack_types"]

    # Convert tensors to numpy
    if hasattr(embeddings, "numpy"):   embeddings   = embeddings.numpy()
    if hasattr(recon_errors, "numpy"): recon_errors = recon_errors.numpy()
    if hasattr(edge_labels, "numpy"):  edge_labels  = edge_labels.numpy()

    # FIT SCALER ON BENIGN ONLY (no leakage)
    scaler = StandardScaler()
    benign_mask = (edge_labels == 0)
    scaler.fit(embeddings[benign_mask])
    embeddings_scaled = scaler.transform(embeddings)
    embeddings_benign = embeddings_scaled[benign_mask]

    logger.info(f"Training anomaly detectors on {embeddings_benign.shape[0]} benign embeddings...")

    # Train
    detectors = train_detectors(embeddings_benign)

    # Predict
    predictions = predict_all(detectors, embeddings_scaled)

    # Evaluate each detector
    print("\n--- Detector Performance (vs ground-truth labels) ---")
    all_metrics = {}
    for name, pred in predictions.items():
        m = compute_metrics(edge_labels, pred, name=f"{name:20s}")
        all_metrics[name] = m

    # Save scaler
    scaler_path = ANOMALY_CONFIG["scaler_path"]
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # Save individual detectors
    det_paths = {
        "IF":    ANOMALY_CONFIG.get("if_path",    str(Path(scaler_path).parent / "iso_forest.joblib")),
        "OCSVM": ANOMALY_CONFIG.get("ocsvm_path", str(Path(scaler_path).parent / "ocsvm.joblib")),
        "HBOS":  ANOMALY_CONFIG.get("hbos_path",  str(Path(scaler_path).parent / "hbos.joblib")),
    }
    for name, model in detectors.items():
        path = det_paths.get(name, str(Path(scaler_path).parent / f"{name}.joblib"))
        joblib.dump(model, path)
        logger.info(f"Saved {name} -> {path}")

    # Build detection_results list (consumed by ATRA)
    detection_results = []
    for i in range(len(edge_labels)):
        d = {
            "flow_id":              i,
            "true_label":           "attack" if edge_labels[i] == 1 else "benign",
            "attack_type":          attack_types[i],
            "reconstruction_error": float(recon_errors[i]),
            "ensemble_verdict":     "anomaly" if predictions["ensemble"][i] == 1 else "normal",
        }
        # Individual verdicts
        for det_name, pred in predictions.items():
            if det_name != "ensemble":
                d[f"{det_name}_verdict"] = "anomaly" if pred[i] == 1 else "normal"
        detection_results.append(d)

    det_path = DATA_CONFIG["detection_results_path"]
    Path(det_path).parent.mkdir(parents=True, exist_ok=True)
    with open(det_path, "wb") as f:
        pickle.dump(detection_results, f)

    flagged = sum(1 for d in detection_results if d["ensemble_verdict"] == "anomaly")
    print(f"\nSaved {det_path}")
    print(f"Total flows: {len(detection_results)} | Flagged by ensemble: {flagged}")


if __name__ == "__main__":
    main()
