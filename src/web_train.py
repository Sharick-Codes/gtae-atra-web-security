"""
src/web_train.py
Full ML training pipeline for GTAE-ATRA Web Security Monitoring.

Pipeline:
  1. Generate synthetic web traffic (benign + attack sessions)
  2. Build bipartite IP-endpoint graph with Laplacian PE
  3. Train GTAE (Graph Transformer Autoencoder) on benign-only sessions
  4. Compute reconstruction errors + embeddings for all sessions
  5. Train anomaly detector ensemble (IF + OCSVM + HBOS) on benign embeddings
  6. Train attack-type classifier (RandomForest) on attack embeddings
  7. Save all trained artifacts

Reuses:
  - gtae_model.py    (GraphTransformerLayer, GTAE, train_gtae, compute_reconstruction_errors)
  - anomaly_detector.py (train_detectors, predict_all)
  - attack_classifier.py (train attack type RF classifier)
  - utils.py         (logging, metrics)

Usage:
  python src/web_train.py
"""

import sys
import os
import json
import pickle
import numpy as np
import torch
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from config import WEB_CONFIG, GTAE_CONFIG, ANOMALY_CONFIG, MODELS_DIR
from web_data_generator import generate_web_records
from web_graph_builder import build_web_graph
from gtae_model import GTAE, train_gtae, compute_reconstruction_errors
from anomaly_detector import train_detectors, predict_all
from utils import get_logger, compute_metrics, compute_per_attack_metrics

logger = get_logger(__name__)


def train_web_attack_classifier(embeddings_np, labels_np, attack_types):
    """Train a RandomForest attack type classifier on attack embeddings."""
    from sklearn.ensemble import RandomForestClassifier

    attack_mask = labels_np == 1
    if not attack_mask.any():
        logger.warning("No attack samples found -- skipping attack classifier training.")
        return None

    X_attack = embeddings_np[attack_mask]
    y_attack  = [attack_types[i] for i in range(len(attack_types)) if labels_np[i] == 1]

    if len(set(y_attack)) < 2:
        logger.warning("Only one attack type found -- skipping attack classifier.")
        return None

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_attack, y_attack)

    preds = clf.predict(X_attack)
    accuracy = (preds == np.array(y_attack)).mean()
    logger.info(f"Attack classifier training accuracy: {accuracy:.4f}")
    return clf


def compute_anomaly_threshold(errors_benign: np.ndarray, percentile: float = 99.0) -> float:
    """
    Compute anomaly threshold from benign reconstruction errors.
    Uses 99th percentile to provide a robust benign baseline with minimal false positives.
    """
    threshold = float(np.percentile(errors_benign, percentile))
    logger.info(
        f"Anomaly threshold ({percentile}th percentile of benign errors): {threshold:.6f}"
    )
    return threshold


def run_training():
    """Full web security ML training pipeline."""
    logger.info("=" * 70)
    logger.info("GTAE-ATRA WEB SECURITY -- TRAINING PIPELINE")
    logger.info("=" * 70)

    # ----------------------------------------------------------------
    # STEP 1: Generate synthetic web traffic
    # ----------------------------------------------------------------
    logger.info("\n[STEP 1] Generating synthetic web traffic...")
    benign_records, attack_records = generate_web_records()
    records = benign_records + attack_records

    # ----------------------------------------------------------------
    # STEP 2: Build graph
    # ----------------------------------------------------------------
    logger.info("\n[STEP 2] Building web traffic graph (IP -> endpoint)...")
    graph_data = build_web_graph(records)

    out_path = WEB_CONFIG["web_graph_path"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(graph_data, f)

    edge_attr   = graph_data["edge_attr"]
    edge_pe     = graph_data["edge_pe"]
    edge_labels = graph_data["edge_labels"]
    attack_types = graph_data["attack_types"]
    dst_idx     = graph_data["edge_index"][1]

    in_features = int(edge_attr.shape[1])
    pe_dim      = int(edge_pe.shape[1])

    logger.info(f"  in_features={in_features}, pe_dim={pe_dim}")

    # ----------------------------------------------------------------
    # STEP 3: Train GTAE on benign-only sessions
    # ----------------------------------------------------------------
    logger.info("\n[STEP 3] Training GTAE (Graph Transformer Autoencoder)...")
    d_model    = GTAE_CONFIG.get("d_model", 64)
    num_heads  = GTAE_CONFIG.get("num_heads", 4)
    num_layers = GTAE_CONFIG.get("num_layers", 3)

    model = GTAE(
        in_features=in_features, pe_dim=pe_dim,
        d_model=d_model, num_heads=num_heads, num_layers=num_layers,
    )
    model = train_gtae(model, edge_attr, edge_pe, dst_idx, edge_labels)

    # Save web-specific model
    web_model_path = WEB_CONFIG["web_model_path"]
    Path(web_model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), web_model_path)
    logger.info(f"  Saved GTAE model -> {web_model_path}")

    # Save feature schema for inference-time normalization
    schema = {
        "in_features": in_features,
        "pe_dim":      pe_dim,
        "feature_names": WEB_CONFIG["feature_names"],
    }
    schema_path = WEB_CONFIG["web_schema_path"]
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info(f"  Saved feature schema -> {schema_path}")

    # ----------------------------------------------------------------
    # STEP 4: Compute reconstruction errors + embeddings
    # ----------------------------------------------------------------
    logger.info("\n[STEP 4] Computing reconstruction errors and embeddings...")
    errors, embeddings = compute_reconstruction_errors(model, edge_attr, edge_pe, dst_idx)

    errors_np     = errors.cpu().numpy()
    embeddings_np = embeddings.cpu().numpy()
    labels_np     = edge_labels.cpu().numpy()

    benign_mask = labels_np == 0
    attack_mask = labels_np == 1

    benign_errors = errors_np[benign_mask]
    attack_errors = errors_np[attack_mask]

    logger.info(f"  Benign  errors: mean={benign_errors.mean():.6f} std={benign_errors.std():.6f}")
    logger.info(f"  Attack  errors: mean={attack_errors.mean():.6f} std={attack_errors.std():.6f}")

    # Data-driven anomaly threshold from benign validation distribution
    threshold = compute_anomaly_threshold(benign_errors, percentile=95.0)

    # ----------------------------------------------------------------
    # STEP 5: Train anomaly detector ensemble on benign embeddings
    # ----------------------------------------------------------------
    logger.info("\n[STEP 5] Training anomaly detector ensemble...")
    scaler = StandardScaler()
    scaler.fit(embeddings_np[benign_mask])
    embeddings_scaled = scaler.transform(embeddings_np)
    embeddings_benign = embeddings_scaled[benign_mask]

    detectors = train_detectors(embeddings_benign)
    predictions = predict_all(detectors, embeddings_scaled)

    # Evaluate ensemble
    ensemble_pred = predictions["ensemble"]
    metrics = compute_metrics(labels_np, ensemble_pred, name="ensemble")

    logger.info(
        f"  Ensemble | TPR={metrics['tpr']*100:.1f}% | "
        f"FPR={metrics['fpr']*100:.1f}% | F1={metrics['f1']:.3f}"
    )

    # Save scaler + detectors
    scaler_path = Path(MODELS_DIR) / "web_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"  Saved scaler -> {scaler_path}")

    det_paths = {
        "IF":    str(Path(MODELS_DIR) / "web_iso_forest.joblib"),
        "OCSVM": str(Path(MODELS_DIR) / "web_ocsvm.joblib"),
        "HBOS":  str(Path(MODELS_DIR) / "web_hbos.joblib"),
    }
    for name, det in detectors.items():
        path = det_paths.get(name, str(Path(MODELS_DIR) / f"web_{name}.joblib"))
        joblib.dump(det, path)
        logger.info(f"  Saved {name} -> {path}")

    # ----------------------------------------------------------------
    # STEP 6: Train attack-type classifier
    # ----------------------------------------------------------------
    logger.info("\n[STEP 6] Training attack-type classifier (RandomForest)...")
    clf = train_web_attack_classifier(embeddings_scaled, labels_np, attack_types)
    if clf is not None:
        clf_path = str(Path(MODELS_DIR) / "web_attack_classifier.joblib")
        joblib.dump(clf, clf_path)
        logger.info(f"  Saved attack classifier -> {clf_path}")

    # ----------------------------------------------------------------
    # STEP 7: Per-attack-type metrics
    # ----------------------------------------------------------------
    logger.info("\n[STEP 7] Per-attack-type detection rates:")
    detection_results = []
    for i in range(len(labels_np)):
        detection_results.append({
            "flow_id":              i,
            "true_label":           "attack" if labels_np[i] == 1 else "benign",
            "attack_type":          attack_types[i],
            "reconstruction_error": float(errors_np[i]),
            "ensemble_verdict":     "anomaly" if ensemble_pred[i] == 1 else "normal",
        })
    per_attack = compute_per_attack_metrics(detection_results)

    # Save detection results for ATRA offline mode
    det_results_path = str(Path(MODELS_DIR).parent / "processed" / "web_detection_results.pkl")
    with open(det_results_path, "wb") as f:
        pickle.dump(detection_results, f)

    # Save comprehensive training results summary
    training_results = {
        "anomaly_threshold":      threshold,
        "in_features":            in_features,
        "pe_dim":                 pe_dim,
        "num_records":            len(records),
        "num_benign":             int(benign_mask.sum()),
        "num_attack":             int(attack_mask.sum()),
        "benign_error_mean":      float(benign_errors.mean()),
        "benign_error_std":       float(benign_errors.std()),
        "attack_error_mean":      float(attack_errors.mean()),
        "attack_error_std":       float(attack_errors.std()),
        "metrics":                metrics,
        "per_attack_metrics":     per_attack,
        "model_path":             web_model_path,
        "scaler_path":            str(scaler_path),
    }

    results_path = str(Path(MODELS_DIR) / "web_training_results.json")
    with open(results_path, "w") as f:
        json.dump(training_results, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Anomaly Threshold : {threshold:.6f}")
    logger.info(f"  TPR               : {metrics['tpr']*100:.1f}%")
    logger.info(f"  FPR               : {metrics['fpr']*100:.1f}%")
    logger.info(f"  F1                : {metrics['f1']:.3f}")
    logger.info(f"  Precision         : {metrics['precision']*100:.1f}%")
    logger.info("=" * 70)

    return training_results


if __name__ == "__main__":
    run_training()
