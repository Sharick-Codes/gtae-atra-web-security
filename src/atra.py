
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
import numpy as np
import joblib
import math
from datetime import datetime, timedelta
import os

from config import ATRA_CONFIG, CLASSIFIER_CONFIG, DATA_CONFIG
from utils import get_logger

logger = get_logger(__name__)

# Constants and Weights
WEIGHT_ATTACK_TYPE = 0.25
WEIGHT_SEVERITY = 0.25
WEIGHT_CONFIDENCE = 0.25
WEIGHT_HISTORY = 0.15
WEIGHT_FREQUENCY = 0.10

# Exports required by instruction
RISK_THRESHOLDS = ATRA_CONFIG["risk_thresholds"]
RESPONSE_ACTIONS = ATRA_CONFIG["response_actions"]
ATTACK_SEVERITY = ATRA_CONFIG["attack_severity"]
ATTACK_TYPE_WEIGHT = ATRA_CONFIG["attack_type_weight"]

class IPHistoryTracker:
    """
    Tracks past attack timestamps per source IP, and computes a
    time-decayed 'history score' and 'frequency score' on demand.
    """
    def __init__(self, half_life_seconds: float = 300):
        self.history = {}
        self.half_life = half_life_seconds
        self.decay_rate = math.log(2) / half_life_seconds if half_life_seconds > 0 else 0

    def get_decayed_history_score(self, ip: str, current_time: datetime) -> float:
        """Sum of exp-decayed weights of all past attacks from this IP."""
        if ip not in self.history:
            return 0.0
        score = 0.0
        for past_time in self.history[ip]:
            dt_seconds = (current_time - past_time).total_seconds()
            if dt_seconds < 0:
                continue
            score += math.exp(-self.decay_rate * dt_seconds)
        return score

    def get_frequency_last_window(self, ip: str, current_time: datetime, window_seconds: float = 600) -> int:
        """Raw count of attacks from this IP within the last `window_seconds`."""
        if ip not in self.history:
            return 0
        return sum(
            1 for t in self.history[ip]
            if 0 <= (current_time - t).total_seconds() <= window_seconds
        )

    def record_attack(self, ip: str, timestamp: datetime):
        """Record an attack timestamp for an IP."""
        self.history.setdefault(ip, []).append(timestamp)

def normalize(value: float, min_val: float, max_val: float, out_max: float = 10.0) -> float:
    """Normalize a value to a 0-out_max scale."""
    if max_val - min_val == 0:
        return 0.0
    scaled = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, scaled)) * out_max

def compute_risk_score(attack_weight: float, severity: float, confidence: float,
                        history_score: float, frequency_score: float) -> float:
    """
    All components normalized to 0-10 scale before weighting.
    Final score scaled to 0-100.
    """
    raw = (
        WEIGHT_ATTACK_TYPE * attack_weight +
        WEIGHT_SEVERITY * severity +
        WEIGHT_CONFIDENCE * confidence +
        WEIGHT_HISTORY * history_score +
        WEIGHT_FREQUENCY * frequency_score
    )
    max_possible = (WEIGHT_ATTACK_TYPE + WEIGHT_SEVERITY + WEIGHT_CONFIDENCE +
                     WEIGHT_HISTORY + WEIGHT_FREQUENCY) * 10
    return (raw / max_possible) * 100

def get_risk_level(score: float) -> str:
    """Map risk score to categorical risk level."""
    if score >= RISK_THRESHOLDS.get("Critical", 80):
        return "Critical"
    elif score >= RISK_THRESHOLDS.get("High", 60):
        return "High"
    elif score >= RISK_THRESHOLDS.get("Medium", 35):
        return "Medium"
    else:
        return "Low"

def get_blacklist() -> set:
    """
    Read currently blacklisted IPs, respecting TTL.
    Removes IPs that have expired based on ATRA_CONFIG['blacklist_ttl'].
    """
    blacklist_path = ATRA_CONFIG["blacklist_path"]
    ttl = ATRA_CONFIG.get("blacklist_ttl", 3600)
    
    if not os.path.exists(blacklist_path):
        return set()
    
    blacklisted_ips = set()
    current_time = datetime.now()
    valid_lines = []
    modified = False
    
    with open(blacklist_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(',')
        if len(parts) == 2:
            ip, timestamp_str = parts
            try:
                ts = datetime.fromisoformat(timestamp_str)
                if (current_time - ts).total_seconds() <= ttl:
                    blacklisted_ips.add(ip)
                    valid_lines.append(line)
                else:
                    modified = True
            except ValueError:
                modified = True
        else:
            blacklisted_ips.add(line)
            valid_lines.append(f"{line},{current_time.isoformat()}")
            modified = True
            
    if modified:
        with open(blacklist_path, 'w') as f:
            for line in valid_lines:
                f.write(line + "\n")
                
    return blacklisted_ips

def run_atra(detection_results: list, graph_data: dict, embeddings: np.ndarray = None, mode: str = 'offline') -> list:
    """
    Run Adaptive Threat Response Algorithm.
    mode: 'offline' uses sequential synthetic timestamps, 'realtime' uses actual datetime.now()
    """
    logger.info(f"Running ATRA in {mode} mode.")
    idx_to_node = {v: k for k, v in graph_data["node_to_idx"].items()}
    src_idx_per_edge = graph_data["edge_index"][0].tolist()

    model_path = CLASSIFIER_CONFIG["model_path"]
    attack_classifier = None
    if os.path.exists(model_path):
        try:
            attack_classifier = joblib.load(model_path)
            logger.info(f"Loaded attack classifier from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load classifier from {model_path}: {e}")
    else:
        logger.warning(f"Attack classifier not found at {model_path}. Using ground-truth.")

    anomaly_errors = [d.get("reconstruction_error", 0.0) for d in detection_results
                       if d.get("ensemble_verdict") == "anomaly"]
    err_min = min(anomaly_errors) if anomaly_errors else 0.0
    err_max = max(anomaly_errors) if anomaly_errors else 1.0

    tracker = IPHistoryTracker()
    base_time = datetime(2026, 8, 13, 15, 0, 0)
    atra_results = []

    batch_preds = None
    if attack_classifier is not None and embeddings is not None:
        try:
            batch_preds = attack_classifier.predict(embeddings)
        except Exception as e:
            logger.warning(f"Batch predict failed: {e}")

    for i, d in enumerate(detection_results):
        if mode == 'realtime':
            current_time = datetime.now()
        else:
            current_time = base_time + timedelta(seconds=i * 2)

        src_idx = src_idx_per_edge[i]
        src_node = idx_to_node[src_idx]
        src_ip = src_node.split(":")[0]

        if d.get("ensemble_verdict") != "anomaly":
            atra_results.append({
                "flow_id": d.get("flow_id", i), "timestamp": current_time, "src_ip": src_ip,
                "attack_type": "None", "risk_score": 0.0, "risk_level": "None",
                "action": "No Action (Benign)", "true_label": d.get("true_label", "Benign"),
            })
            continue

        if batch_preds is not None:
            attack_type = str(batch_preds[i])
        else:
            attack_type = d.get("attack_type", "Unknown")
            if attack_type == "None":
                attack_type = "Unknown"

        attack_weight = ATTACK_TYPE_WEIGHT.get(attack_type, ATTACK_TYPE_WEIGHT.get("Unknown", 5))
        severity = ATTACK_SEVERITY.get(attack_type, ATTACK_SEVERITY.get("Unknown", 6))
        confidence = normalize(d.get("reconstruction_error", 0.0), err_min, err_max, out_max=10.0)

        history_raw = tracker.get_decayed_history_score(src_ip, current_time)
        history_score = min(history_raw, 10.0)

        freq_count = tracker.get_frequency_last_window(src_ip, current_time)
        frequency_score = min(freq_count * 2.0, 10.0)

        risk_score = compute_risk_score(attack_weight, severity, confidence,
                                          history_score, frequency_score)
        risk_level = get_risk_level(risk_score)
        action = RESPONSE_ACTIONS.get(risk_level, "Log")

        tracker.record_attack(src_ip, current_time)

        atra_results.append({
            "flow_id": d.get("flow_id", i),
            "timestamp": current_time,
            "src_ip": src_ip,
            "attack_type": attack_type,
            "confidence": round(confidence, 2),
            "severity": severity,
            "history_score": round(history_score, 2),
            "frequency_score": round(frequency_score, 2),
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "action": action,
            "true_label": d.get("true_label", "Unknown"),
        })

    out_path = DATA_CONFIG["atra_results_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(atra_results, f)
    logger.info(f"Saved {len(atra_results)} ATRA results to {out_path}")

    return atra_results

def main():
    logger.info("Running ATRA module...")
    with open(DATA_CONFIG["detection_results_path"], "rb") as f:
        detection_results = pickle.load(f)
    with open(DATA_CONFIG["graph_data_path"], "rb") as f:
        graph_data = pickle.load(f)
    with open(DATA_CONFIG["gtae_results_path"], "rb") as f:
        gtae_results = pickle.load(f)

    embeddings = gtae_results["embeddings"]
    if hasattr(embeddings, "numpy"):
        embeddings = embeddings.numpy()

    atra_results = run_atra(detection_results, graph_data, embeddings, mode="offline")

    flagged = [r for r in atra_results if r.get("risk_level") not in ("None", None)]
    print(f"\nTotal flows: {len(atra_results)} | Flows sent through ATRA: {len(flagged)}\n")

    print(f"{'Risk Level':10s} | {'Count':6s}")
    for level in ["Low", "Medium", "High", "Critical"]:
        count = sum(1 for r in flagged if r.get("risk_level") == level)
        print(f"{level:10s} | {count:6d}")

    print("\n--- Sample ATRA decisions ---")
    for r in flagged[:5]:
        print(f"IP: {r['src_ip']:16s} | Attack: {r['attack_type']:12s} | "
              f"Risk: {r['risk_score']:5.1f} ({r['risk_level']:8s}) | Action: {r['action']}")

if __name__ == "__main__":
    main()
