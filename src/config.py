"""
src/config.py
Central Configuration File for GTAE-ATRA-NIDS (Web Security Edition)
All settings in one place -- change here to affect the entire system.
Now supports both legacy network-flow mode and web-traffic monitoring mode.
"""

import os
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent   # IDS-Project/
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DATA_DIR    = DATA_DIR    / "raw"
PROCESSED_DIR   = DATA_DIR    / "processed"
MODELS_DIR      = DATA_DIR    / "models"
LOGS_DIR        = PROJECT_ROOT / "logs"
RESULTS_DIR     = PROJECT_ROOT / "results"

# Ensure directories exist on import
for _d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# CIC-IDS2017 raw CSV files (expected in data/raw/)
CIC_FILES = {
    "benign":    "Monday-WorkingHours.pcap_ISCX.csv",
    "dos":       "Wednesday-workingHours.pcap_ISCX.csv",       # DoS / DoS Hulk / GoldenEye / slowloris
    "portscan":  "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "ddos":      "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "webattack": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "infiltrate":"Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "ftp_ssh":   "Tuesday-WorkingHours.pcap_ISCX.csv",         # FTP-Patator, SSH-Patator
    "botnet":    "Friday-WorkingHours-Morning.pcap_ISCX.csv",  # Botnet ARES
}

# ============================================================
# DATA CONFIGURATION
# ============================================================

DATA_CONFIG = {
    # Sample limits (None = use all rows)
    "max_benign":          5000,    # rows of benign traffic to use
    "max_attack_per_type": 500,     # rows per attack type

    # Train / validation split (benign only -- unsupervised training)
    "train_val_split":  0.9,        # 90% train, 10% val
    "random_seed":      42,

    # Normalization
    "normalize":        True,
    "scale_method":     "minmax",   # "minmax" or "zscore"

    # Graph grouping (flows per shared src/dst node pair)
    "group_size":       15,

    # Processed output paths
    "benign_train_path":   str(PROCESSED_DIR / "benign_train.pkl"),
    "attack_test_path":    str(PROCESSED_DIR / "attack_test.pkl"),
    "graph_data_path":     str(PROCESSED_DIR / "graph_data.pkl"),
    "gtae_results_path":   str(PROCESSED_DIR / "gtae_results.pkl"),
    "detection_results_path": str(PROCESSED_DIR / "detection_results.pkl"),
    "atra_results_path":   str(PROCESSED_DIR / "atra_results.pkl"),
}

# ============================================================
# GRAPH CONSTRUCTION CONFIGURATION
# ============================================================

GRAPH_CONFIG = {
    "pos_enc_dim": 8,          # Laplacian Positional Encoding dimension (k)
    # LPE is computed on benign-only subgraph to prevent topological leakage
}

# ============================================================
# GTAE MODEL CONFIGURATION
# ============================================================

GTAE_CONFIG = {
    # Architecture
    "d_model":    64,           # Bottleneck / embedding dimension
    "num_heads":  4,            # Transformer attention heads
    "num_layers": 3,            # Transformer encoder layers

    # Decoder hidden dims
    "decoder_hidden": [128, 256],

    # Training
    "epochs":         100,
    "learning_rate":  0.001,
    "batch_size":     None,     # None = full-batch (set int for mini-batch)
    "early_stopping_patience": 15,
    "checkpoint_every": 10,     # Save checkpoint every N epochs

    # Device
    "device": "cuda" if os.environ.get("USE_GPU") else "cpu",

    # Model save path
    "model_path": str(MODELS_DIR / "gtae_model.pt"),
}

# Alias for backward compatibility
MODEL_CONFIG = GTAE_CONFIG


# ============================================================
# ANOMALY DETECTION CONFIGURATION
# ============================================================

ANOMALY_CONFIG = {
    # Which detectors to use in the ensemble
    # Options: "IF" (Isolation Forest), "OCSVM" (One-Class SVM), "HBOS"
    "ensemble": ["IF", "OCSVM", "HBOS"],

    # Voting: "union" = flag if ANY fires (max recall)
    #         "majority" = flag if >= 2 fire (lower FPR)
    "voting": "union",

    # Contamination (expected fraction of outliers in training data)
    "contamination": 0.05,

    # Isolation Forest
    "if_n_estimators": 150,

    # One-Class SVM
    "ocsvm_kernel": "rbf",
    "ocsvm_nu":     0.05,

    # HBOS (Histogram-Based Outlier Score via pyod)
    "hbos_n_bins":  40,

    # Saved model paths
    "if_path":     str(MODELS_DIR / "iso_forest.joblib"),
    "ocsvm_path":  str(MODELS_DIR / "ocsvm.joblib"),
    "hbos_path":   str(MODELS_DIR / "hbos.joblib"),
    "scaler_path": str(MODELS_DIR / "scaler.joblib"),
}

# ============================================================
# ATTACK TYPE CLASSIFIER
# ============================================================

CLASSIFIER_CONFIG = {
    "n_estimators": 100,
    "model_path": str(MODELS_DIR / "attack_classifier.joblib"),
}

# ============================================================
# ATRA CONFIGURATION
# ============================================================

ATRA_CONFIG = {
    # Component weights (must stay proportional -- all normalized to 0-100)
    "weight_attack_type": 0.25,
    "weight_severity":    0.25,
    "weight_confidence":  0.25,
    "weight_history":     0.15,
    "weight_frequency":   0.10,

    # Attack severity scores (0-10 scale) -- legacy network + web
    "attack_severity": {
        # Legacy network attacks
        "DoS Hulk":         8,
        "DoS GoldenEye":    8,
        "DoS slowloris":    7,
        "DoS Slowhttptest": 7,
        "Heartbleed":       9,
        "DDoS":             9,
        "PortScan":         4,
        "FTP-Patator":      6,
        "SSH-Patator":      6,
        "Bot":              8,
        "Web Attack":       7,
        "Infiltration":     9,
        "None":             0,
        "Unknown":          6,
        # Web-specific attack types
        "BruteForce":       8,
        "SQLInjection":     9,
        "XSS":              7,
        "PathTraversal":    8,
        "RateLimitAbuse":   7,
        "SensitiveEndpoint":8,
        "BotScan":          6,
        "AbnormalMethod":   4,
        "LargePayload":     4,
    },

    # Attack type weights (0-10 scale) -- legacy network + web
    "attack_type_weight": {
        # Legacy network attacks
        "DoS Hulk":         7,
        "DoS GoldenEye":    7,
        "DoS slowloris":    6,
        "DoS Slowhttptest": 6,
        "Heartbleed":       9,
        "DDoS":             9,
        "PortScan":         3,
        "FTP-Patator":      5,
        "SSH-Patator":      5,
        "Bot":              8,
        "Web Attack":       6,
        "Infiltration":     9,
        "None":             0,
        "Unknown":          5,
        # Web-specific attack types
        "BruteForce":       8,
        "SQLInjection":     9,
        "XSS":              7,
        "PathTraversal":    8,
        "RateLimitAbuse":   7,
        "SensitiveEndpoint":8,
        "BotScan":          6,
        "AbnormalMethod":   4,
        "LargePayload":     4,
    },

    # Risk tier thresholds (0-100 scale)
    "risk_thresholds": {
        "Low":      0,
        "Medium":   30,
        "High":     50,
        "Critical": 70,
    },

    # Automated response actions per tier
    "response_actions": {
        "Low":      "ALLOW",
        "Medium":   "MONITOR",
        "High":     "ALERT",
        "Critical": "BLOCK",
    },

    # IP history time-decay (exponential)
    "history_half_life_seconds": 300,   # 5-minute half-life
    "frequency_window_seconds":  600,   # 10-minute sliding window

    # Blacklist TTL (seconds)
    "blacklist_ttl": 3600,              # 1 hour

    # Persistent blocklist (JSON) for real-time IP blocking
    "blocklist_json_path": str(LOGS_DIR / "blocklist.json"),

    # Synthetic timestamp spacing for offline runs
    "timestamp_spacing_seconds": 2,
    "base_timestamp": "2026-08-13 15:00:00",

    # Output paths
    "attack_log_path":  str(LOGS_DIR / "attack_log.csv"),
    "blacklist_path":   str(LOGS_DIR / "blacklist.txt"),
}

# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

EVALUATION_CONFIG = {
    "results_path":        str(RESULTS_DIR / "evaluation_results.json"),
    "confusion_matrix_path": str(RESULTS_DIR / "confusion_matrix.png"),
    "roc_curve_path":      str(RESULTS_DIR / "roc_curve.png"),
    "action_dist_path":    str(RESULTS_DIR / "action_distribution.png"),
    "risk_dist_path":      str(RESULTS_DIR / "risk_score_distribution.png"),
    "report_path":         str(PROJECT_ROOT / "REPORT.md"),
}

# ============================================================
# REAL-TIME MONITORING
# ============================================================

REALTIME_CONFIG = {
    "capture_window_seconds": 10,   # Capture window per inference cycle
    "interface":    None,           # None = default interface
    "packets_per_flow": 8,
    "byte_offset":  34,             # Ethernet(14) + IPv4(20)
    "bytes_per_packet": 25,
}

# ============================================================
# WEB TRAFFIC MONITORING CONFIGURATION
# ============================================================

WEB_CONFIG = {
    # Feature aggregation window (seconds) -- requests grouped per IP per window
    "window_seconds": 30,

    # Minimum requests per window to run inference (skip very sparse windows)
    "min_requests_per_window": 1,

    # Max request rate considered normal (req/min per IP)
    "normal_max_req_rate": 30,

    # Failed login threshold (count in window) before suspicion
    "failed_login_threshold": 5,

    # Endpoints considered sensitive (higher risk weight)
    "sensitive_endpoints": [
        "/admin", "/admin/", "/login", "/api/auth",
        "/api/admin", "/.env", "/config", "/wp-admin",
        "/phpmyadmin", "/api/users", "/api/keys",
    ],

    # Suspicious URL patterns (regex fragments)
    "suspicious_url_patterns": [
        r"(\.\./){2,}",          # Path traversal
        r"(select|union|insert|drop|exec)\s",  # SQL injection hints
        r"(<script|javascript:|onerror=)",      # XSS hints
        r"(passwd|shadow|etc/)",               # File inclusion
    ],

    # Feature names (must match web_feature_extractor.py output order)
    "feature_names": [
        "request_rate",
        "unique_endpoint_count",
        "failed_login_count",
        "status_4xx_rate",
        "status_5xx_rate",
        "avg_response_time_ms",
        "avg_path_length",
        "avg_query_length",
        "post_ratio",
        "get_ratio",
        "sensitive_endpoint_count",
        "repeated_endpoint_ratio",
        "suspicious_pattern_count",
        "abnormal_method_count",
        "avg_request_size",
        "max_request_rate_burst",
        "error_404_count",
        "error_403_count",
        "user_agent_entropy",
        "req_per_minute",
    ],

    # Number of features
    "feature_dim": 20,

    # Graph construction
    "pos_enc_dim": 8,               # Laplacian PE dimension (kept from original)
    "group_size": 5,                # Requests per synthetic node group

    # Web monitor inference server address (0.0.0.0 allows cloud / docker routing)
    "monitor_host": os.environ.get("MONITOR_HOST", "0.0.0.0"),
    "monitor_port": int(os.environ.get("PORT", os.environ.get("MONITOR_PORT", "8765"))),

    # WebApp address (Node.js Express)
    "webapp_host": os.environ.get("WEBAPP_HOST", "127.0.0.1"),
    "webapp_port": int(os.environ.get("WEBAPP_PORT", "3000")),

    # Data paths for web mode
    "web_benign_path":  str(PROCESSED_DIR / "web_benign_train.pkl"),
    "web_attack_path":  str(PROCESSED_DIR / "web_attack_test.pkl"),
    "web_graph_path":   str(PROCESSED_DIR / "web_graph_data.pkl"),
    "web_events_log":   str(LOGS_DIR / "web_security_events.jsonl"),
    "web_model_path":   str(MODELS_DIR / "web_gtae_model.pt"),
    "web_schema_path":  str(MODELS_DIR / "web_feature_schema.json"),

    # Training synthetic data sizes
    "num_benign_sessions": 800,
    "num_attack_sessions": 200,
}

# ============================================================
# LOGGING
# ============================================================

LOGGING_CONFIG = {
    "log_level":   "INFO",
    "log_file":    str(LOGS_DIR / "ids_system.log"),
    "max_bytes":   10_485_760,      # 10 MB
    "backup_count": 5,
}

# ============================================================
# UTILITY FUNCTION
# ============================================================

def print_config():
    """Print a concise summary of the current configuration."""
    print("=" * 70)
    print("  GTAE-ATRA-NIDS WEB SECURITY EDITION -- CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"  GTAE d_model : {GTAE_CONFIG['d_model']}")
    print(f"  GTAE heads   : {GTAE_CONFIG['num_heads']}")
    print(f"  GTAE layers  : {GTAE_CONFIG['num_layers']}")
    print(f"  Ensemble     : {ANOMALY_CONFIG['ensemble']}")
    print(f"  Feature dim  : {WEB_CONFIG['feature_dim']}")
    print(f"  Window (s)   : {WEB_CONFIG['window_seconds']}")
    print(f"  Monitor port : {WEB_CONFIG['monitor_port']}")
    print(f"  WebApp port  : {WEB_CONFIG['webapp_port']}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()

