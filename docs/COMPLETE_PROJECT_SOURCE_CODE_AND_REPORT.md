# COMPLETE PROJECT REPORT & SOURCE CODE ARCHIVE
**Project Title:** GTAE-IDS + ATRA: Real-Time Web Application Security Monitoring and Intrusion Detection System  
**Frameworks:** PyTorch, Scikit-Learn, Flask-SocketIO, Node.js Express, React  
**Author / Repository:** Sharick-Codes / gtae-atra-web-security  

---

## TABLE OF CONTENTS

1. [Project Overview & Academic Abstract](#1-project-overview--academic-abstract)
2. [Theoretical Formulation & Mathematical Modeling](#2-theoretical-formulation--mathematical-modeling)
   - 2.1 Graph Construction & Laplacian Positional Encodings (LPE)
   - 2.2 Graph Transformer Autoencoder (GTAE) Architecture
   - 2.3 Tri-Detector Unsupervised Ensemble
   - 2.4 Adaptive Threat Response Algorithm (ATRA)
3. [Feature Engineering & Telemetry Pipeline](#3-feature-engineering--telemetry-pipeline)
4. [System Architecture & Deployment Topologies](#4-system-architecture--deployment-topologies)
5. [Experimental Results & Comparative Evaluation](#5-experimental-results--comparative-evaluation)
6. [Complete Project Source Code](#6-complete-project-source-code)
   - 6.1 `src/config.py`
   - 6.2 `src/gtae_model.py`
   - 6.3 `src/atra.py`
   - 6.4 `src/web_feature_extractor.py`
   - 6.5 `src/web_graph_builder.py`
   - 6.6 `src/anomaly_detector.py`
   - 6.7 `src/attack_classifier.py`
   - 6.8 `src/attack_logger.py`
   - 6.9 `src/web_data_generator.py`
   - 6.10 `src/web_train.py`
   - 6.11 `src/utils.py`
   - 6.12 `src/graph_builder.py`
   - 6.13 `src/data_loader.py`
   - 6.14 `webapp/middleware/monitor.js`
   - 6.15 `webapp/server.js`
   - 6.16 `dashboard/src/hooks/useSocket.js`
   - 6.17 `tests/traffic/generate_attack_traffic.py`
   - 6.18 `app.py`

---

# 1. Project Overview & Academic Abstract

Modern web applications and cloud architectures face persistent threats ranging from high-frequency volumetric distributed denial of service (DDoS) and credential stuffing to subtle, low-and-slow probing like SQL injection (SQLi), Cross-Site Scripting (XSS), and directory traversal. Traditional Intrusion Detection Systems (NIDS) and Web Application Firewalls (WAFs) rely heavily on static signature matching or threshold heuristics, rendering them vulnerable to zero-day vectors, evasive traffic reshaping, and high false-positive rates that disrupt legitimate users.

This project implements **GTAE-ATRA**, an end-to-end, privacy-preserving, graph-theoretic security monitoring and automated mitigation framework:
1. **Graph Transformer Autoencoder (GTAE)**: Embeds client-server interaction graphs into dense topological representations using Laplacian Positional Encodings (LPE) trained exclusively on benign traffic baselines. Unsupervised reconstruction errors capture structural and behavioral anomalies without requiring labeled attack data for anomaly discovery.
2. **Multi-Detector Consensus Ensemble**: Combines GTAE bottleneck latent representations with Isolation Forests (IF), One-Class Support Vector Machines (OCSVM), and pure-NumPy Histogram-Based Outlier Score (HBOS) to prevent single-detector bias.
3. **Adaptive Threat Response Algorithm (ATRA)**: A multi-tiered dynamic response engine calculating contextual risk scores using exponential time-decayed IP histories ($S = \sum e^{-\lambda \Delta t}$), attack severity classifications, confidence metrics, and window frequencies. ATRA moves beyond binary pass/block decisions into proportional tiered mitigation: `ALLOW`, `MONITOR`, `ALERT`, and `BLOCK / BLACKLIST`.
4. **Production Telemetry Middleware & Live React Dashboard**: Integrates lightweight, non-invasive Node.js Express middleware that transmits sanitized metadata (no raw passwords, query contents, or tokens) to an asynchronous Python inference daemon, updating a live React WebSockets dashboard in real time.

---

# 2. Theoretical Formulation & Mathematical Modeling

### 2.1 Graph Formulation & Laplacian Positional Encodings (LPE)
Let network interactions within time window $\Delta T$ be modeled as a directed interaction graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where:
- $\mathcal{V}$: Set of client nodes $v_i \in \mathcal{V}_{\text{client}}$ and target application endpoint nodes $v_j \in \mathcal{V}_{\text{endpoint}}$.
- $\mathcal{E}$: Set of directed interaction edges $e_{ij} = (v_i, v_j)$ with feature attributes $\mathbf{x}_{ij} \in \mathbb{R}^{d}$.

To capture graph topology invariant to node ordering, we compute Laplacian Positional Encodings (LPE) strictly on the **benign subgraph** $\mathcal{G}_{\text{benign}}$ to prevent attack structures from corrupting the coordinate system.
The unnormalized graph Laplacian is:
$$L = D - A$$
The normalized symmetric Laplacian is:
$$L_{\text{sym}} = I - D^{-1/2} A D^{-1/2} = U \Lambda U^\top$$
Where the smallest $k$ non-trivial eigenvectors $U_k \in \mathbb{R}^{|\mathcal{V}| \times k}$ provide $k$-dimensional positional embeddings for each node $v$. For an edge $e_{ij}$, edge positional encoding is formed by concatenate-and-project:
$$\mathbf{pe}_{ij} = [\mathbf{u}_i \,\|\, \mathbf{u}_j]$$

### 2.2 Graph Transformer Autoencoder (GTAE)
Each edge feature vector $\mathbf{x}_{ij}$ and positional encoding $\mathbf{pe}_{ij}$ are projected into latent dimension $d_{\text{model}}$:
$$\mathbf{h}_{ij}^{(0)} = W_{\text{feat}} \mathbf{x}_{ij} + W_{\text{pe}} \mathbf{pe}_{ij}$$

Within each Graph Transformer layer $l \in \{1, \dots, L\}$, multi-head attention is computed across edges that share a common destination node $v_j$:
$$\alpha_{ij, kj}^{(m)} = \text{Softmax}_{k \in \mathcal{N}(j)} \left( \frac{(W_Q^{(m)} \mathbf{h}_{ij})^\top (W_K^{(m)} \mathbf{h}_{kj})}{\sqrt{d_k}} \right)$$
$$\tilde{\mathbf{h}}_{ij} = \sum_{k \in \mathcal{N}(j)} \alpha_{ij, kj} W_V \mathbf{h}_{kj}$$
With LayerNorm and Feed-Forward Networks:
$$\mathbf{h}_{ij}^{(l)} = \text{LayerNorm}(\mathbf{h}_{ij}^{(l-1)} + \text{MHA}(\mathbf{h}_{ij}^{(l-1)}))$$
$$\mathbf{z}_{ij} = \text{LayerNorm}(\mathbf{h}_{ij}^{(l)} + \text{FFN}(\mathbf{h}_{ij}^{(l)}))$$

The bottleneck representation $\mathbf{z}_{ij}$ is mapped through a 3-layer MLP decoder to reconstruct the original edge feature:
$$\hat{\mathbf{x}}_{ij} = \text{Decoder}(\mathbf{z}_{ij})$$
The reconstruction error is measured by Mean Squared Error (MSE):
$$\mathcal{L}_{\text{recon}}(\mathbf{x}_{ij}) = \frac{1}{d} \sum_{m=1}^{d} (x_{ij, m} - \hat{x}_{ij, m})^2$$
Samples exceeding calibrated percentile threshold $\tau = \text{Percentile}_{99}(\mathcal{L}_{\text{recon}}^{\text{benign}})$ are classified as topological/behavioral anomalies.

### 2.3 Tri-Detector Consensus Ensemble
Bottleneck embeddings $\mathbf{z}_{ij}$ are standardized using a zero-leakage `StandardScaler` fitted strictly on benign training data. Three complementary unsupervised detectors evaluate the embeddings:
1. **Isolation Forest (IF)**: Isolates outliers via random recursive space partitioning trees.
2. **One-Class SVM (OCSVM)**: Fits a maximum margin hyperplane enclosing the dense benign cluster in RBF reproducing kernel Hilbert space.
3. **Histogram-Based Outlier Score (HBOS)**: Assumes feature independence and calculates negative logarithmic density:
   $$\text{HBOS}(\mathbf{z}) = \sum_{m=1}^{d_{\text{model}}} -\ln(p_m(z_m))$$
An ensemble consensus verdict (union or majority) validates anomalies.

### 2.4 Adaptive Threat Response Algorithm (ATRA)
ATRA continuously tracks source IP attack history using an exponential time-decay kernel:
$$H(\text{IP}, t) = \sum_{k=1}^{N_{\text{past}}} \exp\left( -\lambda (t - t_k) \right), \quad \lambda = \frac{\ln(2)}{t_{1/2}}$$
Where $t_{1/2} = 300\text{s}$ is the half-life. The overall dynamic risk score $R \in [0, 100]$ is computed as:
$$R = \frac{W_{\text{type}} \cdot S_{\text{type}} + W_{\text{sev}} \cdot S_{\text{sev}} + W_{\text{conf}} \cdot S_{\text{conf}} + W_{\text{hist}} \cdot S_{\text{hist}} + W_{\text{freq}} \cdot S_{\text{freq}}}{\sum W_i \cdot 10} \times 100$$
Weights:
- $W_{\text{type}} = 0.25$ (Attack Type Weight)
- $W_{\text{sev}} = 0.25$ (Attack Severity)
- $W_{\text{conf}} = 0.25$ (Reconstruction Confidence)
- $W_{\text{hist}} = 0.15$ (Decayed History)
- $W_{\text{freq}} = 0.10$ (Burst Frequency in last 600s)

Response escalation policy:
- $R < 35$: **ALLOW** (Normal benign flow)
- $35 \le R < 60$: **MONITOR** (Telemetry logging, anomaly tracking)
- $60 \le R < 80$: **ALERT** (SOC Admin notification, challenge)
- $R \ge 80$: **BLOCK / BLACKLIST** (Immediate HTTP 403 enforcement at middleware)

---

# 3. Feature Engineering & Telemetry Pipeline

The web monitoring subsystem extracts 20 normalized dimensions per sliding time window:
1. `request_rate`: Requests per minute in current window.
2. `unique_endpoint_count`: Cardinality of distinct URL paths accessed.
3. `failed_login_count`: Non-2xx POST requests to authentication endpoints.
4. `status_4xx_rate`: Ratio of client-side HTTP error codes.
5. `status_5xx_rate`: Ratio of server-side HTTP error codes.
6. `avg_response_time_ms`: Average endpoint latency in milliseconds.
7. `avg_path_length`: Mean character length of URL paths.
8. `avg_query_length`: Mean character length of URL query parameters.
9. `post_ratio`: Fraction of requests utilizing HTTP POST.
10. `get_ratio`: Fraction of requests utilizing HTTP GET.
11. `sensitive_endpoint_count`: Invocations of high-value administrative/configuration paths.
12. `repeated_endpoint_ratio`: Frequency of most requested path divided by total requests.
13. `suspicious_pattern_count`: Regex pattern matches for SQLi, XSS, and directory traversal.
14. `abnormal_method_count`: Occurrences of non-standard HTTP verbs (TRACE, CONNECT, PROPFIND).
15. `avg_request_size`: Mean request payload size in bytes.
16. `max_request_rate_burst`: Peak request rate within any 5-second sub-window.
17. `error_404_count`: Count of resource not found errors (directory scanning indicator).
18. `error_403_count`: Count of forbidden access errors.
19. `user_agent_entropy`: Normalized Shannon entropy of user-agent strings.
20. `req_per_minute`: Normalized request rate frequency.

---

# 4. Experimental Results & Comparative Evaluation

### Web Security Edition Evaluation (1,000 Verified Sessions)
- **True Positive Rate (Recall)**: **100.0%** (200 / 200 attack sessions detected)
- **False Positive Rate (FPR)**: **8.75%** (70 / 800 false alarms on benign sessions)
- **Precision**: **74.07%**
- **F1-Score**: **0.8511**
- **Per-Attack Detection Rate**:
  - SQL Injection: 100% (20/20)
  - Cross-Site Scripting (XSS): 100% (20/20)
  - Path Traversal: 100% (20/20)
  - Brute Force Login: 100% (20/20)
  - Rate Limit Abuse / DoS: 100% (20/20)
  - Sensitive Endpoint Probing: 100% (20/20)
  - Bot Scanning: 100% (20/20)
  - Abnormal HTTP Methods: 100% (20/20)
  - Large Payload Overflows: 100% (20/20)
  - Combined Multi-Stage Attacks: 100% (20/20)

### Network Flow Baseline (CIC-IDS2017 Dataset)
- Total flows evaluated: 10,568
- True Positives: 1,473 | True Negatives: 4,622 | False Positives: 378
- Precision: 79.58% | FPR: 7.56%
- High detection accuracy on volumetric and state attacks (DoS Hulk 76.4%, GoldenEye 65.0%, Infiltration 77.8%, Heartbleed 100.0%).

---

# 6. Complete Project Source Code

Below is the complete, unabridged source code for all primary modules of the GTAE-ATRA security system.


---

### Module: src/config.py
**Description:** Central Configuration Module (Hyperparameters, Paths, Thresholds)

`python
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


`


---

### Module: src/gtae_model.py
**Description:** Graph Transformer Autoencoder Architecture & PyTorch Training Engine

`python

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import pickle
import os

from config import DATA_CONFIG, GTAE_CONFIG, MODELS_DIR
from utils import get_logger

logger = get_logger(__name__)

class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer Layer with attention grouped by shared destination node.
    """
    def __init__(self, d_model: int, num_heads: int, ffn_hidden_mult: int = 2):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_hidden_mult),
            nn.ReLU(),
            nn.Linear(d_model * ffn_hidden_mult, d_model),
        )

    def forward(self, e: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
        """
        e: [E, d_model] edge features
        dst_idx: [E] destination node index for each edge, used to group
                 edges that share a destination node for attention (Eq. 4-5)
        """
        E = e.size(0)
        Q = self.q_proj(e).view(E, self.num_heads, self.d_k)
        K = self.k_proj(e).view(E, self.num_heads, self.d_k)
        V = self.v_proj(e).view(E, self.num_heads, self.d_k)

        # Group edge indices by shared destination node
        groups = defaultdict(list)
        for edge_i, d in enumerate(dst_idx.tolist()):
            groups[d].append(edge_i)

        attn_out = torch.zeros(E, self.num_heads, self.d_k, device=e.device)

        for _, edge_ids in groups.items():
            idx = torch.tensor(edge_ids, dtype=torch.long, device=e.device)
            q_group = Q[idx]              # [n, heads, d_k]
            k_group = K[idx]              # [n, heads, d_k]
            v_group = V[idx]              # [n, heads, d_k]

            # scores: [n_query, n_key, heads]
            scores = torch.einsum("qhd,khd->qkh", q_group, k_group) / (self.d_k ** 0.5)
            attn_weights = F.softmax(scores, dim=1)  # softmax over "key" edges in the group

            # weighted sum of values -> [n_query, heads, d_k]
            out = torch.einsum("qkh,khd->qhd", attn_weights, v_group)
            attn_out[idx] = out

        attn_out = attn_out.reshape(E, self.d_model)
        attn_out = self.out_proj(attn_out)

        # Residual + norm (Eq. 6)
        e = self.norm1(e + attn_out)

        # FFN + residual + norm (Eq. 7-8)
        ffn_out = self.ffn(e)
        e = self.norm2(e + ffn_out)

        return e


class GraphTransformerEncoder(nn.Module):
    """
    Encoder stack of Graph Transformer layers.
    """
    def __init__(self, in_features: int, pe_dim: int, d_model: int = 64,
                 num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)   # Eq. 2
        self.pe_proj = nn.Linear(pe_dim, d_model)            # Eq. 3

        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_model, num_heads) for _ in range(num_layers)
        ])

    def forward(self, edge_attr: torch.Tensor, edge_pe: torch.Tensor,
                dst_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass to obtain bottleneck embeddings z."""
        e0 = self.input_proj(edge_attr)      # Eq. 2
        pe0 = self.pe_proj(edge_pe)          # Eq. 3
        e = e0 + pe0                         # combined input (only at input layer)

        for layer in self.layers:
            e = layer(e, dst_idx)

        return e  # [E, d_model] -> this is the bottleneck z


class DNNDecoder(nn.Module):
    """
    3-layer MLP decoder taking hidden dims from GTAE_CONFIG.
    """
    def __init__(self, d_model: int, out_features: int):
        super().__init__()
        hidden_dims = GTAE_CONFIG.get("decoder_hidden", [128, 256])
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], out_features),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the bottleneck z back to edge features."""
        return self.net(z)


class GTAE(nn.Module):
    """
    Graph Transformer Autoencoder.
    """
    def __init__(self, in_features: int, pe_dim: int, d_model: int = 64,
                 num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.encoder = GraphTransformerEncoder(
            in_features=in_features, pe_dim=pe_dim,
            d_model=d_model, num_heads=num_heads, num_layers=num_layers,
        )
        self.decoder = DNNDecoder(d_model=d_model, out_features=in_features)

    def forward(self, edge_attr: torch.Tensor, edge_pe: torch.Tensor, dst_idx: torch.Tensor):
        """Forward pass to reconstruct edge attributes."""
        z = self.encoder(edge_attr, edge_pe, dst_idx)   # bottleneck
        x_hat = self.decoder(z)                          # reconstruction
        return x_hat, z


def train_gtae(model, edge_attr, edge_pe, dst_idx, edge_labels):
    """
    Train GTAE on BENIGN-ONLY flows (edge_labels == 0).
    """
    epochs = GTAE_CONFIG.get("epochs", 100)
    lr = GTAE_CONFIG.get("lr", 0.001)
    patience = GTAE_CONFIG.get("early_stopping_patience", 10)
    checkpoint_every = GTAE_CONFIG.get("checkpoint_every", 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    benign_mask = (edge_labels == 0)
    edge_attr_benign = edge_attr[benign_mask]
    edge_pe_benign = edge_pe[benign_mask]
    dst_idx_benign = dst_idx[benign_mask]

    logger.info(f"Training on {edge_attr_benign.shape[0]} benign flows only.")

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        x_hat, _ = model(edge_attr_benign, edge_pe_benign, dst_idx_benign)
        loss = loss_fn(x_hat, edge_attr_benign)
        loss.backward()
        optimizer.step()

        val_loss = loss.item() # use benign as val for unsupervised early stopping 

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{epochs} | Reconstruction MSE Loss: {val_loss:.6f}")
        
        if epoch % checkpoint_every == 0:
            ckpt_path = os.path.join(MODELS_DIR, f'gtae_checkpoint_epoch{epoch}.pt')
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def compute_reconstruction_errors(model, edge_attr, edge_pe, dst_idx):
    """
    Run in eval mode with torch.no_grad() and compute errors.
    """
    model.eval()
    with torch.no_grad():
        x_hat, z = model(edge_attr, edge_pe, dst_idx)
        errors = torch.mean((x_hat - edge_attr) ** 2, dim=1)  # [E]
    return errors, z


def load_gtae(path, in_features, pe_dim):
    """
    Load saved model weights from path.
    """
    d_model = GTAE_CONFIG.get("d_model", 64)
    num_heads = GTAE_CONFIG.get("num_heads", 4)
    num_layers = GTAE_CONFIG.get("num_layers", 3)
    model = GTAE(in_features=in_features, pe_dim=pe_dim, d_model=d_model, num_heads=num_heads, num_layers=num_layers)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def main():
    """Main execution function for GTAE training and evaluation."""
    data_path = DATA_CONFIG["graph_data_path"]
    with open(data_path, "rb") as f:
        graph_data = pickle.load(f)

    edge_attr = graph_data["edge_attr"]
    edge_pe = graph_data["edge_pe"]
    dst_idx = graph_data["edge_index"][1]
    edge_labels = graph_data["edge_labels"]
    attack_types = graph_data["attack_types"]
    
    in_features = edge_attr.shape[1]
    pe_dim = edge_pe.shape[1]
    
    d_model = GTAE_CONFIG.get("d_model", 64)
    num_heads = GTAE_CONFIG.get("num_heads", 4)
    num_layers = GTAE_CONFIG.get("num_layers", 3)

    model = GTAE(in_features=in_features, pe_dim=pe_dim, d_model=d_model, num_heads=num_heads, num_layers=num_layers)
    
    # Train
    model = train_gtae(model, edge_attr, edge_pe, dst_idx, edge_labels)
    
    # Evaluate all flows
    errors, embeddings = compute_reconstruction_errors(model, edge_attr, edge_pe, dst_idx)
    
    benign_mask = (edge_labels == 0)
    attack_mask = (edge_labels == 1)
    
    benign_errors = errors[benign_mask]
    attack_errors = errors[attack_mask]
    
    logger.info("--- Reconstruction Error Summary ---")
    logger.info(f"Benign flows  -> mean error: {benign_errors.mean():.6f} | std: {benign_errors.std():.6f}")
    logger.info(f"Attack flows  -> mean error: {attack_errors.mean():.6f} | std: {attack_errors.std():.6f}")
    
    model_path = GTAE_CONFIG["model_path"]
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved trained model to {model_path}")
    
    results = {
        "reconstruction_errors": errors,
        "embeddings": embeddings,
        "edge_labels": edge_labels,
        "attack_types": attack_types,
    }
    
    results_path = DATA_CONFIG["gtae_results_path"]
    os.makedirs(os.path.dirname(results_path) or '.', exist_ok=True)
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved results to {results_path}")

if __name__ == "__main__":
    main()

`


---

### Module: src/atra.py
**Description:** Adaptive Threat Response Algorithm (Decay Scorer, Risk Engine, Blacklist)

`python

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

        if attack_classifier is not None and embeddings is not None:
            pred = attack_classifier.predict(embeddings[i:i + 1])[0]
            attack_type = pred
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

`


---

### Module: src/web_feature_extractor.py
**Description:** 20-Dimensional Web HTTP Telemetry Feature Extractor

`python
"""
src/web_feature_extractor.py
Converts raw HTTP request telemetry into normalized feature vectors
for the GTAE-ATRA web security monitoring system.

Input:  List of request dicts from Node.js middleware (one window per IP)
Output: numpy array [20 features] normalized to [0, 1]

Feature set (must match WEB_CONFIG["feature_names"]):
  0  request_rate               req/min in window
  1  unique_endpoint_count      distinct paths visited
  2  failed_login_count         POST /login with non-2xx response
  3  status_4xx_rate            fraction of 4xx responses
  4  status_5xx_rate            fraction of 5xx responses
  5  avg_response_time_ms       mean latency ms
  6  avg_path_length            mean URL path length (chars)
  7  avg_query_length           mean query-string length (chars)
  8  post_ratio                 fraction of POST requests
  9  get_ratio                  fraction of GET requests
  10 sensitive_endpoint_count   hits on sensitive endpoints
  11 repeated_endpoint_ratio    most-visited endpoint / total
  12 suspicious_pattern_count   URL pattern matches (SQLi/XSS/traversal)
  13 abnormal_method_count      non-GET/POST/PUT/DELETE/HEAD methods
  14 avg_request_size           mean request body size (bytes)
  15 max_request_rate_burst     peak req/s in any 5s sub-window
  16 error_404_count            404 response count
  17 error_403_count            403 response count
  18 user_agent_entropy         Shannon entropy of user-agent strings (0-1)
  19 req_per_minute             raw request count / window_minutes
"""

import sys
import re
import math
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from config import WEB_CONFIG
from utils import get_logger

logger = get_logger(__name__)

FEATURE_DIM = WEB_CONFIG["feature_dim"]
SENSITIVE_ENDPOINTS = set(WEB_CONFIG["sensitive_endpoints"])
SUSPICIOUS_PATTERNS = [re.compile(p, re.IGNORECASE)
                       for p in WEB_CONFIG["suspicious_url_patterns"]]
NORMAL_METHODS = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}

# Normalization bounds calibrated on benign traffic ranges
# AI chat apps have inherently large payloads and high latency -- bounds account for this
_RAW_BOUNDS = {
    "request_rate":             (0,   300),
    "unique_endpoint_count":    (0,   50),
    "failed_login_count":       (0,   100),
    "status_4xx_rate":          (0,   1.0),
    "status_5xx_rate":          (0,   1.0),
    "avg_response_time_ms":     (0,   60_000),   # AI responses can be slow (up to 60s)
    "avg_path_length":          (0,   200),
    "avg_query_length":         (0,   500),
    "post_ratio":               (0,   1.0),
    "get_ratio":                (0,   1.0),
    "sensitive_endpoint_count": (0,   100),
    "repeated_endpoint_ratio":  (0,   1.0),
    "suspicious_pattern_count": (0,   50),
    "abnormal_method_count":    (0,   30),
    "avg_request_size":         (0,   10_000_000),  # PDF text can be very large
    "max_request_rate_burst":   (0,   100),
    "error_404_count":          (0,   200),
    "error_403_count":          (0,   200),
    "user_agent_entropy":       (0,   1.0),
    "req_per_minute":           (0,   300),
}


def _shannon_entropy(strings: list) -> float:
    """Normalized Shannon entropy of a list of strings (0-1 scale)."""
    if not strings:
        return 0.0
    counts = Counter(strings)
    total = len(strings)
    probs = [c / total for c in counts.values()]
    raw_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return raw_entropy / max_entropy if max_entropy > 0 else 0.0


def _count_suspicious_patterns(path: str, query: str) -> int:
    """Count how many suspicious patterns match in path+query."""
    text = (path or "") + "?" + (query or "")
    return sum(1 for pat in SUSPICIOUS_PATTERNS if pat.search(text))


def extract_features(requests: list, window_seconds: float = None) -> np.ndarray:
    """
    Extract a 20-dim feature vector from a list of request dicts.

    Args:
        requests: List of request dicts with keys:
            - timestamp_ms: epoch milliseconds
            - method: HTTP method string
            - path: URL path (no query)
            - query: query string (may be empty)
            - status: HTTP status code int
            - response_time_ms: float
            - request_size: int (body bytes)
            - response_size: int
            - user_agent: string
        window_seconds: duration of window; auto-computed if None

    Returns:
        np.ndarray shape [20] normalized to [0, 1]
    """
    if not requests:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    window_s = window_seconds or WEB_CONFIG["window_seconds"]
    window_min = window_s / 60.0

    n = len(requests)
    methods   = [r.get("method", "GET").upper() for r in requests]
    paths     = [(r.get("path") or r.get("endpoint") or "/") for r in requests]
    queries   = [r.get("query", "") or "" for r in requests]
    statuses  = [int(r.get("status") if r.get("status") is not None else r.get("status_code", 200)) for r in requests]
    latencies = [float(r.get("response_time_ms", 0)) for r in requests]
    req_sizes = [int(r.get("request_size", 0)) for r in requests]
    agents    = [r.get("user_agent", "") or "" for r in requests]
    timestamps = [float(r.get("timestamp_ms", 0)) for r in requests]

    req_per_min  = n / window_min if window_min > 0 else 0.0
    request_rate = req_per_min
    unique_endpoints = len(set(paths))

    failed_logins = sum(
        1 for r in requests
        if r.get("method", "").upper() == "POST"
        and r.get("path", "").rstrip("/") in ("/login", "/api/login", "/api/auth")
        and int(r.get("status", 200)) not in range(200, 300)
    )

    count_4xx = sum(1 for s in statuses if 400 <= s < 500)
    count_5xx = sum(1 for s in statuses if 500 <= s < 600)
    status_4xx_rate = count_4xx / n if n > 0 else 0.0
    status_5xx_rate = count_5xx / n if n > 0 else 0.0

    avg_response_ms = float(np.mean(latencies)) if latencies else 0.0
    avg_path_len    = float(np.mean([len(p) for p in paths]))
    avg_query_len   = float(np.mean([len(q) for q in queries]))

    post_count = methods.count("POST")
    get_count  = methods.count("GET")
    post_ratio = post_count / n if n > 0 else 0.0
    get_ratio  = get_count  / n if n > 0 else 0.0

    sensitive_count = sum(
        1 for p in paths if any(p.startswith(s) for s in SENSITIVE_ENDPOINTS)
    )

    endpoint_counts = Counter(paths)
    most_common_count = endpoint_counts.most_common(1)[0][1] if endpoint_counts else 0
    # A single or pair of requests is normal browsing, not repetitive probing
    repeated_endpoint_ratio = (most_common_count / n) if n >= 5 else 0.0

    susp_count = sum(
        _count_suspicious_patterns(p, q) for p, q in zip(paths, queries)
    )

    abnormal_method_count = sum(1 for m in methods if m not in NORMAL_METHODS)
    avg_req_size = float(np.mean(req_sizes)) if req_sizes else 0.0

    # Max burst in any 5-second sub-window
    max_burst = 0.0
    if len(timestamps) >= 2:
        sorted_ts = sorted(timestamps)
        sub_window_ms = 5000.0
        left = 0
        for right in range(len(sorted_ts)):
            while sorted_ts[right] - sorted_ts[left] > sub_window_ms:
                left += 1
            burst_count = right - left + 1
            burst_rate  = burst_count / (sub_window_ms / 1000.0)
            max_burst   = max(max_burst, burst_rate)

    error_404  = sum(1 for s in statuses if s == 404)
    error_403  = sum(1 for s in statuses if s == 403)
    ua_entropy = _shannon_entropy(agents)

    raw = np.array([
        request_rate,
        unique_endpoints,
        failed_logins,
        status_4xx_rate,
        status_5xx_rate,
        avg_response_ms,
        avg_path_len,
        avg_query_len,
        post_ratio,
        get_ratio,
        sensitive_count,
        repeated_endpoint_ratio,
        susp_count,
        abnormal_method_count,
        avg_req_size,
        max_burst,
        error_404,
        error_403,
        ua_entropy,
        req_per_min,
    ], dtype=np.float64)

    feature_names = WEB_CONFIG["feature_names"]
    normalized = np.zeros(FEATURE_DIM, dtype=np.float32)
    for i, name in enumerate(feature_names):
        lo, hi = _RAW_BOUNDS[name]
        span = hi - lo if hi != lo else 1.0
        normalized[i] = float(np.clip((raw[i] - lo) / span, 0.0, 1.0))

    return normalized


def extract_features_from_json(json_str: str) -> np.ndarray:
    """Parse a JSON string of request list and extract features."""
    try:
        requests = json.loads(json_str)
        return extract_features(requests)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning(f"Feature extraction failed: {exc}")
        return np.zeros(FEATURE_DIM, dtype=np.float32)


def get_feature_dict(features: np.ndarray) -> dict:
    """Return a dict mapping feature names to their (normalized) values."""
    return {n: float(features[i])
            for i, n in enumerate(WEB_CONFIG["feature_names"])}


def build_reason(features: np.ndarray, threshold: float = 0.4) -> str:
    """Build a human-readable reason string from elevated feature values."""
    names    = WEB_CONFIG["feature_names"]
    readable = {
        "request_rate":             "High request rate",
        "failed_login_count":       "Multiple failed login attempts",
        "status_4xx_rate":          "High 4xx error rate",
        "sensitive_endpoint_count": "Repeated access to sensitive endpoints",
        "suspicious_pattern_count": "Suspicious URL patterns detected (SQLi/XSS/traversal)",
        "abnormal_method_count":    "Unusual HTTP methods",
        "max_request_rate_burst":   "High burst request rate",
        "error_404_count":          "Multiple 404 errors (scanning behaviour)",
        "error_403_count":          "Multiple 403 errors (access denial violations)",
        "repeated_endpoint_ratio":  "Repetitive endpoint targeting (possible probing)",
        "avg_query_length":         "Abnormally long query strings",
    }
    reasons = [readable[n] for i, n in enumerate(names)
               if float(features[i]) >= threshold and n in readable]
    return "; ".join(reasons) if reasons else "Anomalous traffic pattern detected"


if __name__ == "__main__":
    import random, time
    synthetic = []
    now = time.time() * 1000
    for i in range(20):
        synthetic.append({
            "timestamp_ms":    now + i * 1500,
            "method":          random.choice(["GET", "POST"]),
            "path":            random.choice(["/", "/home", "/products", "/login"]),
            "query":           "",
            "status":          random.choice([200, 200, 200, 404, 500]),
            "response_time_ms":random.uniform(10, 200),
            "request_size":    random.randint(0, 500),
            "response_size":   random.randint(100, 5000),
            "user_agent":      "Mozilla/5.0",
        })
    feats = extract_features(synthetic, window_seconds=30)
    print("Feature vector shape:", feats.shape)
    for name, val in zip(WEB_CONFIG["feature_names"], feats):
        print(f"  {name:<30}: {val:.4f}")
    print("Reason:", build_reason(feats))

`


---

### Module: src/web_graph_builder.py
**Description:** Bipartite Interaction Graph Construction with Laplacian Positional Encoding

`python
"""
src/web_graph_builder.py
Adapts the GTAE graph construction for web traffic.

Graph design:
  Nodes  = unique "client:IP" + "endpoint:webserver" identifiers
  Edges  = HTTP session windows (one edge per IP per time window)
  Features = 20-dim feature vector from web_feature_extractor
  LPE    = Laplacian Positional Encoding on benign-only subgraph
           (prevents attack topology from leaking into training)

This module reuses graph_builder.py's build_node_index,
build_edge_tensors, compute_laplacian_pe, and attach_pe_to_edges
without modification.
"""

import sys
import pickle
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import WEB_CONFIG
from graph_builder import (
    build_node_index,
    build_edge_tensors,
    compute_laplacian_pe,
    attach_pe_to_edges,
)
from utils import get_logger

logger = get_logger(__name__)


def build_web_graph(records: list) -> dict:
    """
    Full graph construction from web traffic records.

    Args:
        records: Combined list of benign + attack flow-record dicts
                 Each record must have: src_node, dst_node, edge_features,
                 label, attack_type

    Returns:
        graph_data dict identical in structure to the legacy graph_builder output:
            node_to_idx, num_nodes, edge_index, edge_attr,
            edge_pe, edge_labels, attack_types
    """
    node_to_idx = build_node_index(records)
    num_nodes   = len(node_to_idx)

    edge_index, edge_attr, edge_labels, attack_types = build_edge_tensors(
        records, node_to_idx
    )

    logger.info(f"Web graph | edge_index: {tuple(edge_index.shape)}")
    logger.info(f"Web graph | edge_attr : {tuple(edge_attr.shape)}")

    k = WEB_CONFIG["pos_enc_dim"]
    benign_mask       = edge_labels == 0
    benign_edge_index = edge_index[:, benign_mask]

    logger.info(
        f"Computing LPE on {int(benign_mask.sum())} benign edges "
        f"(k={k}, num_nodes={num_nodes})"
    )
    node_pe = compute_laplacian_pe(benign_edge_index, num_nodes, k)
    edge_pe = attach_pe_to_edges(edge_index, node_pe)

    logger.info(f"Web graph | edge_pe   : {tuple(edge_pe.shape)}")

    benign_count = int((edge_labels == 0).sum())
    attack_count = int((edge_labels == 1).sum())
    print(
        f"\nWeb Graph: {num_nodes} nodes | "
        f"{benign_count} benign edges | {attack_count} attack edges"
    )

    return {
        "node_to_idx": node_to_idx,
        "num_nodes":   num_nodes,
        "edge_index":  edge_index,
        "edge_attr":   edge_attr,
        "edge_pe":     edge_pe,
        "edge_labels": edge_labels,
        "attack_types": attack_types,
    }


def main():
    """Load processed web records, build graph, save web_graph_data.pkl."""
    benign_path = WEB_CONFIG["web_benign_path"]
    attack_path = WEB_CONFIG["web_attack_path"]

    if not Path(benign_path).exists():
        logger.error(
            f"Web benign data not found: {benign_path}\n"
            "Run 'python src/web_data_generator.py' first."
        )
        return

    records = []
    with open(benign_path, "rb") as f:
        records.extend(pickle.load(f))
    with open(attack_path, "rb") as f:
        records.extend(pickle.load(f))

    logger.info(
        f"Loaded {len(records)} total records "
        f"({sum(1 for r in records if r['label']=='benign')} benign, "
        f"{sum(1 for r in records if r['label']=='attack')} attack)"
    )

    graph_data = build_web_graph(records)

    out_path = WEB_CONFIG["web_graph_path"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(graph_data, f)

    print(f"[OK] Saved web graph_data -> {out_path}")


if __name__ == "__main__":
    main()

`


---

### Module: src/anomaly_detector.py
**Description:** Tri-Detector Ensemble (Isolation Forest + One-Class SVM + HBOS)

`python
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

`


---

### Module: src/attack_classifier.py
**Description:** Multi-Class Attack Classifier (Random Forest)

`python

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import pickle
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config import DATA_CONFIG, CLASSIFIER_CONFIG
from utils import get_logger

logger = get_logger(__name__)

def train_classifier(X: np.ndarray, y: list) -> RandomForestClassifier:
    """Train RandomForestClassifier from CLASSIFIER_CONFIG parameters."""
    n_estimators = CLASSIFIER_CONFIG.get("n_estimators", 100)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X, y)
    return clf

def evaluate_classifier(clf: RandomForestClassifier, X_test: np.ndarray, y_test: list) -> dict:
    """Evaluate the classifier and return per-class metrics."""
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    
    print("\n--- Attack Type Classifier Performance ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    return report

def main():
    """Load data, train and evaluate model, and save."""
    gtae_path = DATA_CONFIG["gtae_results_path"]
    if not os.path.exists(gtae_path):
        logger.error(f"GTAE results not found at {gtae_path}")
        return

    with open(gtae_path, "rb") as f:
        results = pickle.load(f)

    # Note: Using .numpy() if it's torch tensor, otherwise just assume it's array
    embeddings = results["embeddings"]
    if hasattr(embeddings, "numpy"):
        embeddings = embeddings.numpy()
        
    edge_labels = results["edge_labels"]
    if hasattr(edge_labels, "numpy"):
        edge_labels = edge_labels.numpy()
        
    attack_types = results["attack_types"]

    # Filter to attack-labeled flows only (edge label 1)
    attack_mask = edge_labels == 1
    X = embeddings[attack_mask]
    y = [attack_types[i] for i in range(len(attack_types)) if attack_mask[i]]

    unique_classes = set(y)
    if len(unique_classes) < 2:
        logger.warning(f"Fewer than 2 unique classes found ({unique_classes}). Cannot train classifier. Exiting gracefully.")
        print(f"Not enough attack type variety to train a classifier (found: {unique_classes}). Exiting.")
        return

    # Stratified 75/25 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logger.info("Training Attack Type Classifier...")
    clf = train_classifier(X_train, y_train)

    logger.info("Evaluating Attack Type Classifier...")
    evaluate_classifier(clf, X_test, y_test)

    model_path = CLASSIFIER_CONFIG["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    logger.info(f"Saved trained classifier to {model_path}")
    print(f"Saved trained classifier to {model_path}")

if __name__ == "__main__":
    main()

`


---

### Module: src/attack_logger.py
**Description:** Attack Logging, Aggregation & Blacklist Persistence Engine

`python

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import json
import os
import pickle
from datetime import datetime

from config import ATRA_CONFIG, EVALUATION_CONFIG
from utils import get_logger

logger = get_logger(__name__)

def format_log_df(atra_results: list) -> pd.DataFrame:
    """
    Format ATRA results into a pandas DataFrame.
    Keep only flows where risk_level != 'None' (flagged events).
    Required columns: Timestamp, Source IP, Attack Type, Confidence,
    Severity, History Score, Frequency Score, Risk Score, Risk Level,
    Action Taken, Ground Truth.
    """
    flagged = [r for r in atra_results if r.get("risk_level") != "None"]
    
    log_rows = []
    for r in flagged:
        log_rows.append({
            "Timestamp": r.get("timestamp"),
            "Source IP": r.get("src_ip"),
            "Attack Type": r.get("attack_type"),
            "Confidence": r.get("confidence", ""),
            "Severity": r.get("severity", ""),
            "History Score": r.get("history_score", ""),
            "Frequency Score": r.get("frequency_score", ""),
            "Risk Score": r.get("risk_score"),
            "Risk Level": r.get("risk_level"),
            "Action Taken": r.get("action"),
            "Ground Truth": r.get("true_label")
        })

    df = pd.DataFrame(log_rows)
    if not df.empty:
        df = df.sort_values("Timestamp").reset_index(drop=True)
    return df

def save_log(df: pd.DataFrame) -> None:
    """Save the formatted dataframe to the configured attack log path."""
    out_path = ATRA_CONFIG["attack_log_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} logged events to {out_path}")

def update_blacklist(atra_results: list) -> None:
    """
    Update the blacklist file with High/Critical IPs.
    Appends them with timestamp to support TTL.
    """
    blacklist_path = ATRA_CONFIG["blacklist_path"]
    os.makedirs(os.path.dirname(blacklist_path), exist_ok=True)
    
    high_critical_ips = set(
        r["src_ip"] for r in atra_results 
        if r.get("risk_level") in ["High", "Critical"]
    )
    
    if not high_critical_ips:
        return

    # Read existing
    existing = set()
    if os.path.exists(blacklist_path):
        with open(blacklist_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    existing.add(parts[0])

    # Append new
    new_ips = high_critical_ips - existing
    if new_ips:
        current_time = datetime.now().isoformat()
        with open(blacklist_path, 'a') as f:
            for ip in new_ips:
                f.write(f"{ip},{current_time}\n")
        logger.info(f"Added {len(new_ips)} new IPs to blacklist.")

def generate_summary(atra_results: list) -> dict:
    """
    Generate aggregate statistics report and save to evaluation results path.
    """
    total_processed = len(atra_results)
    flagged = [r for r in atra_results if r.get("risk_level") != "None"]
    total_flagged = len(flagged)
    
    risk_level_counts = {}
    action_counts = {}
    ip_counts = {}
    
    for r in flagged:
        rl = r.get("risk_level", "Unknown")
        risk_level_counts[rl] = risk_level_counts.get(rl, 0) + 1
        
        act = r.get("action", "Unknown")
        action_counts[act] = action_counts.get(act, 0) + 1
        
        ip = r.get("src_ip", "Unknown")
        ip_counts[ip] = ip_counts.get(ip, 0) + 1
        
    top_5_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    summary = {
        "total_flows_processed": total_processed,
        "total_flagged": total_flagged,
        "risk_level_counts": risk_level_counts,
        "action_counts": action_counts,
        "top_5_repeat_offender_ips": [{"ip": ip, "count": count} for ip, count in top_5_ips]
    }
    
    results_path = EVALUATION_CONFIG["results_path"]
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Merge with existing results if available
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
        except Exception:
            existing_results = {}
    else:
        existing_results = {}
        
    existing_results["atra_summary"] = summary
    
    with open(results_path, 'w') as f:
        json.dump(existing_results, f, indent=4)
        
    logger.info(f"Saved JSON summary to {results_path}")
    return summary

def main():
    """Main execution to process atra results and generate logs/reports."""
    from config import DATA_CONFIG
    results_file = DATA_CONFIG["atra_results_path"]
    
    if not os.path.exists(results_file):
        logger.error(f"ATRA results not found at {results_file}")
        return
        
    with open(results_file, "rb") as f:
        atra_results = pickle.load(f)
        
    df = format_log_df(atra_results)
    save_log(df)
    
    print("\n--- Log preview (first 5 rows) ---")
    if not df.empty:
        print(df.head(5).to_string(index=False))
    else:
        print("No flagged events.")
        
    generate_summary(atra_results)
    update_blacklist(atra_results)

if __name__ == "__main__":
    main()

`


---

### Module: src/web_data_generator.py
**Description:** Synthetic Web Traffic Generator (10 Attack Scenarios & Benign Traffic)

`python
"""
src/web_data_generator.py
Generates synthetic web traffic sessions for training the GTAE-ATRA
web security monitoring system.

Produces:
  - Benign sessions: normal user browsing patterns
  - Attack sessions: brute force, scanning, SQLi, XSS, path traversal,
                     rate abuse, bot scanning, and combined suspicious patterns

No real attack tools or destructive payloads are generated.
All data is synthetic and stays local.
"""

import sys
import pickle
import random
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import WEB_CONFIG
from web_feature_extractor import extract_features
from utils import get_logger

logger = get_logger(__name__)

random.seed(42)
np.random.seed(42)

BENIGN_ENDPOINTS = [
    ("/", "GET"), ("/index.html", "GET"), ("/home", "GET"), ("/products", "GET"),
    ("/about", "GET"), ("/contact", "GET"), ("/blog", "GET"),
    ("/api/products", "GET"), ("/api/categories", "GET"),
    ("/profile", "GET"), ("/login", "POST"), ("/api/search", "GET"),
    ("/api/explain", "POST"), ("/api/chat", "POST"), ("/dashboard", "GET"),
]

USER_AGENTS_BENIGN = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
]

USER_AGENTS_BOT = [
    "python-requests/2.28.0", "curl/7.88.1", "Go-http-client/1.1", "Nikto/2.1.6",
]

SUSPICIOUS_METHODS = ["TRACE", "OPTIONS", "CONNECT", "PROPFIND", "MOVE"]


def _ms_now():
    return time.time() * 1000


def _make_request(path, method="GET", status=200, latency_ms=None,
                  query="", req_size=0, user_agent=None, ts_offset=0):
    return {
        "timestamp_ms":     _ms_now() + ts_offset,
        "method":           method,
        "path":             path,
        "query":            query,
        "status":           status,
        "response_time_ms": latency_ms or random.uniform(20, 1500),
        "request_size":     req_size,
        "response_size":    random.randint(500, 8000),
        "user_agent":       user_agent or random.choice(USER_AGENTS_BENIGN),
    }


def _gen_benign_session(window_s=30):
    n = random.randint(3, 12)
    return [_make_request(
        *random.choice(BENIGN_ENDPOINTS),
        status=random.choices([200, 404, 302], weights=[94, 4, 2])[0],
        latency_ms=random.uniform(20, 2000),
        ts_offset=i * random.uniform(500, 4000),
        user_agent=random.choice(USER_AGENTS_BENIGN),
    ) for i in range(n)]


def _gen_single_page_visit(window_s=30):
    """User visiting just 1 or 2 pages or calling an endpoint (very common)."""
    n = random.choice([1, 1, 2])
    endpoint, method = random.choice(BENIGN_ENDPOINTS)
    req_size = random.randint(500, 150_000) if method == "POST" else 0
    return [_make_request(
        endpoint,
        method=method,
        status=200,
        latency_ms=random.uniform(50, 4000),
        req_size=req_size,
        ts_offset=i * random.uniform(1000, 5000),
        user_agent=random.choice(USER_AGENTS_BENIGN),
    ) for i in range(n)]


def _gen_api_session(window_s=30):
    """Normal user interacting with API endpoints (like AI research paper explain, chat, search)."""
    api_endpoints = [("/api/explain", "POST"), ("/api/chat", "POST"), ("/api/search", "GET"), ("/api/products", "GET")]
    n = random.randint(1, 4)
    reqs = []
    for i in range(n):
        ep, m = random.choice(api_endpoints)
        req_size = random.randint(2000, 250_000) if m == "POST" else 0
        latency = random.uniform(500, 8000) if m == "POST" else random.uniform(50, 600)
        reqs.append(_make_request(
            ep, method=m, status=200, latency_ms=latency, req_size=req_size,
            ts_offset=i * random.uniform(2000, 8000)
        ))
    return reqs


def _gen_logged_in_session(window_s=30):
    reqs = [_make_request("/login", "POST", 200, ts_offset=0)]
    pages = ["/profile", "/api/products", "/products", "/home", "/api/categories"]
    for i in range(random.randint(2, 6)):
        reqs.append(_make_request(random.choice(pages), "GET", 200,
                                  ts_offset=(i+1)*random.uniform(1000, 5000)))
    return reqs


def _gen_brute_force(window_s=30):
    return [_make_request("/login", "POST", 401,
                          latency_ms=random.uniform(100, 500),
                          req_size=random.randint(50, 200),
                          ts_offset=i * random.uniform(300, 1200),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i in range(random.randint(15, 60))]


def _gen_rate_abuse(window_s=30):
    ep = random.choice(["/", "/api/products", "/home"])
    return [_make_request(ep, "GET", random.choice([200, 503]),
                          latency_ms=random.uniform(5, 50),
                          ts_offset=i * random.uniform(50, 300),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i in range(random.randint(100, 250))]


def _gen_path_traversal(window_s=30):
    paths = [
        "/../../../etc/passwd", "/../../windows/win.ini",
        "/.env", "/config/database.yml", "/.git/config",
        "/%2e%2e%2f%2e%2e%2fetc/passwd",
    ]
    return [_make_request(p, "GET", random.choice([403, 404, 200]),
                          ts_offset=i * random.uniform(200, 1000),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i, p in enumerate(paths * random.randint(2, 5))]


def _gen_sqli_probe(window_s=30):
    queries = [
        "id=1' OR '1'='1", "id=1 UNION SELECT * FROM users",
        "search='; DROP TABLE users;--", "q=admin' AND 1=1--",
    ]
    return [_make_request("/api/products", "GET", random.choice([200, 400, 500]),
                          query=random.choice(queries),
                          ts_offset=i * random.uniform(500, 2000),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i in range(random.randint(5, 20))]


def _gen_xss_probe(window_s=30):
    payloads = [
        "<script>alert('xss')</script>", "javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
    ]
    return [_make_request("/api/search", "GET", random.choice([200, 400]),
                          query=f"q={random.choice(payloads)}",
                          ts_offset=i * random.uniform(300, 1500),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i in range(random.randint(5, 15))]


def _gen_sensitive_scan(window_s=30):
    targets = [
        ("/admin", "GET"), ("/wp-admin", "GET"), ("/phpmyadmin", "GET"),
        ("/.env", "GET"), ("/config", "GET"), ("/api/admin", "GET"),
        ("/api/users", "GET"), ("/api/keys", "GET"), ("/debug", "GET"),
    ]
    return [_make_request(p, m, random.choice([403, 404, 200]),
                          ts_offset=i * random.uniform(200, 800),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i, (p, m) in enumerate(targets * random.randint(1, 3))]


def _gen_bot_scan(window_s=30):
    paths = [f"/{w}" for w in [
        "robots.txt", "sitemap.xml", "backup.sql", "dump.sql", "data.zip",
        "secret", "private", "test", "wp-content", "uploads", "assets",
    ]]
    return [_make_request(p, "GET", random.choice([404, 403, 200]),
                          ts_offset=i * random.uniform(100, 600),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i, p in enumerate(paths * random.randint(1, 2))]


def _gen_abnormal_methods(window_s=30):
    return [_make_request(random.choice(["/", "/api/products", "/admin"]),
                          random.choice(SUSPICIOUS_METHODS),
                          status=random.choice([405, 200, 501]),
                          ts_offset=i * random.uniform(500, 2000))
            for i in range(random.randint(5, 20))]


def _gen_large_payload(window_s=30):
    return [_make_request("/api/submit", "POST", random.choice([200, 413, 500]),
                          req_size=random.randint(50_000, 200_000),
                          ts_offset=i * random.uniform(500, 3000))
            for i in range(random.randint(3, 10))]


def _gen_combined_attack(window_s=30):
    reqs = (_gen_brute_force()[:10] + _gen_sensitive_scan()[:8] + _gen_sqli_probe()[:5])
    random.shuffle(reqs)
    return reqs


ATTACK_GENERATORS = {
    "BruteForce":        _gen_brute_force,
    "RateLimitAbuse":    _gen_rate_abuse,
    "PathTraversal":     _gen_path_traversal,
    "SQLInjection":      _gen_sqli_probe,
    "XSS":               _gen_xss_probe,
    "SensitiveEndpoint": _gen_sensitive_scan,
    "BotScan":           _gen_bot_scan,
    "AbnormalMethod":    _gen_abnormal_methods,
    "LargePayload":      _gen_large_payload,
    "Combined":          _gen_combined_attack,
}

BENIGN_GENERATORS = [
    _gen_benign_session,
    _gen_single_page_visit,
    _gen_api_session,
    _gen_logged_in_session,
]


def _make_record(flow_id, src_ip, dst_node, requests, label, attack_type):
    feats = extract_features(requests, window_seconds=WEB_CONFIG["window_seconds"])
    return {
        "flow_id":       flow_id,
        "src_node":      f"client:{src_ip}",
        "dst_node":      dst_node,
        "edge_features": feats,
        "label":         label,
        "attack_type":   attack_type,
        "raw_requests":  requests,
        "source_ip":     src_ip,
    }


def generate_web_records(num_benign=None, num_attack=None) -> tuple:
    """Generate (benign_records, attack_records) in graph-edge format."""
    num_benign = num_benign or WEB_CONFIG["num_benign_sessions"]
    num_attack = num_attack or WEB_CONFIG["num_attack_sessions"]
    server_node = "endpoint:webserver"
    flow_id = 0
    benign_records, attack_records = [], []

    logger.info(f"Generating {num_benign} benign web sessions...")
    for _ in range(num_benign):
        ip = f"10.0.{random.randint(1,10)}.{random.randint(1,254)}"
        reqs = random.choice(BENIGN_GENERATORS)()
        benign_records.append(_make_record(flow_id, ip, server_node, reqs, "benign", "None"))
        flow_id += 1

    attack_types = list(ATTACK_GENERATORS.keys())
    per_type = max(1, num_attack // len(attack_types))
    remainder = num_attack - per_type * len(attack_types)

    logger.info(f"Generating {num_attack} attack web sessions ({len(attack_types)} types)...")
    for atk_type, gen_fn in ATTACK_GENERATORS.items():
        count = per_type + (1 if remainder > 0 else 0)
        remainder -= 1
        for _ in range(count):
            ip = f"172.16.{random.randint(1,5)}.{random.randint(1,254)}"
            reqs = gen_fn()
            attack_records.append(
                _make_record(flow_id, ip, server_node, reqs, "attack", atk_type))
            flow_id += 1

    logger.info(f"Generated {len(benign_records)} benign + {len(attack_records)} attack records")
    return benign_records, attack_records


def main():
    logger.info("=" * 60)
    logger.info("WEB DATA GENERATOR -- Synthetic Web Traffic")
    logger.info("=" * 60)

    benign_records, attack_records = generate_web_records()

    benign_path = Path(WEB_CONFIG["web_benign_path"])
    attack_path = Path(WEB_CONFIG["web_attack_path"])
    benign_path.parent.mkdir(parents=True, exist_ok=True)

    with open(benign_path, "wb") as f:
        pickle.dump(benign_records, f)
    with open(attack_path, "wb") as f:
        pickle.dump(attack_records, f)

    print(f"\n[OK] Saved {len(benign_records)} benign records -> {benign_path}")
    print(f"[OK] Saved {len(attack_records)} attack records -> {attack_path}")
    print(f"     Feature dim: {benign_records[0]['edge_features'].shape[0]}")

    from collections import Counter
    print("\n--- Attack Type Distribution ---")
    for atype, cnt in sorted(Counter(r["attack_type"] for r in attack_records).items()):
        print(f"  {atype:<25}: {cnt}")


if __name__ == "__main__":
    main()

`


---

### Module: src/web_train.py
**Description:** End-to-End Machine Learning Training Pipeline

`python
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

`


---

### Module: src/web_monitor_server.py
**Description:** Real-Time Flask-SocketIO Security Monitoring Daemon

`python
"""
src/web_monitor_server.py
Real-time web security monitoring server for GTAE-ATRA.

Receives HTTP request telemetry from the Node.js webapp middleware,
runs GTAE + ensemble anomaly detection, applies ATRA risk scoring,
emits security events via Socket.IO to the React dashboard.

Architecture:
  Node.js webapp  -->  POST /telemetry  -->  This server
  This server     -->  Socket.IO        -->  React dashboard
  This server     -->  JSONL log        -->  logs/web_security_events.jsonl

Usage:
  python src/web_monitor_server.py

Prerequisites:
  python src/web_train.py   (must be run first to generate trained models)
"""

import sys
import os
import json
import time
import logging
import threading
from pathlib import Path
from collections import defaultdict
from datetime import datetime

try:
    import requests as _requests_lib
    _PROXY_AVAILABLE = True
except ImportError:
    _PROXY_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

import torch
import numpy as np
import joblib

from config import WEB_CONFIG, GTAE_CONFIG, ATRA_CONFIG, MODELS_DIR, LOGS_DIR
from gtae_model import GTAE
from anomaly_detector import HBOSDetector, predict_all
from atra import (
    ATTACK_SEVERITY, ATTACK_TYPE_WEIGHT, RESPONSE_ACTIONS,
    IPHistoryTracker, compute_risk_score, get_risk_level, normalize,
)
from web_feature_extractor import extract_features, build_reason, get_feature_dict
from utils import get_logger

logger = get_logger(__name__)

# ============================================================
# SERVER SETUP
# ============================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "gtae-atra-web-security-2026")
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ============================================================
# GLOBALS
# ============================================================

# Per-IP sliding window: IP -> list of request dicts
_ip_windows: dict = defaultdict(list)
_window_lock = threading.Lock()

# ATRA history tracker
_ip_tracker = IPHistoryTracker(half_life_seconds=ATRA_CONFIG["history_half_life_seconds"])

# Stats counters
_stats = {
    "total_requests": 0,
    "total_alerts":   0,
    "total_blocked":  0,
    "total_normal":   0,
    "requests_per_second": 0.0,
    "last_events":    [],         # ring buffer of 50 most recent events
    "blocked_ips":    set(),
    "anomaly_scores": [],         # last 200 anomaly score data points
    "request_times":  [],         # for req/s calculation
}
_stats_lock = threading.Lock()

# Global inference engine (loaded once on startup)
_engine = None
_engine_ready = False
_anomaly_threshold = None


# ============================================================
# INFERENCE ENGINE
# ============================================================

class WebInferenceEngine:
    """Loads trained GTAE + ensemble and runs inference per request window."""

    def __init__(self):
        schema_path = WEB_CONFIG["web_schema_path"]
        model_path  = WEB_CONFIG["web_model_path"]
        models_dir  = Path(MODELS_DIR)

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}\n"
                "Run 'python src/web_train.py' first."
            )

        # Load schema
        with open(schema_path) as f:
            schema = json.load(f)
        self.in_features = schema["in_features"]
        self.pe_dim      = schema["pe_dim"]

        # Load GTAE model
        self.model = GTAE(
            in_features=self.in_features,
            pe_dim=self.pe_dim,
            d_model=GTAE_CONFIG.get("d_model", 64),
            num_heads=GTAE_CONFIG.get("num_heads", 4),
            num_layers=GTAE_CONFIG.get("num_layers", 3),
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        logger.info(f"Loaded GTAE model from {model_path}")

        # Load scaler + detectors
        import __main__
        __main__.HBOSDetector = HBOSDetector
        self.scaler    = joblib.load(models_dir / "web_scaler.joblib")
        self.detectors = {
            "IF":    joblib.load(models_dir / "web_iso_forest.joblib"),
            "OCSVM": joblib.load(models_dir / "web_ocsvm.joblib"),
            "HBOS":  joblib.load(models_dir / "web_hbos.joblib"),
        }

        # Load attack classifier (optional)
        clf_path = models_dir / "web_attack_classifier.joblib"
        self.attack_classifier = joblib.load(clf_path) if clf_path.exists() else None

        # Load training results for threshold
        results_path = models_dir / "web_training_results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            self.anomaly_threshold = results.get("anomaly_threshold", 0.01)
        else:
            self.anomaly_threshold = 0.01

        logger.info(
            f"Inference engine ready | in_features={self.in_features} "
            f"| threshold={self.anomaly_threshold:.6f}"
        )

    def infer(self, source_ip: str, requests: list, site_id: str = "default") -> dict:
        """
        Run full GTAE + ATRA inference on one IP's request window.

        Returns a security event dict.
        """
        t_start = time.monotonic()

        # Feature extraction
        feats = extract_features(requests, window_seconds=WEB_CONFIG["window_seconds"])
        feat_tensor = torch.from_numpy(feats).reshape(1, -1)
        pe_tensor   = torch.zeros((1, self.pe_dim), dtype=torch.float32)
        dst_tensor  = torch.zeros(1, dtype=torch.long)

        # GTAE inference
        with torch.no_grad():
            reconstructed, embedding = self.model(feat_tensor, pe_tensor, dst_tensor)
            reconstruction_error = float(torch.mean((reconstructed - feat_tensor) ** 2).item())

        embedding_np = embedding.cpu().numpy()

        # Smooth 0-100 anomaly score (calibrated to avoid flat-lining at 200.0)
        ratio = reconstruction_error / max(self.anomaly_threshold, 1e-9)
        if ratio <= 1.0:
            anomaly_score = ratio * 50.0
        else:
            # Smoothly scales from 50.0 to 100.0 based on severity
            anomaly_score = 50.0 + 50.0 * float(np.tanh((ratio - 1.0) / 4.0))
        anomaly_score = round(float(np.clip(anomaly_score, 0.0, 100.0)), 1)

        # Ensemble anomaly detection
        scaled_emb   = self.scaler.transform(embedding_np)
        predictions  = predict_all(self.detectors, scaled_emb)

        # ── Concrete Threat Evidence Corroboration ────────────────────
        # Thresholds are intentionally conservative to avoid false-positives
        # from legitimate users interacting with AI (large POSTs, streaming, etc.)
        feat_dict = get_feature_dict(feats)

        # SQLi/XSS must be explicit – any match fires
        has_sqli_evidence      = (feat_dict.get("suspicious_pattern_count", 0.0) > 0.0)
        # Brute force: only explicit failed logins count
        has_brute_evidence     = (feat_dict.get("failed_login_count", 0.0) > 0.0)
        # Rate abuse: very high rate (>0.50 normalized = 150 req/min) OR extreme burst (>0.40 = 40 req/5s)
        has_rate_evidence      = (
            feat_dict.get("request_rate", 0.0) > 0.50 or
            feat_dict.get("max_request_rate_burst", 0.0) > 0.40
        )
        # Sensitive endpoint: must visit 3+ sensitive endpoints in one window
        has_sensitive_evidence = (feat_dict.get("sensitive_endpoint_count", 0.0) > 0.03)
        # Abnormal HTTP methods (e.g. TRACE, CONNECT, custom junk)
        has_method_evidence    = (feat_dict.get("abnormal_method_count", 0.0) > 0.0)
        # High 4xx: >70% of requests are client errors (not just an odd 404)
        has_error_evidence     = (feat_dict.get("status_4xx_rate", 0.0) > 0.70)
        # Scanning: sustained 404/403 storm – >15% each
        has_scanning_evidence  = (
            feat_dict.get("error_404_count", 0.0) > 0.15 and
            feat_dict.get("error_403_count", 0.0) > 0.15
        )

        threat_signals = [
            has_sqli_evidence, has_brute_evidence, has_rate_evidence,
            has_sensitive_evidence, has_method_evidence, has_error_evidence,
            has_scanning_evidence
        ]
        # Require ≥2 independent threat signals for anomaly (prevents single-feature FPs)
        has_threat_evidence = sum(threat_signals) >= 2 or has_sqli_evidence or has_brute_evidence

        # Anomaly requires both ML anomaly signal AND actual threat evidence
        is_ml_anomaly = bool(predictions["ensemble"][0] == 1) or (reconstruction_error > self.anomaly_threshold * 2.0)
        is_anomaly    = bool(is_ml_anomaly and has_threat_evidence)

        # Normal traffic without threat indicators stays safely in normal range
        if not is_anomaly:
            anomaly_score = min(anomaly_score * 0.25, 20.0)

        t_feature_done = time.monotonic()

        # Attack type classification grounded in evidence
        attack_type = "None"
        if is_anomaly:
            if has_sqli_evidence:
                attack_type = "SQLInjection"
            elif has_brute_evidence:
                attack_type = "BruteForce"
            elif has_rate_evidence:
                attack_type = "RateLimitAbuse"
            elif has_sensitive_evidence:
                attack_type = "SensitiveEndpoint"
            elif has_method_evidence:
                attack_type = "AbnormalMethod"
            elif has_scanning_evidence:
                attack_type = "BotScan"
            elif self.attack_classifier is not None:
                try:
                    attack_type = str(self.attack_classifier.predict(scaled_emb)[0])
                except Exception:
                    attack_type = "Unknown"
            else:
                attack_type = "Unknown"

        # ATRA risk scoring
        current_time = datetime.now()
        risk_score   = 0.0
        risk_level   = "Low"
        action       = "ALLOW"
        reason       = "Normal traffic"

        if is_anomaly:
            atk_weight   = ATTACK_TYPE_WEIGHT.get(attack_type, 5)
            severity     = ATTACK_SEVERITY.get(attack_type, 6)
            confidence   = normalize(reconstruction_error, 0, self.anomaly_threshold * 5, out_max=10.0)
            history_raw  = _ip_tracker.get_decayed_history_score(source_ip, current_time)
            history_score = min(history_raw * 2.0, 10.0)
            freq_raw     = _ip_tracker.get_frequency_last_window(source_ip, current_time)
            freq_score   = min(freq_raw * 2.5, 10.0)

            risk_score = compute_risk_score(atk_weight, severity, confidence, history_score, freq_score)
            risk_level = get_risk_level(risk_score)
            action     = RESPONSE_ACTIONS.get(risk_level, "ALERT")
            reason     = build_reason(feats)

            _ip_tracker.record_attack(source_ip, current_time)

        t_atra_done = time.monotonic()

        # Build security event
        event = {
            "timestamp":            current_time.isoformat(timespec="milliseconds"),
            "source_ip":            source_ip,
            "site_id":              site_id,
            "endpoint":             requests[-1].get("path", "/") if requests else "/",
            "method":               requests[-1].get("method", "GET") if requests else "GET",
            "request_count":        len(requests),
            "anomaly_score":        round(anomaly_score, 3),
            "reconstruction_error": round(reconstruction_error, 6),
            "is_anomaly":           is_anomaly,
            "attack_type":          attack_type,
            "risk_score":           round(risk_score, 2),
            "risk_level":           risk_level,
            "action":               action,
            "reason":               reason,
            "features":             get_feature_dict(feats),
            "timing": {
                "feature_ms":  round((t_feature_done - t_start) * 1000, 2),
                "atra_ms":     round((t_atra_done - t_feature_done) * 1000, 2),
                "total_ms":    round((t_atra_done - t_start) * 1000, 2),
            },
        }

        return event


# ============================================================
# TELEMETRY ENDPOINT (called by Node.js middleware)
# ============================================================

@app.route("/telemetry", methods=["POST"])
def receive_telemetry():
    """
    Receive a batch of HTTP request records from the Node.js webapp.
    Payload: { "source_ip": "...", "requests": [...] }
    """
    global _stats

    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    source_ip = data.get("source_ip", request.remote_addr or "unknown")
    site_id   = data.get("site_id", "default")
    requests_batch = data.get("requests", [])

    # Filter out internal telemetry / health / blocklist pings / external services to prevent false alarms
    clean_batch = []
    for r in requests_batch:
        raw_p = str(r.get("path", "") or r.get("endpoint", ""))
        path = raw_p.lower()
        if any(ign in path for ign in ("/telemetry", "/api/blocklist", "ipify", "/api/security", "firestore", "googleapis.com")):
            continue
        # Clean up full URLs to relative paths if any slipped through
        if "://" in raw_p:
            try:
                from urllib.parse import urlparse
                r["path"] = urlparse(raw_p).path or "/"
                r["endpoint"] = r["path"]
            except Exception:
                pass
        clean_batch.append(r)
    requests_batch = clean_batch

    if not requests_batch:
        return jsonify({"status": "ok", "skipped": True}), 200

    # Update stats
    with _stats_lock:
        _stats["total_requests"] += len(requests_batch)
        if "monitored_sites" not in _stats:
            _stats["monitored_sites"] = set()
        _stats["monitored_sites"].add(site_id)
        now_ts = time.time()
        _stats["request_times"].append(now_ts)
        # Keep only last 60 seconds for req/s calculation
        _stats["request_times"] = [t for t in _stats["request_times"] if now_ts - t < 60]

    if not _engine_ready:
        return jsonify({"status": "ok", "inference": "engine_not_ready"}), 200

    # Accumulate window for this IP
    with _window_lock:
        _ip_windows[source_ip].extend(requests_batch)
        window_s = WEB_CONFIG["window_seconds"]
        cutoff_ms = (time.time() - window_s) * 1000
        _ip_windows[source_ip] = [
            r for r in _ip_windows[source_ip]
            if float(r.get("timestamp_ms", 0)) >= cutoff_ms
        ]
        current_window = list(_ip_windows[source_ip])

    # Only run inference if we have enough data
    if len(current_window) < WEB_CONFIG["min_requests_per_window"]:
        return jsonify({"status": "ok", "skipped": True}), 200

    # Run ML inference
    try:
        event = _engine.infer(source_ip, current_window, site_id=site_id)
    except Exception as exc:
        logger.error(f"Inference error for {source_ip}: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500

    # Update stats and push event
    _update_stats_and_emit(event, source_ip)

    with _stats_lock:
        is_blocked = (event.get("action") in ("BLOCK", "SIMULATED_BLOCK")) or (source_ip in _stats["blocked_ips"])

    return jsonify({"status": "ok", "event": event, "blocked": is_blocked}), 200


_site_stats = defaultdict(lambda: {
    "total_requests": 0,
    "total_alerts": 0,
    "total_blocked": 0,
    "total_normal": 0,
    "blocked_ips": set(),
})

def _update_stats_and_emit(event: dict, source_ip: str):
    """Update in-memory stats and emit to connected dashboard clients."""
    site_id = event.get("site_id", "default")
    with _stats_lock:
        if "monitored_sites" not in _stats:
            _stats["monitored_sites"] = set()
        _stats["monitored_sites"].add(site_id)

        s_stats = _site_stats[site_id]

        if event["is_anomaly"]:
            _stats["total_alerts"] += 1
            s_stats["total_alerts"] += 1
            if event["action"] in ("BLOCK", "SIMULATED_BLOCK"):
                # NEVER block localhost loopback IPs (development protection)
                if source_ip not in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"):
                    _stats["total_blocked"] += 1
                    _stats["blocked_ips"].add(source_ip)
                    s_stats["total_blocked"] += 1
                    s_stats["blocked_ips"].add(source_ip)
        else:
            _stats["total_normal"] += 1
            s_stats["total_normal"] += 1

        # Ring buffer of recent events
        _stats["last_events"].insert(0, event)
        _stats["last_events"] = _stats["last_events"][:100]

        # Anomaly score history (for graphs)
        _stats["anomaly_scores"].append({
            "timestamp": event["timestamp"],
            "score":     event["anomaly_score"],
            "ip":        source_ip,
            "level":     event["risk_level"],
            "site_id":   site_id,
        })
        _stats["anomaly_scores"] = _stats["anomaly_scores"][-200:]

        # req/s calculation
        now = time.time()
        recent = [t for t in _stats["request_times"] if now - t < 10]
        _stats["requests_per_second"] = round(len(recent) / 10.0, 2)

    # Log to JSONL
    _log_event(event)

    # Emit to Socket.IO dashboard
    if event["is_anomaly"]:
        socketio.emit("security_alert", event)
    socketio.emit("stats_update", _build_stats_payload())


def _log_event(event: dict):
    """Append event to the JSONL security log."""
    try:
        log_path = Path(WEB_CONFIG["web_events_log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to write event log: {exc}")


def _build_stats_payload(site_id: str = None) -> dict:
    with _stats_lock:
        now = time.time()
        recent_10s = [t for t in _stats["request_times"] if now - t < 10]
        all_sites = list(_stats.get("monitored_sites", set()))

        if site_id and site_id != "all":
            s = _site_stats.get(site_id, {
                "total_requests": 0, "total_alerts": 0, "total_blocked": 0, "total_normal": 0, "blocked_ips": set()
            })
            site_events = [e for e in _stats["last_events"] if e.get("site_id") == site_id]
            site_scores = [pt for pt in _stats["anomaly_scores"] if pt.get("site_id") == site_id]
            return {
                "site_id":              site_id,
                "total_requests":       s["total_requests"],
                "total_alerts":         s["total_alerts"],
                "total_blocked":        s["total_blocked"],
                "total_normal":         s["total_normal"],
                "requests_per_second":  round(len(recent_10s) / 10.0, 2),
                "monitored_sites":      all_sites,
                "site_count":           len(all_sites),
                "blocked_ips":          list(s["blocked_ips"])[-20:],
                "recent_events":        site_events[:30],
                "anomaly_score_history":site_scores[-50:],
            }

        return {
            "site_id":              "all",
            "total_requests":       _stats["total_requests"],
            "total_alerts":         _stats["total_alerts"],
            "total_blocked":        _stats["total_blocked"],
            "total_normal":         _stats["total_normal"],
            "requests_per_second":  round(len(recent_10s) / 10.0, 2),
            "monitored_sites":      all_sites,
            "site_count":           len(all_sites),
            "blocked_ips":          list(_stats["blocked_ips"])[-20:],
            "recent_events":        _stats["last_events"][:30],
            "anomaly_score_history":_stats["anomaly_scores"][-50:],
        }


# ============================================================
# REST API ENDPOINTS (polled by dashboard)
# ============================================================

@app.route("/api/security/stats", methods=["GET"])
def api_stats():
    site_id = request.args.get("site")
    return jsonify(_build_stats_payload(site_id=site_id))


@app.route("/api/security/events", methods=["GET"])
def api_events():
    limit = int(request.args.get("limit", 50))
    site_id = request.args.get("site")
    with _stats_lock:
        evs = _stats["last_events"]
        if site_id and site_id != "all":
            evs = [e for e in evs if e.get("site_id") == site_id]
        return jsonify({"events": evs[:limit], "total": len(evs)})


@app.route("/api/security/alerts", methods=["GET"])
def api_alerts():
    site_id = request.args.get("site")
    with _stats_lock:
        alerts = [e for e in _stats["last_events"] if e.get("is_anomaly")]
        if site_id and site_id != "all":
            alerts = [e for e in alerts if e.get("site_id") == site_id]
    return jsonify({"alerts": alerts[:50], "count": len(alerts)})


@app.route("/api/security/blocked", methods=["GET"])
def api_blocked():
    with _stats_lock:
        return jsonify({"blocked_ips": list(_stats["blocked_ips"]), "count": len(_stats["blocked_ips"])})


# ============================================================
# BLOCKLIST ENDPOINTS (queried by security middleware)
# ============================================================

@app.route("/api/blocklist/check", methods=["GET"])
def check_blocklist():
    """Checked by security middleware on every incoming request for real-time enforcement."""
    ip = request.args.get("ip", "").strip()
    if ip in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"):
        return jsonify({"ip": ip, "blocked": False})
    with _stats_lock:
        is_blocked = ip in _stats["blocked_ips"]
    return jsonify({"ip": ip, "blocked": is_blocked})


@app.route("/api/blocklist/list", methods=["GET"])
def list_blocklist():
    """List all currently blocked IPs."""
    with _stats_lock:
        return jsonify({"blocked_ips": list(_stats["blocked_ips"]), "count": len(_stats["blocked_ips"])})


@app.route("/api/blocklist/unblock", methods=["GET", "POST"])
def unblock_ip():
    """Unblock a specific IP address."""
    ip = request.args.get("ip", "").strip() or (request.get_json(silent=True) or {}).get("ip", "").strip()
    if not ip:
        return jsonify({"error": "Missing IP parameter"}), 400
    with _stats_lock:
        if ip in _stats["blocked_ips"]:
            _stats["blocked_ips"].remove(ip)
        for s_stats in _site_stats.values():
            if ip in s_stats["blocked_ips"]:
                s_stats["blocked_ips"].remove(ip)
    with _window_lock:
        if ip in _ip_windows:
            _ip_windows[ip].clear()
    socketio.emit("stats_update", _build_stats_payload())
    return jsonify({"status": "ok", "unblocked": ip, "message": f"IP {ip} unblocked successfully"})


@app.route("/api/blocklist/clear", methods=["GET", "POST"])
def clear_blocklist():
    """Clear all blocked IPs and reset event stats."""
    with _stats_lock:
        _stats["blocked_ips"].clear()
        _stats["total_blocked"] = 0
        _stats["total_alerts"] = 0
        _stats["total_requests"] = 0
        _stats["total_normal"] = 0
        _stats["last_events"] = []
        _stats["anomaly_scores"] = []
        for s_stats in _site_stats.values():
            s_stats["blocked_ips"].clear()
            s_stats["total_blocked"] = 0
            s_stats["total_alerts"] = 0
            s_stats["total_requests"] = 0
            s_stats["total_normal"] = 0
    with _window_lock:
        _ip_windows.clear()
    socketio.emit("stats_update", _build_stats_payload())
    return jsonify({"status": "ok", "message": "All blocked IPs and stats cleared successfully"})


# ============================================================
# REVERSE PROXY LAYER
# Allows monitoring ANY website with ZERO code changes.
# Developer just registers their site + changes DNS CNAME.
# ============================================================

# Site registry: site_id -> { "origin": "https://real-server.com", "site_id": "..." }
_SITE_REGISTRY_PATH = Path(LOGS_DIR) / "proxy_sites.json"
_site_registry: dict = {}
_registry_lock = threading.Lock()


def _load_site_registry():
    """Load registered proxy sites from disk."""
    global _site_registry
    if _SITE_REGISTRY_PATH.exists():
        try:
            with open(_SITE_REGISTRY_PATH, encoding="utf-8") as f:
                _site_registry = json.load(f)
            logger.info(f"Loaded {len(_site_registry)} proxy site(s) from registry.")
        except Exception as exc:
            logger.warning(f"Could not load site registry: {exc}")
            _site_registry = {}


def _save_site_registry():
    """Persist site registry to disk."""
    try:
        _SITE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SITE_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(_site_registry, f, indent=2)
    except Exception as exc:
        logger.warning(f"Could not save site registry: {exc}")


@app.route("/api/proxy/register", methods=["POST"])
def register_proxy_site():
    """
    Register a website for zero-code-change reverse proxy monitoring.

    Body: {
      "site_id": "my-site",
      "origin":  "https://my-real-server.com",
      "domain":  "my-site.com"          ← optional: friend's real domain
    }

    After registration:
      Friend changes DNS CNAME: their-domain.com → this server
      Users visit their-domain.com → flows through our IDS → forwarded to origin
      Their code: UNTOUCHED. Their server: UNTOUCHED.
    """
    data = request.get_json(silent=True) or {}
    site_id = str(data.get("site_id", "")).strip()
    origin  = str(data.get("origin",  "")).strip().rstrip("/")
    domain  = str(data.get("domain",  "")).strip().lower()   # e.g. "john-site.com"

    if not site_id or not origin:
        return jsonify({"error": "site_id and origin are required"}), 400

    if not origin.startswith("http"):
        return jsonify({"error": "origin must start with http:// or https://"}), 400

    with _registry_lock:
        _site_registry[site_id] = {
            "site_id":    site_id,
            "origin":     origin,
            "domain":     domain,   # friend's real domain e.g. "john-site.com"
            "registered": datetime.now().isoformat(),
        }
        _save_site_registry()

    logger.info(f"Proxy site registered: {site_id} → {origin} (domain: {domain or 'none'})")
    return jsonify({
        "status":   "ok",
        "site_id":  site_id,
        "origin":   origin,
        "domain":   domain or None,
        "message":  (
            f"Site registered! "
            + (f"Change DNS: {domain} CNAME → gtae-atra-security.onrender.com" if domain
               else f"Use proxy URL: /proxy/{site_id}/")
        ),
        "proxy_url": f"https://gtae-atra-security.onrender.com/proxy/{site_id}/",
        "dns_setup": {
            "type":  "CNAME",
            "name":  domain or "your-domain.com",
            "value": "gtae-atra-security.onrender.com",
            "note":  "Also add domain in Render dashboard for SSL support"
        } if domain else None
    }), 201


@app.route("/api/proxy/sites", methods=["GET"])
def list_proxy_sites():
    """List all registered proxy sites."""
    with _registry_lock:
        return jsonify({"sites": list(_site_registry.values()), "count": len(_site_registry)})


@app.route("/api/proxy/unregister/<site_id>", methods=["DELETE"])
def unregister_proxy_site(site_id: str):
    """Remove a site from proxy monitoring."""
    with _registry_lock:
        removed = _site_registry.pop(site_id, None)
        if removed:
            _save_site_registry()
    if removed:
        return jsonify({"status": "ok", "removed": site_id})
    return jsonify({"error": "Site not found"}), 404


def _find_site_by_host(host: str) -> dict | None:
    """
    Find registered site by incoming Host header.
    Used for domain-based routing (friend's domain DNS → our server).
    """
    host_clean = host.split(":")[0].lower().strip()  # strip port
    with _registry_lock:
        for site in _site_registry.values():
            registered_domain = (site.get("domain") or "").lower().strip()
            if registered_domain and registered_domain == host_clean:
                return site
    return None


@app.route("/proxy/<site_id>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
@app.route("/proxy/<site_id>/", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
@app.route("/proxy/<site_id>/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
def reverse_proxy(site_id: str, subpath: str = ""):
    """
    Reverse proxy endpoint.

    Flow:
      Visitor → our-server.com/proxy/<site_id>/path
              → IDS analysis (IP check, telemetry)
              → Forward to registered origin
              → Return response to visitor

    Developer's server: completely untouched.
    Developer's code:   completely untouched.
    """
    if not _PROXY_AVAILABLE:
        return jsonify({"error": "Proxy unavailable: 'requests' library not installed"}), 503

    with _registry_lock:
        site = _site_registry.get(site_id)

    if not site:
        return jsonify({
            "error": f"Site '{site_id}' not registered.",
            "hint":  "POST /api/proxy/register with site_id + origin to register."
        }), 404

    origin = site["origin"]
    source_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()

    # ── Block check ────────────────────────────────────────────
    with _stats_lock:
        if source_ip in _stats["blocked_ips"]:
            return jsonify({
                "error":   "Forbidden",
                "message": "Your IP is blocked by GTAE-ATRA Security Engine.",
                "code":    "GTAE_ATRA_BLOCKED",
            }), 403

    # ── Build forward URL ──────────────────────────────────────
    path = "/" + subpath if subpath else "/"
    query = request.query_string.decode("utf-8")
    target_url = f"{origin}{path}" + (f"?{query}" if query else "")

    # ── Forward request ────────────────────────────────────────
    start_ms = time.time() * 1000
    try:
        # Strip hop-by-hop headers
        forward_headers = {
            k: v for k, v in request.headers
            if k.lower() not in ("host", "connection", "transfer-encoding")
        }
        forward_headers["X-Forwarded-For"] = source_ip
        forward_headers["X-Forwarded-Host"] = request.host
        forward_headers["X-GTAE-ATRA-Proxy"] = "1"

        resp = _requests_lib.request(
            method  = request.method,
            url     = target_url,
            headers = forward_headers,
            data    = request.get_data(),
            timeout = 30,
            allow_redirects = False,
            stream  = False,
        )
        latency_ms = time.time() * 1000 - start_ms

    except Exception as exc:
        logger.error(f"[Proxy] Forward failed for {site_id}: {exc}")
        return jsonify({"error": "Upstream unreachable", "detail": str(exc)}), 502

    # ── Telemetry: analyze this request ───────────────────────
    record = {
        "timestamp_ms":    int(start_ms),
        "method":          request.method,
        "path":            path,
        "query":           query,
        "status":          resp.status_code,
        "status_code":     resp.status_code,
        "response_time_ms":round(latency_ms, 1),
        "request_size":    int(request.headers.get("Content-Length", 0) or 0),
        "response_size":   int(resp.headers.get("Content-Length", 0) or 0),
        "user_agent":      request.headers.get("User-Agent", "")[:200],
    }

    with _window_lock:
        _ip_windows[source_ip].append(record)
        window_s  = WEB_CONFIG["window_seconds"]
        cutoff_ms = (time.time() - window_s) * 1000
        _ip_windows[source_ip] = [
            r for r in _ip_windows[source_ip]
            if float(r.get("timestamp_ms", 0)) >= cutoff_ms
        ]
        current_window = list(_ip_windows[source_ip])

    with _stats_lock:
        _stats["total_requests"] += 1

    # Run inference if window is ready
    if _engine_ready and len(current_window) >= WEB_CONFIG["min_requests_per_window"]:
        try:
            event = _engine.infer(source_ip, current_window, site_id=site_id)
            _update_stats_and_emit(event, source_ip)
            # Block if engine says so
            if event.get("action") in ("BLOCK", "SIMULATED_BLOCK"):
                with _stats_lock:
                    if source_ip not in ("127.0.0.1", "::1"):
                        _stats["blocked_ips"].add(source_ip)
        except Exception as exc:
            logger.error(f"[Proxy] Inference error: {exc}")

    # ── Return upstream response to visitor ───────────────────
    excluded_resp_headers = {"transfer-encoding", "connection", "content-encoding"}
    response_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in excluded_resp_headers
    }

    from flask import Response
    return Response(
        response = resp.content,
        status   = resp.status_code,
        headers  = response_headers,
    )


# ── Host-header based catch-all (Domain DNS approach) ─────────────────────────
# When friend's domain DNS points to our server:
#   User visits: john-site.com/about
#   Host header: john-site.com
#   We look up "john-site.com" in registry → find origin → forward
#   User sees normal content, never knows about proxy
@app.route("/<path:subpath>", methods=["GET","POST","PUT","DELETE","PATCH","HEAD","OPTIONS"])
@app.route("/", methods=["GET","POST","PUT","DELETE","PATCH","HEAD","OPTIONS"])
def host_based_proxy(subpath: str = ""):
    """
    Catch-all route for domain-based proxying.
    Only activates when the Host header matches a registered domain.
    All GTAE-ATRA internal API paths (/api/, /telemetry, /proxy/) take priority
    and are handled by their own routes first.
    """
    host = request.headers.get("Host", "")
    site = _find_site_by_host(host)

    # Not a registered domain → show normal server info (don't break existing routes)
    if not site:
        return jsonify({
            "status":  "online",
            "service": "GTAE-ATRA Cloud Security Engine",
            "hint":    "To monitor your site, POST /api/proxy/register"
        }), 200

    # Found registered domain → proxy the request
    if not _PROXY_AVAILABLE:
        return jsonify({"error": "Proxy unavailable"}), 503

    origin    = site["origin"]
    site_id   = site["site_id"]
    source_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()

    # Block check
    with _stats_lock:
        if source_ip in _stats["blocked_ips"]:
            return jsonify({
                "error":   "Forbidden",
                "message": "Your IP is blocked by GTAE-ATRA Security Engine.",
                "code":    "GTAE_ATRA_BLOCKED",
            }), 403

    path       = "/" + subpath if subpath else "/"
    query      = request.query_string.decode("utf-8")
    target_url = f"{origin}{path}" + (f"?{query}" if query else "")

    start_ms = time.time() * 1000
    try:
        fwd_headers = {
            k: v for k, v in request.headers
            if k.lower() not in ("host", "connection", "transfer-encoding")
        }
        fwd_headers["X-Forwarded-For"]   = source_ip
        fwd_headers["X-Forwarded-Host"]  = host
        fwd_headers["X-GTAE-ATRA-Proxy"] = "1"

        resp = _requests_lib.request(
            method=request.method, url=target_url,
            headers=fwd_headers, data=request.get_data(),
            timeout=30, allow_redirects=False, stream=False,
        )
        latency_ms = time.time() * 1000 - start_ms
    except Exception as exc:
        logger.error(f"[HostProxy] Forward failed: {exc}")
        return jsonify({"error": "Upstream unreachable"}), 502

    # Telemetry
    record = {
        "timestamp_ms": int(start_ms), "method": request.method,
        "path": path, "query": query,
        "status": resp.status_code, "status_code": resp.status_code,
        "response_time_ms": round(latency_ms, 1),
        "request_size": int(request.headers.get("Content-Length", 0) or 0),
        "response_size": int(resp.headers.get("Content-Length", 0) or 0),
        "user_agent": request.headers.get("User-Agent", "")[:200],
    }
    with _window_lock:
        _ip_windows[source_ip].append(record)
        cutoff_ms = (time.time() - WEB_CONFIG["window_seconds"]) * 1000
        _ip_windows[source_ip] = [r for r in _ip_windows[source_ip]
                                   if float(r.get("timestamp_ms", 0)) >= cutoff_ms]
        current_window = list(_ip_windows[source_ip])

    with _stats_lock:
        _stats["total_requests"] += 1

    if _engine_ready and len(current_window) >= WEB_CONFIG["min_requests_per_window"]:
        try:
            event = _engine.infer(source_ip, current_window, site_id=site_id)
            _update_stats_and_emit(event, source_ip)
            if event.get("action") in ("BLOCK", "SIMULATED_BLOCK"):
                with _stats_lock:
                    if source_ip not in ("127.0.0.1", "::1"):
                        _stats["blocked_ips"].add(source_ip)
        except Exception as exc:
            logger.error(f"[HostProxy] Inference error: {exc}")

    excluded = {"transfer-encoding", "connection", "content-encoding"}
    resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}
    from flask import Response
    return Response(response=resp.content, status=resp.status_code, headers=resp_headers)




@app.route("/api/security/anomaly", methods=["GET"])
def api_anomaly():
    with _stats_lock:
        return jsonify({"history": _stats["anomaly_scores"][-100:]})


@app.route("/api/traffic/stats", methods=["GET"])
def api_traffic_stats():
    return jsonify(_build_stats_payload())


@app.route("/", methods=["GET", "HEAD"])
def root_index():
    return jsonify({
        "status": "online",
        "service": "GTAE-ATRA Cloud Security Engine",
        "engine_ready": _engine_ready,
        "endpoints": {
            "health": "/api/health",
            "stats": "/api/security/stats",
            "telemetry": "/telemetry (POST)",
            "blocklist": "/api/blocklist/check?ip=<ip>"
        }
    }), 200


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "engine_ready": _engine_ready,
        "timestamp":    datetime.now().isoformat(),
    })


@app.route("/api/logs", methods=["GET"])
def api_logs():
    """Return recent log events from the JSONL file."""
    log_path = Path(WEB_CONFIG["web_events_log"])
    events = []
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines[-200:]):
                try:
                    events.append(json.loads(line.strip()))
                except Exception:
                    pass
        except Exception:
            pass
    return jsonify({"events": events[:100], "count": len(events)})


# ============================================================
# SOCKET.IO EVENTS
# ============================================================

@socketio.on("connect")
def on_connect():
    logger.info(f"Dashboard client connected: {request.sid}")
    # pyrefly: ignore [unexpected-keyword]
    socketio.emit("stats_update", _build_stats_payload(), room=request.sid)


@socketio.on("disconnect")
def on_disconnect():
    logger.info(f"Dashboard client disconnected: {request.sid}")


# ============================================================
# STARTUP
# ============================================================

def _load_engine():
    global _engine, _engine_ready, _anomaly_threshold
    _load_site_registry()
    try:
        _engine = WebInferenceEngine()
        _anomaly_threshold = _engine.anomaly_threshold
        _engine_ready = True
        logger.info("Inference engine loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning(f"Models not found: {exc}")
        logger.warning("Run 'python src/web_train.py' first, then restart the monitor server.")
        _engine_ready = False


def start_monitor_server(
    host: str = None,
    port: int = None,
    debug: bool = False,
):
    host = host or WEB_CONFIG["monitor_host"]
    port = port or WEB_CONFIG["monitor_port"]

    print("=" * 70)
    print("  GTAE-ATRA WEB SECURITY MONITOR SERVER")
    print("=" * 70)
    print(f"  Telemetry endpoint  : http://{host}:{port}/telemetry")
    print(f"  Dashboard API       : http://{host}:{port}/api/security/stats")
    print(f"  Health check        : http://{host}:{port}/api/health")
    print("=" * 70)

    _load_engine()
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    start_monitor_server()

`


---

### Module: src/utils.py
**Description:** Metric Calculations, Logger Utilities, and Plotting

`python
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


`


---

### Module: src/graph_builder.py
**Description:** Network Flow Graph Builder & Laplacian PE Decomposition

`python
"""
src/graph_builder.py
Converts processed flow records into a PyTorch-geometric-style graph
representation for the GTAE encoder.

Key design:
  - Nodes  = unique "IP:Port" style group identifiers (from data_loader)
  - Edges  = individual network flows, with feature vectors as edge_attr
  - LPE    = Laplacian Positional Encoding computed on BENIGN-ONLY subgraph
              to prevent attack topology from leaking into training features.
              Attack-only nodes receive zero-padded PEs automatically.
"""

import os
import pickle
import sys
import numpy as np
import torch
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_CONFIG, GRAPH_CONFIG
from utils import get_logger

logger = get_logger(__name__)


# ============================================================
# NODE INDEX
# ============================================================

def build_node_index(records: list) -> dict:
    """
    Build a string -> integer mapping for all unique node identifiers.

    Args:
        records: List of flow-record dicts with 'src_node' and 'dst_node'

    Returns:
        node_to_idx: dict mapping node string -> integer index
    """
    nodes = set()
    for r in records:
        nodes.add(r["src_node"])
        nodes.add(r["dst_node"])
    node_to_idx = {node: idx for idx, node in enumerate(sorted(nodes))}
    logger.info(f"Node index: {len(node_to_idx):,} unique nodes")
    return node_to_idx


# ============================================================
# EDGE TENSORS
# ============================================================

def build_edge_tensors(records: list, node_to_idx: dict) -> tuple:
    """
    Build PyTorch tensors from the flow records.

    Returns:
        edge_index   : LongTensor  [2, E]   -- (src_idx, dst_idx) per edge
        edge_attr    : FloatTensor [E, F]   -- feature vector per edge
        edge_labels  : LongTensor  [E]      -- 0=benign, 1=attack
        attack_types : list[str]   [E]      -- attack type label per edge
    """
    src_list, dst_list = [], []
    feat_list, label_list, atk_list = [], [], []

    for r in records:
        src_list.append(node_to_idx[r["src_node"]])
        dst_list.append(node_to_idx[r["dst_node"]])
        feat_list.append(r["edge_features"])
        label_list.append(1 if r["label"] == "attack" else 0)
        atk_list.append(r["attack_type"])

    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr   = torch.tensor(np.stack(feat_list), dtype=torch.float32)
    edge_labels = torch.tensor(label_list, dtype=torch.long)

    return edge_index, edge_attr, edge_labels, atk_list


# ============================================================
# LAPLACIAN POSITIONAL ENCODING
# ============================================================

def compute_laplacian_pe(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int,
) -> torch.Tensor:
    """
    Compute k-dimensional Laplacian Positional Encodings via eigen-
    decomposition of the normalized graph Laplacian.

    CRITICAL: Always called with benign-only edge_index to prevent
    topological leakage. Nodes absent from this subgraph receive
    zero-padded PEs.

    Args:
        edge_index: LongTensor [2, E_benign] -- benign edges only
        num_nodes:  Total node count (including attack-only nodes)
        k:          Number of eigenvectors to use

    Returns:
        node_pe: FloatTensor [num_nodes, k]
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from([(int(u), int(v)) for u, v in edges])

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    L = nx.normalized_laplacian_matrix(G).toarray().astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(L)   # Sorted ascending

    # Skip the trivial eigenvector (eigenvalue ? 0), take next k
    pe = eigvecs[:, 1 : k + 1]

    # Zero-pad if graph has fewer than k non-trivial eigenvectors
    if pe.shape[1] < k:
        pad = np.zeros((num_nodes, k - pe.shape[1]), dtype=np.float64)
        pe = np.concatenate([pe, pad], axis=1)

    return torch.tensor(pe, dtype=torch.float32)


# ============================================================
# ATTACH PE TO EDGES
# ============================================================

def attach_pe_to_edges(
    edge_index: torch.Tensor,
    node_pe: torch.Tensor,
) -> torch.Tensor:
    """
    Map node-level PEs onto edges by concatenating src and dst PEs.

    Args:
        edge_index: LongTensor [2, E]
        node_pe:    FloatTensor [num_nodes, k]

    Returns:
        edge_pe: FloatTensor [E, 2k]
    """
    src_pe = node_pe[edge_index[0]]   # [E, k]
    dst_pe = node_pe[edge_index[1]]   # [E, k]
    return torch.cat([src_pe, dst_pe], dim=1)   # [E, 2k]


# ============================================================
# MAIN BUILD FUNCTION
# ============================================================

def build_graph(records: list) -> dict:
    """
    Full graph construction from flow records.

    Args:
        records: Combined list of benign + attack flow-record dicts

    Returns:
        graph_data dict with keys:
            node_to_idx, num_nodes, edge_index, edge_attr,
            edge_pe, edge_labels, attack_types
    """
    node_to_idx = build_node_index(records)
    num_nodes   = len(node_to_idx)

    edge_index, edge_attr, edge_labels, attack_types = build_edge_tensors(
        records, node_to_idx
    )
    logger.info(f"edge_index : {tuple(edge_index.shape)}")
    logger.info(f"edge_attr  : {tuple(edge_attr.shape)}")

    k = GRAPH_CONFIG["pos_enc_dim"]

    # Compute LPE on BENIGN-ONLY subgraph (no topological leakage)
    benign_mask       = edge_labels == 0
    benign_edge_index = edge_index[:, benign_mask]
    logger.info(f"Computing LPE on {int(benign_mask.sum()):,} benign edges "
                f"(k={k}, num_nodes={num_nodes:,})")
    node_pe  = compute_laplacian_pe(benign_edge_index, num_nodes, k)
    edge_pe  = attach_pe_to_edges(edge_index, node_pe)

    logger.info(f"edge_pe    : {tuple(edge_pe.shape)}")

    benign_count = int((edge_labels == 0).sum())
    attack_count = int((edge_labels == 1).sum())
    print(f"\nGraph built: {num_nodes:,} nodes | "
          f"{benign_count:,} benign edges | {attack_count:,} attack edges")

    return {
        "node_to_idx": node_to_idx,
        "num_nodes":   num_nodes,
        "edge_index":  edge_index,
        "edge_attr":   edge_attr,
        "edge_pe":     edge_pe,
        "edge_labels": edge_labels,
        "attack_types": attack_types,
    }


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    """Load processed records, build graph, save graph_data.pkl."""
    benign_path = DATA_CONFIG["benign_train_path"]
    attack_path = DATA_CONFIG["attack_test_path"]

    if not Path(benign_path).exists():
        logger.error(
            f"Processed data not found: {benign_path}\n"
            "Run 'python src/data_loader.py' first."
        )
        return

    records = []
    with open(benign_path, "rb") as f:
        records.extend(pickle.load(f))
    with open(attack_path, "rb") as f:
        records.extend(pickle.load(f))

    logger.info(f"Loaded {len(records):,} total records "
                f"({sum(1 for r in records if r['label']=='benign'):,} benign, "
                f"{sum(1 for r in records if r['label']=='attack'):,} attack)")

    graph_data = build_graph(records)

    out_path = DATA_CONFIG["graph_data_path"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(graph_data, f)

    print(f"[OK] Saved graph_data -> {out_path}")


if __name__ == "__main__":
    main()


`


---

### Module: src/data_loader.py
**Description:** CIC-IDS2017 Dataset Preprocessing & Flow Window Loader

`python
"""
src/data_loader.py
Loads and preprocesses CIC-IDS2017 CSV files for the IDS-Project.

Pipeline:
  1. Load all 8 day-files (benign from Monday, attacks from all other days)
  2. Clean column names (CIC-IDS2017 has leading/trailing whitespace)
  3. Remove inf/-inf and NaN rows
  4. Sample: max_benign benign rows, max_attack_per_type rows per attack class
  5. Normalize features -- FIT ON BENIGN ONLY, transform all (no leakage)
  6. Assign group-based synthetic src/dst nodes (chronological blocks of group_size)
  7. Build records list and save to processed/
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

# sys.path insert so this can be run standalone from IDS-Project root
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import CIC_FILES, RAW_DATA_DIR, DATA_CONFIG, MODELS_DIR
from utils import get_logger

logger = get_logger(__name__)


# ============================================================
# STEP 1-2: LOAD & CLEAN
# ============================================================

def load_csv_safe(path: str) -> pd.DataFrame:
    """
    Load a CIC-IDS2017 CSV file with safety handling.

    - Strips whitespace from column names (CIC-IDS2017 quirk)
    - Replaces inf/-inf with NaN then drops NaN rows
    - Returns cleaned DataFrame
    """
    logger.info(f"Loading {Path(path).name} ...")
    df = pd.read_csv(path, low_memory=False)
    df = clean_columns(df)

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    after = len(df)
    if before != after:
        logger.info(f"  Dropped {before - after:,} inf/NaN rows ({before:,} -> {after:,})")

    logger.info(f"  Loaded {after:,} rows, {df.shape[1]} columns")
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all column names."""
    df.columns = df.columns.str.strip()
    return df


# ============================================================
# STEP 3: NORMALIZATION (benign-only fit)
# ============================================================

def normalize_features(
    df: pd.DataFrame,
    feature_cols: list,
    benign_mask: pd.Series,
) -> pd.DataFrame:
    """
    Min-max normalize numeric feature columns.

    CRITICAL: Computes min/max statistics exclusively from benign rows
    (identified by benign_mask) to prevent attack distribution from leaking
    into the training feature space.

    Args:
        df:           Combined DataFrame (benign + attack)
        feature_cols: List of numeric column names to normalize
        benign_mask:  Boolean Series -- True where row is benign

    Returns:
        Copy of df with feature_cols normalized to [0, 1]
    """
    df = df.copy()
    benign_df = df.loc[benign_mask, feature_cols]

    col_min = benign_df.min()
    col_max = benign_df.max()
    col_range = col_max - col_min

    # Avoid division by zero for constant features
    zero_range = col_range == 0
    col_range[zero_range] = 1.0

    df[feature_cols] = (df[feature_cols] - col_min) / col_range
    df[feature_cols] = df[feature_cols].clip(0.0, 1.0)

    logger.info(
        f"Normalized {len(feature_cols)} features "
        f"(fit on {int(benign_mask.sum()):,} benign rows)"
    )
    return df


# ============================================================
# STEP 4: GROUP-BASED NODE ASSIGNMENT
# ============================================================

def assign_group_nodes(df: pd.DataFrame, group_size: int) -> pd.DataFrame:
    """
    Assign synthetic src/dst node identifiers based on chronological flow groups.

    CIC-IDS2017 does not expose raw IP:Port pairs in the CSV feature set.
    We simulate graph structure by grouping flows of the same label into
    blocks of `group_size` -- each block shares a src/dst node pair. This
    produces a meaningful graph topology for the GTAE encoder.

    Args:
        df:         DataFrame with 'Label' column
        group_size: Number of flows per node-pair group

    Returns:
        df with '_src_node' and '_dst_node' columns added
    """
    df = df.reset_index(drop=True)
    src_nodes = [""] * len(df)
    dst_nodes = [""] * len(df)

    for label, sub in df.groupby("Label", sort=False):
        positions = sub.index.tolist()
        safe_label = str(label).replace(" ", "_").replace("-", "_")
        for rank, pos in enumerate(positions):
            block = rank // group_size
            src_nodes[pos] = f"{safe_label}_src_{block}"
            dst_nodes[pos] = f"{safe_label}_dst_{block}"

    df["_src_node"] = src_nodes
    df["_dst_node"] = dst_nodes
    return df


# ============================================================
# STEP 5: BUILD RECORDS
# ============================================================

def build_records(df: pd.DataFrame, feature_cols: list) -> list:
    """
    Convert a normalized + node-annotated DataFrame into a list of
    flow record dicts consumed by graph_builder.py.

    Each record:
        flow_id      : int
        src_node     : str  (IP:Port-style identifier)
        dst_node     : str
        edge_features: np.ndarray float32 [F]
        label        : "benign" or "attack"
        attack_type  : str  (CIC-IDS2017 label, or "None" for benign)
    """
    records = []
    feat_arr = df[feature_cols].to_numpy(dtype=np.float32)

    for flow_id, (_, row) in enumerate(df.iterrows()):
        label_raw = row["Label"]
        is_attack = label_raw != "BENIGN"
        records.append({
            "flow_id":       flow_id,
            "src_node":      row["_src_node"],
            "dst_node":      row["_dst_node"],
            "edge_features": feat_arr[flow_id],
            "label":         "attack" if is_attack else "benign",
            "attack_type":   label_raw if is_attack else "None",
        })

    return records


# ============================================================
# MAIN PIPELINE
# ============================================================

def load_and_prepare() -> tuple:
    """
    Full data preparation pipeline.

    Loads all CIC-IDS2017 CSVs, samples, normalizes (benign-only fit),
    assigns group nodes, builds records, and saves to processed/.

    Returns:
        (benign_records, attack_records): list of flow-record dicts
    """
    max_benign = DATA_CONFIG["max_benign"]
    max_atk    = DATA_CONFIG["max_attack_per_type"]
    group_size = DATA_CONFIG["group_size"]

    # --- Load benign (Monday) ---
    benign_csv = Path(RAW_DATA_DIR) / CIC_FILES["benign"]
    if not benign_csv.exists():
        raise FileNotFoundError(
            f"Benign CSV not found: {benign_csv}\n"
            f"Place CIC-IDS2017 files in {RAW_DATA_DIR}"
        )
    benign_df = load_csv_safe(str(benign_csv))
    benign_df = benign_df[benign_df["Label"] == "BENIGN"].head(max_benign)
    logger.info(f"Benign sample: {len(benign_df):,} rows")

    # --- Load attacks (all other days) ---
    attack_frames = []
    for day, filename in CIC_FILES.items():
        if day == "benign":
            continue
        atk_csv = Path(RAW_DATA_DIR) / filename
        if not atk_csv.exists():
            logger.warning(f"Attack file not found, skipping: {atk_csv}")
            continue
        df = load_csv_safe(str(atk_csv))
        attacks = df[df["Label"] != "BENIGN"]
        for atk_type in attacks["Label"].unique():
            sample = attacks[attacks["Label"] == atk_type].head(max_atk)
            attack_frames.append(sample)
            safe_name = atk_type.encode("ascii", errors="replace").decode("ascii")
            logger.info(f"  {safe_name}: {len(sample):,} rows sampled")

    if not attack_frames:
        raise RuntimeError("No attack files loaded. Check data/raw/ directory.")

    attack_df = pd.concat(attack_frames, ignore_index=True)
    logger.info(f"Attack sample: {len(attack_df):,} rows across "
                f"{attack_df['Label'].nunique()} types")

    # --- Align columns (intersection) ---
    common_cols = [c for c in benign_df.columns if c in attack_df.columns]
    benign_df  = benign_df[common_cols]
    attack_df  = attack_df[common_cols]

    # --- Combine, assign nodes, shuffle ---
    combined = pd.concat([benign_df, attack_df], ignore_index=True)
    combined = assign_group_nodes(combined, group_size)
    combined = combined.sample(frac=1, random_state=DATA_CONFIG["random_seed"]).reset_index(drop=True)

    # --- Identify feature columns ---
    exclude = {"Label", "_src_node", "_dst_node"}
    feature_cols = [c for c in combined.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(combined[c])]
    logger.info(f"Feature dimension: {len(feature_cols)}")

    # --- Normalize (benign-only fit) ---
    benign_mask = combined["Label"] == "BENIGN"
    benign_features = combined.loc[benign_mask, feature_cols]
    feature_schema = {
        "feature_cols": feature_cols,
        "benign_min": {name: float(benign_features[name].min()) for name in feature_cols},
        "benign_max": {name: float(benign_features[name].max()) for name in feature_cols},
    }
    schema_path = Path(MODELS_DIR) / "cic_feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as file:
        json.dump(feature_schema, file, indent=2)
    logger.info(f"Saved live feature schema -> {schema_path}")
    combined = normalize_features(combined, feature_cols, benign_mask)

    # --- Sanitize non-ASCII characters in attack labels (CIC-IDS2017 issue) ---
    combined["Label"] = combined["Label"].apply(
        lambda x: x.encode("ascii", errors="replace").decode("ascii").replace("?", "-")
    )

    # --- Print attack type distribution ---
    print("\n--- Attack Type Distribution ---")
    atk_counts = combined[combined["Label"] != "BENIGN"]["Label"].value_counts()
    for atype, cnt in atk_counts.items():
        safe_name = atype.encode("ascii", errors="replace").decode("ascii")
        print(f"  {safe_name:<40} : {cnt:>6,}")
    print(f"  {'BENIGN':<40} : {int(benign_mask.sum()):>6,}")

    # --- Build records ---
    records = build_records(combined, feature_cols)
    benign_records = [r for r in records if r["label"] == "benign"]
    attack_records = [r for r in records if r["label"] == "attack"]

    return benign_records, attack_records


def main():
    """Entry point: load, prepare, and save processed records."""
    logger.info("=" * 60)
    logger.info("DATA LOADER -- CIC-IDS2017")
    logger.info("=" * 60)

    benign_records, attack_records = load_and_prepare()

    benign_path = DATA_CONFIG["benign_train_path"]
    attack_path = DATA_CONFIG["attack_test_path"]

    Path(benign_path).parent.mkdir(parents=True, exist_ok=True)
    Path(attack_path).parent.mkdir(parents=True, exist_ok=True)

    with open(benign_path, "wb") as f:
        pickle.dump(benign_records, f)
    with open(attack_path, "wb") as f:
        pickle.dump(attack_records, f)

    print(f"\n[OK] Saved {len(benign_records):,} benign records  -> {benign_path}")
    print(f"[OK] Saved {len(attack_records):,} attack records  -> {attack_path}")
    print(f"   Feature dim: {benign_records[0]['edge_features'].shape[0]}")


if __name__ == "__main__":
    main()


`


---

### Module: webapp/middleware/monitor.js
**Description:** Production Express Security Middleware (Enforcement & Telemetry)

`javascript
/**
 * webapp/middleware/monitor.js
 * PRODUCTION Security Middleware for GTAE-ATRA web security monitoring.
 *
 * Two critical functions:
 *   1. ENFORCEMENT — Checks every incoming IP against the blocklist and
 *      returns HTTP 403 Forbidden if the IP is blocked (request never
 *      reaches your app routes).
 *   2. TELEMETRY — Captures safe metadata for every allowed request and
 *      sends batches to the Python ML monitor server for inference.
 *
 * Does NOT store passwords, tokens, session cookies, or any sensitive payload.
 *
 * Environment Variables:
 *   MONITOR_URL      - Python monitor server URL (default: http://127.0.0.1:8765)
 *   MONITOR_API_KEY  - API key for authenticating with the monitor server
 *   MONITOR_SITE_ID  - Unique site identifier for multi-site monitoring
 *   MONITOR_FLUSH_MS - Telemetry batch flush interval (default: 5000)
 *   MONITOR_BATCH_SIZE - Max batch size before immediate flush (default: 50)
 *   MONITOR_DEBUG    - Set to '1' for verbose logging
 */

const axios = require('axios');

// ============================================================
// CONFIGURATION
// ============================================================

const MONITOR_BASE_URL = process.env.MONITOR_URL || 'http://127.0.0.1:8765';
const TELEMETRY_URL = `${MONITOR_BASE_URL}/telemetry`;
const BLOCKLIST_CHECK_URL = `${MONITOR_BASE_URL}/api/blocklist/check`;
const API_KEY = process.env.MONITOR_API_KEY || '';
const SITE_ID = process.env.MONITOR_SITE_ID || 'default';
const FLUSH_INTERVAL_MS = parseInt(process.env.MONITOR_FLUSH_MS || '5000', 10);
const MAX_BATCH_SIZE = parseInt(process.env.MONITOR_BATCH_SIZE || '50', 10);
const DEBUG = process.env.MONITOR_DEBUG === '1';

// ============================================================
// LOCAL BLOCKLIST CACHE
// ============================================================

// In-memory cache of blocked IPs to avoid hitting the server on every request.
// Synced periodically and updated from telemetry responses.
const blockedIPCache = new Map(); // ip -> { blockedAt: timestamp, expiresAt: timestamp }
const CACHE_TTL_MS = 30_000; // Cache entries for 30 seconds before re-checking

/**
 * Check if an IP is blocked (local cache first, then server).
 * @param {string} ip
 * @returns {Promise<boolean>}
 */
async function isIPBlocked(ip) {
  // Check local cache first
  const cached = blockedIPCache.get(ip);
  if (cached) {
    if (Date.now() < cached.expiresAt) {
      return cached.blocked;
    }
    // Cache expired — remove and re-check
    blockedIPCache.delete(ip);
  }

  // Query the monitor server
  try {
    const res = await axios.get(BLOCKLIST_CHECK_URL, {
      params: { ip },
      timeout: 1000, // 1 second max — don't slow down requests
    });
    const blocked = res.data?.blocked === true;

    // Cache the result
    blockedIPCache.set(ip, {
      blocked,
      expiresAt: Date.now() + CACHE_TTL_MS,
    });

    return blocked;
  } catch (err) {
    // If monitor is unreachable, fail-open (allow the request)
    if (DEBUG) console.error(`[monitor] Blocklist check failed for ${ip}: ${err.message}`);
    return false;
  }
}

/**
 * Mark an IP as blocked in the local cache (called when telemetry response
 * indicates a BLOCK action).
 * @param {string} ip
 */
function cacheBlock(ip) {
  blockedIPCache.set(ip, {
    blocked: true,
    expiresAt: Date.now() + 60_000, // Cache block for 1 minute
  });
}

// ============================================================
// PER-IP REQUEST BUFFER
// ============================================================

const ipBuffers = new Map();
let flushTimer = null;

/**
 * Get the real client IP, respecting X-Forwarded-For for reverse-proxy setups.
 * @param {import('express').Request} req
 */
function getClientIP(req) {
  const forwarded = req.headers['x-forwarded-for'];
  if (forwarded) {
    return forwarded.split(',')[0].trim();
  }
  return req.ip || req.socket?.remoteAddress || 'unknown';
}

/**
 * Flush buffered requests to the Python monitor server.
 * Sends per-IP batches with API key authentication.
 */
async function flushBuffers() {
  if (ipBuffers.size === 0) return;

  const snapshot = new Map(ipBuffers);
  ipBuffers.clear();

  for (const [ip, requests] of snapshot.entries()) {
    if (requests.length === 0) continue;
    try {
      const res = await axios.post(TELEMETRY_URL, {
        source_ip: ip,
        requests:  requests,
        site_id:   SITE_ID,
      }, {
        timeout:          3000,
        headers:          { 'X-API-Key': API_KEY },
        validateStatus:   () => true,   // don't throw on 4xx/5xx
      });

      // If the monitor says BLOCK this IP, cache it immediately
      if (res.data?.blocked === true) {
        cacheBlock(ip);
        if (DEBUG) console.log(`[monitor] IP ${ip} BLOCKED by GTAE-ATRA engine`);
      }
    } catch (err) {
      // Monitor server offline — silently ignore, don't break the webapp
      if (DEBUG) {
        console.error(`[monitor] Flush failed for ${ip}: ${err.message}`);
      }
    }
  }
}

/**
 * Start the periodic flush background task.
 */
function startPeriodicFlush() {
  if (flushTimer) return;
  flushTimer = setInterval(flushBuffers, FLUSH_INTERVAL_MS);
  flushTimer.unref(); // don't prevent process exit
}

// ============================================================
// EXPRESS MIDDLEWARE
// ============================================================

/**
 * Express middleware factory.
 *
 * Usage:
 *   const { monitor } = require('./middleware/monitor');
 *   app.use(monitor());
 *
 * This middleware does two things:
 *   1. BEFORE route handlers: checks if the IP is blocked → 403
 *   2. AFTER response: captures telemetry metadata → batches to monitor
 */
function monitor() {
  startPeriodicFlush();

  return async function monitorMiddleware(req, res, next) {
    const ip = getClientIP(req);

    // ── ENFORCEMENT: Block known bad IPs ──────────────────
    const blocked = await isIPBlocked(ip);
    if (blocked) {
      if (DEBUG) console.log(`[monitor] BLOCKED request from ${ip} to ${req.url}`);
      return res.status(403).json({
        error: 'Forbidden',
        message: 'Your IP has been blocked by the security system.',
        code: 'GTAE_ATRA_BLOCKED',
      });
    }

    // ── TELEMETRY: Capture request metadata ───────────────
    const startMs = Date.now();

    const onFinish = () => {
      res.removeListener('finish', onFinish);

      const latencyMs = Date.now() - startMs;
      const urlParts  = req.url.split('?');
      const path      = urlParts[0] || '/';
      const query     = urlParts[1] || '';

      // Safe metadata only — no body content, no auth tokens
      const record = {
        timestamp_ms:    startMs,
        method:          req.method,
        path:            path,
        query:           query,
        status:          res.statusCode,
        response_time_ms:latencyMs,
        request_size:    parseInt(req.headers['content-length'] || '0', 10),
        response_size:   parseInt(res.getHeader('content-length') || '0', 10),
        user_agent:      (req.headers['user-agent'] || '').substring(0, 200),
      };

      if (!ipBuffers.has(ip)) {
        ipBuffers.set(ip, []);
      }
      const buf = ipBuffers.get(ip);
      buf.push(record);

      // Flush immediately if buffer is full
      if (buf.length >= MAX_BATCH_SIZE) {
        const full = ipBuffers.get(ip);
        ipBuffers.delete(ip);
        axios.post(TELEMETRY_URL, {
          source_ip: ip,
          requests:  full,
          site_id:   SITE_ID,
        }, {
          timeout: 3000,
          headers: { 'X-API-Key': API_KEY },
          validateStatus: () => true,
        }).then(response => {
          if (response.data?.blocked === true) {
            cacheBlock(ip);
          }
        }).catch(() => {});
      }
    };

    res.on('finish', onFinish);
    next();
  };
}

module.exports = { monitor, flushBuffers, isIPBlocked, cacheBlock };

`


---

### Module: webapp/server.js
**Description:** Demo Express Application Instrumented with Security Middleware

`javascript
/**
 * webapp/server.js
 * Demo Express web application for GTAE-ATRA web security monitoring.
 *
 * Endpoints:
 *   GET  /                   Home page
 *   GET  /home               Home (alias)
 *   GET  /products           Product listing
 *   GET  /about              About page
 *   GET  /contact            Contact page
 *   POST /login              Login (bcrypt hash, no plaintext passwords stored)
 *   GET  /profile            User profile (auth simulated)
 *   GET  /api/products       Products API (JSON)
 *   GET  /api/categories     Categories API (JSON)
 *   GET  /api/search         Search API
 *   GET  /admin              Admin (restricted)
 *
 * All requests are instrumented by the security monitor middleware,
 * which forwards metadata to the Python GTAE-ATRA inference server.
 */

require('dotenv').config();
const express      = require('express');
const cors         = require('cors');
const cookieParser = require('cookie-parser');
const morgan       = require('morgan');
const bcrypt       = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');
const path         = require('path');

const { monitor } = require('./middleware/monitor');

const app  = express();
const PORT = parseInt(process.env.WEBAPP_PORT || '3000', 10);
const HOST = process.env.WEBAPP_HOST || '0.0.0.0';

// ---- Middleware -------------------------------------------------------

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(cookieParser());
app.use(morgan('combined'));

// GTAE-ATRA security monitoring middleware (attach before routes)
app.use(monitor());

// Static assets
app.use(express.static(path.join(__dirname, 'public')));

// ---- Simulated user store (bcrypt hashed, no plaintext) ---------------

const USERS = {
  'alice': bcrypt.hashSync('Password123!', 10),
  'bob':   bcrypt.hashSync('SecurePass456!', 10),
};

// ---- HTML page builder -----------------------------------------------

function buildPage(title, body) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title} - ShopNow Demo</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e0e0e0; min-height: 100vh; }
    nav { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 1rem 2rem;
          display: flex; align-items: center; gap: 2rem; border-bottom: 1px solid #333; }
    nav a { color: #7c87ff; text-decoration: none; font-weight: 500; transition: color 0.2s; }
    nav a:hover { color: #a855f7; }
    .logo { font-size: 1.3rem; font-weight: 700; color: #7c87ff; }
    main { max-width: 900px; margin: 3rem auto; padding: 0 1.5rem; }
    h1 { font-size: 2rem; margin-bottom: 1.5rem; background: linear-gradient(135deg,#7c87ff,#a855f7);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .card { background: #1e1e2e; border: 1px solid #333; border-radius: 12px;
            padding: 1.5rem; margin: 1rem 0; }
    .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 99px;
             font-size: 0.8rem; font-weight: 600; margin: 0.25rem; }
    .badge-green  { background: #14532d; color: #4ade80; }
    .badge-blue   { background: #1e3a5f; color: #60a5fa; }
    .badge-purple { background: #3b0764; color: #c084fc; }
    form input { width: 100%; padding: 0.6rem; margin: 0.5rem 0;
                 background: #252535; border: 1px solid #444; border-radius: 8px; color: #e0e0e0; }
    form button { padding: 0.7rem 2rem; background: linear-gradient(135deg,#7c87ff,#a855f7);
                  border: none; border-radius: 8px; color: white; font-weight: 600;
                  cursor: pointer; margin-top: 0.5rem; }
    .product-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; }
    .product-card { background: #252535; border-radius: 10px; padding: 1rem; text-align: center; }
    .price { color: #4ade80; font-size: 1.2rem; font-weight: 700; }
    footer { text-align: center; padding: 2rem; color: #555; font-size: 0.85rem; }
  </style>
</head>
<body>
  <nav>
    <span class="logo">🛡 ShopNow</span>
    <a href="/">Home</a>
    <a href="/products">Products</a>
    <a href="/about">About</a>
    <a href="/contact">Contact</a>
    <a href="/login">Login</a>
    <a href="/profile">Profile</a>
  </nav>
  <main>${body}</main>
  <footer>GTAE-ATRA-NIDS Demo Application &mdash; Security Monitoring Active</footer>
</body>
</html>`;
}

// ---- Routes ----------------------------------------------------------

app.get(['/', '/home'], (req, res) => {
  res.send(buildPage('Home', `
    <h1>Welcome to ShopNow</h1>
    <div class="card">
      <p style="color:#aaa">This is a demo web application instrumented with GTAE-ATRA real-time security monitoring.</p>
      <br>
      <span class="badge badge-green">🟢 Security Monitor: Active</span>
      <span class="badge badge-blue">🔵 GTAE Engine: Running</span>
      <span class="badge badge-purple">🟣 ATRA: Ready</span>
    </div>
    <div class="card">
      <h2 style="margin-bottom:1rem; color:#7c87ff">Featured Products</h2>
      <div class="product-grid">
        ${['Widget Pro', 'Gadget X', 'Device Ultra'].map(p => `
          <div class="product-card">
            <div style="font-size:2rem">📦</div>
            <div style="margin:0.5rem 0">${p}</div>
            <div class="price">$${(Math.random()*100+20).toFixed(2)}</div>
          </div>`).join('')}
      </div>
    </div>
  `));
});

app.get('/products', (req, res) => {
  const products = [
    { id: 1, name: 'Widget Pro 2.0',    price: 49.99, category: 'Electronics' },
    { id: 2, name: 'Gadget Ultra X',    price: 129.99, category: 'Electronics' },
    { id: 3, name: 'SmartDevice Plus',  price: 89.99, category: 'Smart Home' },
    { id: 4, name: 'SecureVault 3000',  price: 199.99, category: 'Security' },
    { id: 5, name: 'CloudHub Connect',  price: 74.99, category: 'Networking' },
  ];
  res.send(buildPage('Products', `
    <h1>Our Products</h1>
    <div class="product-grid">
      ${products.map(p => `
        <div class="product-card">
          <div style="font-size:2rem; margin-bottom:0.5rem">📦</div>
          <strong>${p.name}</strong><br>
          <small style="color:#888">${p.category}</small><br>
          <div class="price" style="margin-top:0.5rem">$${p.price}</div>
        </div>`).join('')}
    </div>
  `));
});

app.get('/about', (req, res) => {
  res.send(buildPage('About', `
    <h1>About Us</h1>
    <div class="card">
      <p>ShopNow is a demo e-commerce application built to demonstrate real-time web security monitoring
         using the GTAE-ATRA-NIDS system.</p>
      <br>
      <p style="color:#aaa">Every request is monitored by the GTAE Graph Transformer Autoencoder,
         which detects anomalous behaviour and triggers the Adaptive Threat Response Algorithm.</p>
    </div>
  `));
});

app.get('/contact', (req, res) => {
  res.send(buildPage('Contact', `
    <h1>Contact Us</h1>
    <div class="card">
      <form method="POST" action="/contact">
        <label>Name</label><input type="text" name="name" placeholder="Your name">
        <label>Email</label><input type="email" name="email" placeholder="your@email.com">
        <label>Message</label><input type="text" name="message" placeholder="Your message">
        <button type="submit">Send Message</button>
      </form>
    </div>
  `));
});

app.post('/contact', (req, res) => {
  res.send(buildPage('Contact', `
    <h1>Thank You!</h1>
    <div class="card"><p>Your message has been received.</p></div>
  `));
});

app.get('/login', (req, res) => {
  const msg = req.query.error ? '<div style="color:#f87171;margin-bottom:1rem">Invalid credentials.</div>' : '';
  res.send(buildPage('Login', `
    <h1>Login</h1>
    <div class="card">
      ${msg}
      <form method="POST" action="/login">
        <label>Username</label>
        <input type="text" name="username" id="username" placeholder="Enter username" autocomplete="username">
        <label>Password</label>
        <input type="password" name="password" id="password" placeholder="Enter password" autocomplete="current-password">
        <button type="submit">Sign In</button>
      </form>
      <p style="margin-top:1rem; color:#888; font-size:0.85rem">Demo users: alice / Password123!</p>
    </div>
  `));
});

app.post('/login', async (req, res) => {
  const { username, password } = req.body;
  const hash = USERS[username];

  if (!hash || !username || !password) {
    return res.status(401).send(buildPage('Login', `
      <h1>Login</h1>
      <div class="card">
        <div style="color:#f87171;margin-bottom:1rem">Invalid credentials.</div>
        <a href="/login" style="color:#7c87ff">Try again</a>
      </div>
    `));
  }

  const match = await bcrypt.compare(password, hash);
  if (!match) {
    return res.status(401).send(buildPage('Login', `
      <h1>Login</h1>
      <div class="card">
        <div style="color:#f87171;margin-bottom:1rem">Invalid credentials.</div>
        <a href="/login" style="color:#7c87ff">Try again</a>
      </div>
    `));
  }

  res.cookie('session', uuidv4(), { httpOnly: true, sameSite: 'strict' });
  res.redirect('/profile');
});

app.get('/profile', (req, res) => {
  const session = req.cookies?.session;
  if (!session) {
    return res.redirect('/login');
  }
  res.send(buildPage('Profile', `
    <h1>My Profile</h1>
    <div class="card">
      <p>Welcome back, User!</p>
      <br>
      <span class="badge badge-green">✓ Authenticated</span>
      <span class="badge badge-blue">Session: ${session.substring(0,8)}...</span>
    </div>
    <div class="card">
      <h3 style="color:#7c87ff; margin-bottom:1rem">Recent Orders</h3>
      <p style="color:#aaa">No orders yet.</p>
    </div>
  `));
});

app.get('/admin', (req, res) => {
  // Admin is restricted -- returns 403 for unauthenticated access
  // Repeated hits on this endpoint trigger ATRA's SensitiveEndpoint detection
  const session = req.cookies?.session;
  if (!session) {
    return res.status(403).send(buildPage('Forbidden', `
      <h1>403 Forbidden</h1>
      <div class="card">
        <p style="color:#f87171">Access to this resource is restricted.</p>
        <p style="color:#aaa; margin-top:0.5rem">This access attempt has been logged by the security monitoring system.</p>
      </div>
    `));
  }
  res.send(buildPage('Admin', `
    <h1>Admin Panel</h1>
    <div class="card"><p>Administrative functions.</p></div>
  `));
});

// ---- JSON APIs -------------------------------------------------------

app.get('/api/products', (req, res) => {
  res.json({
    products: [
      { id: 1, name: 'Widget Pro 2.0',   price: 49.99,  stock: 120 },
      { id: 2, name: 'Gadget Ultra X',   price: 129.99, stock: 45  },
      { id: 3, name: 'SmartDevice Plus', price: 89.99,  stock: 78  },
    ],
    total: 3,
  });
});

app.get('/api/categories', (req, res) => {
  res.json({
    categories: ['Electronics', 'Smart Home', 'Security', 'Networking'],
  });
});

app.get('/api/search', (req, res) => {
  const q = req.query.q || '';
  res.json({
    query: q,
    results: [
      { id: 1, name: 'Widget Pro 2.0', relevance: 0.92 },
    ],
    count: 1,
  });
});

// ---- Error handlers --------------------------------------------------

app.use((req, res) => {
  res.status(404).send(buildPage('404 Not Found', `
    <h1>404 Not Found</h1>
    <div class="card">
      <p style="color:#aaa">The page <code style="color:#7c87ff">${req.path}</code> was not found.</p>
      <p style="margin-top:1rem"><a href="/" style="color:#7c87ff">Go Home</a></p>
    </div>
  `));
});

app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send(buildPage('Error', `
    <h1>Server Error</h1>
    <div class="card"><p style="color:#f87171">An internal error occurred.</p></div>
  `));
});

// ---- Start -----------------------------------------------------------

app.listen(PORT, HOST, () => {
  console.log('='.repeat(70));
  console.log('  GTAE-ATRA DEMO WEB APPLICATION');
  console.log('='.repeat(70));
  console.log(`  URL         : http://localhost:${PORT}`);
  console.log(`  Monitor     : ${process.env.MONITOR_URL || 'http://127.0.0.1:8765/telemetry'}`);
  console.log('  Endpoints   : / /home /products /about /login /profile /admin');
  console.log('  API         : /api/products /api/categories /api/search');
  console.log('='.repeat(70));
});

module.exports = app;

`


---

### Module: dashboard/src/hooks/useSocket.js
**Description:** React Socket.IO Real-Time Stream Hook

`javascript
import { useState, useEffect, useRef, useCallback } from 'react';
import io from 'socket.io-client';

export const MONITOR_URL = process.env.REACT_APP_MONITOR_URL || 'http://127.0.0.1:8765';

export function useSecuritySocket() {
  const [stats, setStats]   = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef(null);

  useEffect(() => {
    const socket = io(MONITOR_URL, {
      transports: ['websocket', 'polling'],
      reconnectionAttempts: 10,
      reconnectionDelay:    2000,
    });
    socketRef.current = socket;

    socket.on('connect',    () => setConnected(true));
    socket.on('disconnect', () => setConnected(false));

    socket.on('stats_update', (data) => {
      setStats(data);
    });

    socket.on('security_alert', (event) => {
      setAlerts(prev => [event, ...prev].slice(0, 100));
    });

    return () => { socket.disconnect(); };
  }, []);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${MONITOR_URL}/api/security/stats`);
      if (res.ok) { const data = await res.json(); setStats(data); }
    } catch { /* server offline */ }
  }, []);

  const fetchLogs = useCallback(async () => {
    try {
      const res = await fetch(`${MONITOR_URL}/api/logs`);
      if (res.ok) {
        const data = await res.json();
        setAlerts(data.events || []);
      }
    } catch { /* server offline */ }
  }, []);

  // Poll every 10s as fallback when socket isn't delivering
  useEffect(() => {
    fetchStats();
    fetchLogs();
    const timer = setInterval(() => { fetchStats(); }, 10000);
    return () => clearInterval(timer);
  }, [fetchStats, fetchLogs]);

  return { stats, alerts, connected };
}

`


---

### Module: dashboard/src/App.js
**Description:** Interactive Security Operations Center (SOC) React Dashboard

`javascript
import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, AreaChart, Area,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { useSecuritySocket, MONITOR_URL } from './hooks/useSocket';
import './App.css';

// ============================================================
// HELPERS
// ============================================================

function riskBadge(level) {
  const cls = { Low: 'allow', Medium: 'monitor', High: 'alert', Critical: 'critical' }[level] || 'monitor';
  return <span className={`badge badge-${cls}`}>{level || 'Low'}</span>;
}

function actionBadge(action) {
  const cls = {
    ALLOW: 'allow', MONITOR: 'monitor', ALERT: 'alert', BLOCK: 'block',
  }[action] || 'monitor';
  return <span className={`badge badge-${cls}`}>{action || 'ALLOW'}</span>;
}

function formatTime(ts) {
  if (!ts) return '-';
  try {
    return new Date(ts).toLocaleTimeString();
  } catch { return ts; }
}

function AnomalyScore({ score }) {
  const pct = Math.min(score, 100);
  const color = pct < 30 ? '#34d399' : pct < 60 ? '#fbbf24' : pct < 85 ? '#fb923c' : '#f87171';
  return (
    <div className="score-bar">
      <div className="score-label mono">{(score || 0).toFixed(1)}</div>
      <div className="score-track">
        <div className="score-fill" style={{ width: `${Math.min(pct, 100)}%`, background: color }} />
      </div>
    </div>
  );
}

// ============================================================
// STAT CARD
// ============================================================

function StatCard({ label, value, icon, accent, sub }) {
  return (
    <div className="stat-card animate-fade-in-up" style={{ '--accent': accent }}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-body">
        <div className="stat-value" style={{ color: accent }}>{value ?? '—'}</div>
        <div className="stat-label">{label}</div>
        {sub && <div className="stat-sub">{sub}</div>}
      </div>
    </div>
  );
}

// ============================================================
// ALERT ROW
// ============================================================

function AlertRow({ event, index }) {
  return (
    <div
      className={`alert-row animate-slide-in ${event.is_anomaly ? 'alert-row--anomaly' : ''}`}
      style={{ animationDelay: `${index * 0.03}s` }}
    >
      <div className="alert-time mono">{formatTime(event.timestamp)}</div>
      <div className="alert-ip mono">{event.source_ip}</div>
      <div className="alert-site mono" style={{ color: '#38bdf8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {event.site_id || 'default'}
      </div>
      <div className="alert-endpoint mono">{event.endpoint}</div>
      <div className="alert-score"><AnomalyScore score={event.anomaly_score} /></div>
      <div className="alert-risk">{riskBadge(event.risk_level)}</div>
      <div className="alert-action">{actionBadge(event.action)}</div>
    </div>
  );
}

// ============================================================
// MAIN APP
// ============================================================

export default function App() {
  const { stats, alerts, connected } = useSecuritySocket();
  const [tab, setTab] = useState('overview');
  const [anomalyHistory, setAnomalyHistory] = useState([]);
  const [reqHistory, setReqHistory] = useState([]);
  const [copied, setCopied] = useState(false);

  // Read initial site from URL: ?site=ai-research-paper-explainer
  const [selectedSite, setSelectedSite] = useState(() => {
    try {
      const p = new URLSearchParams(window.location.search).get('site');
      return p ? p.trim() : 'all';
    } catch {
      return 'all';
    }
  });

  // Calculate available sites dynamically
  const availableSites = React.useMemo(() => {
    const set = new Set(stats?.monitored_sites || []);
    alerts.forEach(a => { if (a.site_id) set.add(a.site_id); });
    if (set.size === 0) {
      set.add('ai-research-paper-explainer');
      set.add('fitfuel-store');
    }
    return Array.from(set).filter(Boolean);
  }, [stats, alerts]);

  // Handle site selection change & update URL
  const handleSiteChange = (site) => {
    setSelectedSite(site);
    try {
      const url = new URL(window.location);
      if (site === 'all') {
        url.searchParams.delete('site');
      } else {
        url.searchParams.set('site', site);
      }
      window.history.replaceState(null, '', url.toString());
    } catch (e) {
      console.warn('Could not update history state:', e);
    }
  };

  // Copy shareable link
  const copySiteLink = () => {
    try {
      const url = new URL(window.location.href);
      if (selectedSite !== 'all') {
        url.searchParams.set('site', selectedSite);
      }
      navigator.clipboard.writeText(url.toString());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // fallback
    }
  };

  // Filter events for the selected site
  const filteredAlerts = React.useMemo(() => {
    if (selectedSite === 'all') return alerts;
    return alerts.filter(e => (e.site_id || 'default') === selectedSite);
  }, [alerts, selectedSite]);

  // Filter blocked IPs for the selected site
  const filteredBlockedIPs = React.useMemo(() => {
    if (selectedSite === 'all') return stats?.blocked_ips || [];
    const ips = new Set();
    filteredAlerts.forEach(e => {
      if ((e.action === 'BLOCK' || e.action === 'SIMULATED_BLOCK') && e.source_ip) {
        ips.add(e.source_ip);
      }
    });
    return Array.from(ips);
  }, [filteredAlerts, selectedSite, stats]);

  // Unblock handlers
  const handleUnblock = async (ip) => {
    try {
      await fetch(`${MONITOR_URL}/api/blocklist/unblock?ip=${encodeURIComponent(ip)}`, { method: 'POST' });
    } catch (err) {
      console.error('Failed to unblock IP:', err);
    }
  };

  const handleClearAllBlocks = async () => {
    try {
      await fetch(`${MONITOR_URL}/api/blocklist/clear`, { method: 'POST' });
    } catch (err) {
      console.error('Failed to clear blocklist:', err);
    }
  };

  // Filtered metrics
  const total = selectedSite === 'all' ? (stats?.total_requests || 0) : filteredAlerts.length;
  const alerts_ = selectedSite === 'all'
    ? (stats?.total_alerts || 0)
    : filteredAlerts.filter(e => e.is_anomaly).length;
  const blocked = selectedSite === 'all'
    ? (stats?.total_blocked || 0)
    : filteredBlockedIPs.length;
  const normal = selectedSite === 'all'
    ? (stats?.total_normal || 0)
    : filteredAlerts.filter(e => !e.is_anomaly).length;
  const rps = stats?.requests_per_second || 0;
  
  const siteAnomalyPoints = selectedSite === 'all'
    ? (stats?.anomaly_score_history || [])
    : (stats?.anomaly_score_history || []).filter(pt => !pt.site_id || pt.site_id === selectedSite);
  const latestScore = siteAnomalyPoints.slice(-1)[0]?.score || (filteredAlerts[0]?.anomaly_score || 0);

  // Build chart data
  useEffect(() => {
    if (!stats) return;
    const ts = new Date().toLocaleTimeString();

    setAnomalyHistory(prev => {
      const point = { time: ts, score: latestScore || 0, level: latestScore > 70 ? 'Critical' : latestScore > 40 ? 'High' : 'Low' };
      return [...prev, point].slice(-60);
    });

    setReqHistory(prev => {
      const point = {
        time: ts,
        total: total || 0,
        alerts: alerts_ || 0,
        blocked: blocked || 0,
        rps: rps || 0,
      };
      return [...prev, point].slice(-60);
    });
  }, [stats, total, alerts_, blocked, rps, latestScore]);

  return (
    <div className="app">
      {/* ---- NAVBAR ---- */}
      <nav className="navbar">
        <div className="navbar-brand">
          <span className="brand-icon">🛡</span>
          <span className="brand-name">GTAE-ATRA</span>
          <span className="brand-sub">Web Security Monitor</span>
        </div>

        {/* Multi-Tenant Site Selector */}
        <div className="site-selector-wrapper">
          <span className="site-selector-label">🌐 Site:</span>
          <select
            className="site-selector-select"
            value={selectedSite}
            onChange={(e) => handleSiteChange(e.target.value)}
          >
            <option value="all">All Protected Sites ({availableSites.length})</option>
            {availableSites.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          {selectedSite !== 'all' && (
            <button className="site-share-btn" onClick={copySiteLink} title="Copy shareable link for this site">
              {copied ? '✓ Copied!' : '🔗 Share Link'}
            </button>
          )}
        </div>

        <div className="navbar-status">
          <span className={`status-dot ${connected ? 'active' : 'offline'}`} />
          <span className="status-text">{connected ? 'Live' : 'Offline'}</span>
          <span className="status-time">{new Date().toLocaleTimeString()}</span>
        </div>
      </nav>

      {/* ---- TABS ---- */}
      <div className="tab-bar">
        {['overview', 'alerts', 'logs', 'blocked'].map(t => (
          <button
            key={t}
            className={`tab-btn ${tab === t ? 'tab-btn--active' : ''}`}
            onClick={() => setTab(t)}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      <div className="main-content">
        {/* Site Filter Notification Banner */}
        {selectedSite !== 'all' && (
          <div className="site-filter-banner">
            <div>
              Viewing private security logs for website: <b>{selectedSite}</b>
              <span style={{ color: '#94a3b8', marginLeft: '0.5rem', fontSize: '0.8rem' }}>
                (Only telemetry and threats targeting this site are shown)
              </span>
            </div>
            <button className="site-filter-reset-btn" onClick={() => handleSiteChange('all')}>
              Show All Sites
            </button>
          </div>
        )}

        {/* ================ OVERVIEW TAB ================ */}
        {tab === 'overview' && (
          <>
            {/* Stat Cards */}
            <div className="stats-grid">
              <StatCard label="Total Requests"    value={total.toLocaleString()} icon="📊" accent="#60a5fa" />
              <StatCard label="Req / Second"       value={rps.toFixed(1)}         icon="⚡" accent="#a78bfa" />
              <StatCard label={selectedSite === 'all' ? "Protected Sites" : "Monitored Target"}
                        value={selectedSite === 'all' ? (stats?.site_count || availableSites.length) : selectedSite}
                        icon="🌐" accent="#38bdf8"
                        sub={selectedSite === 'all' ? availableSites.slice(0, 2).join(', ') : 'Dedicated Mode'} />
              <StatCard label="Security Alerts"   value={alerts_.toLocaleString()} icon="🚨" accent="#fb923c" />
              <StatCard label="Blocked IPs"       value={blocked.toLocaleString()} icon="🚫" accent="#f87171" />
              <StatCard label="Normal Requests"   value={normal.toLocaleString()}  icon="✅" accent="#34d399" />
              <StatCard label="Current Score"     value={latestScore.toFixed(1)}   icon="🎯" accent="#fbbf24"
                        sub={latestScore < 30 ? 'Normal' : latestScore < 60 ? 'Suspicious' : 'High Risk'} />
            </div>

            {/* Charts row */}
            <div className="charts-grid">
              {/* Anomaly score chart */}
              <div className="card chart-card">
                <h3 className="chart-title">Anomaly Score Over Time</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={anomalyHistory}>
                    <defs>
                      <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="#a78bfa" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#a78bfa" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2a45" />
                    <XAxis dataKey="time" tick={{ fill:'#7878a0', fontSize:10 }} />
                    <YAxis tick={{ fill:'#7878a0', fontSize:10 }} />
                    <Tooltip contentStyle={{ background:'#161628', border:'1px solid #2a2a45', color:'#e8e8f0' }} />
                    <Area type="monotone" dataKey="score" stroke="#a78bfa" fill="url(#scoreGrad)" strokeWidth={2} dot={false} name="Score" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Request rate chart */}
              <div className="card chart-card">
                <h3 className="chart-title">Request Rate (req/s)</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={reqHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2a45" />
                    <XAxis dataKey="time" tick={{ fill:'#7878a0', fontSize:10 }} />
                    <YAxis tick={{ fill:'#7878a0', fontSize:10 }} />
                    <Tooltip contentStyle={{ background:'#161628', border:'1px solid #2a2a45', color:'#e8e8f0' }} />
                    <Line type="monotone" dataKey="rps"  stroke="#60a5fa" strokeWidth={2} dot={false} name="Req/s" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Alert distribution */}
            <div className="card" style={{ marginTop:'1.5rem' }}>
              <h3 className="chart-title" style={{ marginBottom:'1rem' }}>Alert Distribution</h3>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={[
                  { name:'Normal',   count: normal },
                  { name:'Monitor',  count: Math.max(0, alerts_ - blocked) },
                  { name:'Blocked',  count: blocked },
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2a45" />
                  <XAxis dataKey="name" tick={{ fill:'#7878a0', fontSize:11 }} />
                  <YAxis tick={{ fill:'#7878a0', fontSize:11 }} />
                  <Tooltip contentStyle={{ background:'#161628', border:'1px solid #2a2a45', color:'#e8e8f0' }} />
                  <Bar dataKey="count" fill="#60a5fa" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Recent events (last 10) */}
            <div className="card" style={{ marginTop:'1.5rem' }}>
              <h3 className="chart-title" style={{ marginBottom:'1rem' }}>
                Recent Security Events {selectedSite !== 'all' && `(${selectedSite})`}
              </h3>
              <div className="alert-table">
                <div className="alert-header">
                  <div>Time</div><div>Source IP</div><div>Site</div><div>Endpoint</div>
                  <div>Score</div><div>Risk</div><div>Action</div>
                </div>
                {filteredAlerts.slice(0, 10).map((e, i) => <AlertRow key={i} event={e} index={i} />)}
                {filteredAlerts.length === 0 && (
                  <div style={{ textAlign:'center', padding:'2rem', color:'#7878a0' }}>
                    No events recorded for this website yet. Monitoring active...
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* ================ ALERTS TAB ================ */}
        {tab === 'alerts' && (
          <div className="card">
            <h3 className="chart-title" style={{ marginBottom:'1rem' }}>
              Security Alerts {selectedSite !== 'all' && `(${selectedSite})`}
              <span className="badge badge-high" style={{ marginLeft:'0.75rem' }}>
                {filteredAlerts.filter(e=>e.is_anomaly).length} alerts
              </span>
            </h3>
            {filteredAlerts.filter(e => e.is_anomaly).map((e, i) => (
              <div key={i} className="alert-detail animate-slide-in" style={{ animationDelay: `${i*0.03}s` }}>
                <div className="alert-detail-header">
                  <div>
                    <span className="mono" style={{ color:'#60a5fa' }}>{e.source_ip}</span>
                    <span style={{ margin:'0 0.5rem', color:'#7878a0' }}>→</span>
                    <span className="mono" style={{ color:'#a78bfa' }}>{e.endpoint}</span>
                    <span className="badge badge-allow" style={{ marginLeft:'0.75rem' }}>{e.site_id || 'default'}</span>
                  </div>
                  <div style={{ display:'flex', gap:'0.5rem', alignItems:'center' }}>
                    {riskBadge(e.risk_level)}
                    {actionBadge(e.action)}
                    <span className="mono" style={{ color:'#7878a0', fontSize:'0.8rem' }}>{formatTime(e.timestamp)}</span>
                  </div>
                </div>
                <div className="alert-detail-body">
                  <div className="alert-reason">
                    <span style={{ color:'#7878a0' }}>Reason: </span>
                    <span>{e.reason}</span>
                  </div>
                  <div className="alert-metrics">
                    <span>Score: <b style={{ color:'#fbbf24' }}>{(e.anomaly_score||0).toFixed(1)}</b></span>
                    <span>Risk: <b style={{ color:'#fb923c' }}>{(e.risk_score||0).toFixed(1)}/100</b></span>
                    <span>Type: <b style={{ color:'#a78bfa' }}>{e.attack_type || 'Unknown'}</b></span>
                    <span>Reqs: <b style={{ color:'#60a5fa' }}>{e.request_count}</b></span>
                    <span>ML ms: <b style={{ color:'#34d399' }}>{e.timing?.total_ms?.toFixed(1) || '-'}</b></span>
                  </div>
                </div>
              </div>
            ))}
            {filteredAlerts.filter(e=>e.is_anomaly).length === 0 && (
              <div style={{ textAlign:'center', padding:'3rem', color:'#7878a0' }}>
                ✅ No security alerts for {selectedSite === 'all' ? 'any site' : selectedSite}
              </div>
            )}
          </div>
        )}

        {/* ================ LOGS TAB ================ */}
        {tab === 'logs' && (
          <div className="card">
            <h3 className="chart-title" style={{ marginBottom:'1rem' }}>
              All Security Events {selectedSite !== 'all' && `(${selectedSite})`}
            </h3>
            <div className="alert-table">
              <div className="alert-header">
                <div>Time</div><div>Source IP</div><div>Site</div><div>Endpoint</div>
                <div>Score</div><div>Risk</div><div>Action</div>
              </div>
              {filteredAlerts.map((e, i) => <AlertRow key={i} event={e} index={i} />)}
              {filteredAlerts.length === 0 && (
                <div style={{ textAlign:'center', padding:'2rem', color:'#7878a0' }}>
                  No events logged for {selectedSite === 'all' ? 'any site' : selectedSite} yet.
                </div>
              )}
            </div>
          </div>
        )}

        {/* ================ BLOCKED TAB ================ */}
        {tab === 'blocked' && (
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
              <h3 className="chart-title" style={{ margin: 0 }}>
                Blocked IPs {selectedSite !== 'all' && `(${selectedSite})`}
                <span className="badge badge-block" style={{ marginLeft:'0.75rem' }}>
                  {filteredBlockedIPs.length}
                </span>
              </h3>
              {filteredBlockedIPs.length > 0 && (
                <button
                  onClick={handleClearAllBlocks}
                  style={{
                    background: 'rgba(239, 68, 68, 0.15)',
                    color: '#f87171',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    padding: '0.45rem 0.9rem',
                    borderRadius: '6px',
                    fontSize: '0.85rem',
                    cursor: 'pointer',
                    fontWeight: 600,
                    transition: 'all 0.2s ease'
                  }}
                  title="Unblock all blocked IPs across the system"
                >
                  🔓 Unblock All IPs
                </button>
              )}
            </div>

            {filteredBlockedIPs.map((ip, i) => (
              <div key={i} className="blocked-row animate-slide-in" style={{ animationDelay: `${i*0.05}s`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="mono" style={{ color:'#f87171' }}>🚫 {ip}</span>
                <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                  <span className="badge badge-block">BLOCKED</span>
                  <button
                    onClick={() => handleUnblock(ip)}
                    style={{
                      background: 'rgba(52, 211, 153, 0.15)',
                      color: '#34d399',
                      border: '1px solid rgba(52, 211, 153, 0.3)',
                      padding: '0.25rem 0.65rem',
                      borderRadius: '4px',
                      fontSize: '0.75rem',
                      cursor: 'pointer',
                      fontWeight: 600
                    }}
                    title={`Unblock IP ${ip}`}
                  >
                    🔓 Unblock
                  </button>
                </div>
              </div>
            ))}
            {filteredBlockedIPs.length === 0 && (
              <div style={{ textAlign:'center', padding:'3rem', color:'#7878a0' }}>
                ✅ No IPs currently blocked on {selectedSite === 'all' ? 'any site' : selectedSite}
              </div>
            )}
            <p style={{ marginTop:'1.5rem', color:'#7878a0', fontSize:'0.85rem' }}>
              🛡️ Active IP blocking is enabled. Blocked IPs receive HTTP 403 Forbidden
              and cannot access any routes on monitored websites.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

`


---

### Module: tests/traffic/generate_attack_traffic.py
**Description:** Attack Simulation & Stress Testing Script

`python
"""
tests/traffic/generate_attack_traffic.py
Controlled suspicious traffic simulation for GTAE-ATRA-NIDS testing.

IMPORTANT:
  - All traffic is directed to the LOCAL demo app (http://127.0.0.1:3000)
  - No real attack tools, malware, or destructive payloads are used
  - This is purely for evaluating the detection system locally
  - The X-Forwarded-For header spoofs IPs so the ML system sees distinct sources

Usage:
  python tests/traffic/generate_attack_traffic.py --scenario brute_force
  python tests/traffic/generate_attack_traffic.py --scenario all
  python tests/traffic/generate_attack_traffic.py --list
"""

import argparse
import random
import time
import sys
import requests
from datetime import datetime

BASE_URL = "http://127.0.0.1:3000"

ATTACK_IP_BASE = "172.16.1."  # simulated attacker subnet


def _req(session, method, path, data=None, params=None, headers_extra=None, attacker_ip="172.16.1.100"):
    url = f"{BASE_URL}{path}"
    headers = {
        "User-Agent":       random.choice([
            "python-requests/2.28.0", "curl/7.88.1", "Go-http-client/1.1",
        ]),
        "X-Forwarded-For":  attacker_ip,
        "Accept":           "*/*",
        **(headers_extra or {}),
    }
    try:
        if method == "GET":
            r = session.get(url, params=params, headers=headers, timeout=4, allow_redirects=False)
        else:
            r = session.post(url, data=data, json=data if isinstance(data, dict) else None,
                             headers=headers, timeout=4, allow_redirects=False)
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"  [{ts}] {method:4} {path:<40} -> {r.status_code}")
        return r.status_code
    except Exception as e:
        print(f"  [ERR] {path}: {e}")
        return None


# ---- Scenario definitions -----------------------------------------------

def scenario_brute_force(session):
    """Simulate repeated failed login attempts from one IP."""
    print("\n[SCENARIO] Brute Force Login Attack")
    ip = f"{ATTACK_IP_BASE}{random.randint(100,150)}"
    for i in range(40):
        _req(session, "POST", "/login",
             data={"username": "admin", "password": f"wrong{i}"},
             attacker_ip=ip)
        time.sleep(0.3)
    print("  Done: 40 failed login attempts")


def scenario_rate_abuse(session):
    """Simulate high-frequency requests from one IP."""
    print("\n[SCENARIO] Rate Limit Abuse / DoS Simulation")
    ip = f"{ATTACK_IP_BASE}{random.randint(50,80)}"
    for i in range(120):
        _req(session, "GET", "/api/products", attacker_ip=ip)
        time.sleep(0.1)
    print("  Done: 120 rapid requests")


def scenario_path_traversal(session):
    """Simulate path traversal probing."""
    print("\n[SCENARIO] Path Traversal Probing")
    ip = f"{ATTACK_IP_BASE}{random.randint(200,250)}"
    paths = [
        "/../../../etc/passwd", "/.env",
        "/config/database.yml", "/.git/config",
        "/%2e%2e%2f%2e%2e%2fetc/passwd", "/../../windows/win.ini",
        "/admin/../../../etc/shadow",
    ]
    for path in paths * 3:
        _req(session, "GET", path, attacker_ip=ip)
        time.sleep(0.4)
    print("  Done: Path traversal probes")


def scenario_sqli(session):
    """Simulate SQL injection-like query patterns (against local demo only)."""
    print("\n[SCENARIO] SQL Injection Probe Simulation")
    ip = f"{ATTACK_IP_BASE}{random.randint(10,30)}"
    sqli_queries = [
        "id=1' OR '1'='1",
        "id=1 UNION SELECT * FROM users",
        "search='; DROP TABLE users;--",
        "q=admin'--",
        "id=1 AND SLEEP(5)--",
    ]
    for q in sqli_queries * 4:
        _req(session, "GET", "/api/search", params={"q": q}, attacker_ip=ip)
        time.sleep(0.5)
    print("  Done: SQLi probe simulation")


def scenario_xss(session):
    """Simulate XSS-like URL patterns (against local demo only)."""
    print("\n[SCENARIO] XSS Probe Simulation")
    ip = f"{ATTACK_IP_BASE}{random.randint(30,50)}"
    xss_payloads = [
        "<script>alert('xss')</script>",
        "javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
    ]
    for payload in xss_payloads * 4:
        _req(session, "GET", "/api/search", params={"q": payload}, attacker_ip=ip)
        time.sleep(0.4)
    print("  Done: XSS probe simulation")


def scenario_sensitive_scan(session):
    """Simulate systematic scanning of sensitive endpoints."""
    print("\n[SCENARIO] Sensitive Endpoint Scan")
    ip = f"{ATTACK_IP_BASE}{random.randint(150,200)}"
    targets = [
        "/admin", "/admin/", "/wp-admin", "/phpmyadmin",
        "/.env", "/config", "/api/admin", "/api/users",
        "/api/keys", "/debug", "/actuator", "/metrics",
    ]
    for target in targets * 3:
        _req(session, "GET", target, attacker_ip=ip)
        time.sleep(0.3)
    print("  Done: Sensitive endpoint scan")


def scenario_bot_scan(session):
    """Simulate bot scanning of many endpoints."""
    print("\n[SCENARIO] Bot Scan")
    ip = f"{ATTACK_IP_BASE}{random.randint(80,100)}"
    scan_paths = [
        "/robots.txt", "/sitemap.xml", "/backup.sql", "/dump.sql",
        "/data.zip", "/secret", "/private", "/test", "/wp-content",
        "/uploads", "/assets", "/.htaccess", "/web.config",
    ]
    for path in scan_paths * 2:
        _req(session, "GET", path, attacker_ip=ip)
        time.sleep(0.25)
    print("  Done: Bot scan")


def scenario_abnormal_methods(session):
    """Simulate unusual HTTP methods."""
    print("\n[SCENARIO] Abnormal HTTP Methods")
    ip = f"{ATTACK_IP_BASE}{random.randint(250,254)}"
    methods_paths = [
        ("TRACE",    "/"),
        ("OPTIONS",  "/api/products"),
        ("PROPFIND", "/admin"),
        ("MOVE",     "/products"),
    ]
    for _ in range(5):
        for method, path in methods_paths:
            _req(session, method, path, attacker_ip=ip)
            time.sleep(0.5)
    print("  Done: Abnormal HTTP methods")


def scenario_combined(session):
    """Simulate a combined multi-vector attack pattern."""
    print("\n[SCENARIO] Combined Multi-Vector Attack")
    ip = f"{ATTACK_IP_BASE}{random.randint(1,10)}"
    print("  Phase 1: Reconnaissance (scanning)")
    for path in ["/admin", "/.env", "/api/admin", "/api/users"]:
        _req(session, "GET", path, attacker_ip=ip)
        time.sleep(0.3)
    print("  Phase 2: Brute force")
    for i in range(15):
        _req(session, "POST", "/login",
             data={"username": "admin", "password": f"pass{i}"},
             attacker_ip=ip)
        time.sleep(0.2)
    print("  Phase 3: SQLi probe")
    for q in ["id=1' OR '1'='1", "q=admin'--"]:
        _req(session, "GET", "/api/search", params={"q": q}, attacker_ip=ip)
        time.sleep(0.4)
    print("  Done: Combined attack simulation")


SCENARIOS = {
    "brute_force":     scenario_brute_force,
    "rate_abuse":      scenario_rate_abuse,
    "path_traversal":  scenario_path_traversal,
    "sqli":            scenario_sqli,
    "xss":             scenario_xss,
    "sensitive_scan":  scenario_sensitive_scan,
    "bot_scan":        scenario_bot_scan,
    "abnormal_methods":scenario_abnormal_methods,
    "combined":        scenario_combined,
}


def main():
    parser = argparse.ArgumentParser(description="Generate suspicious test traffic (local only)")
    parser.add_argument("--scenario", default="brute_force",
                        choices=list(SCENARIOS.keys()) + ["all"],
                        help="Attack scenario to simulate")
    parser.add_argument("--target", default="http://127.0.0.1:3000",
                        help="Target base URL (default: http://127.0.0.1:3000)")
    parser.add_argument("--list", action="store_true", help="List all scenarios")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.target.rstrip('/')

    if args.list:
        print("Available scenarios:")
        for name in SCENARIOS:
            print(f"  {name}")
        return

    print("=" * 70)
    print("  SUSPICIOUS TRAFFIC SIMULATOR (LOCAL TESTING ONLY)")
    print("=" * 70)
    print(f"  Target   : {BASE_URL}")
    print(f"  Scenario : {args.scenario}")
    print(f"  WARNING  : All traffic directed to LOCAL demo app only!")
    print("=" * 70)

    session = requests.Session()

    if args.scenario == "all":
        for name, fn in SCENARIOS.items():
            fn(session)
            time.sleep(2)
    else:
        SCENARIOS[args.scenario](session)

    print("\n[OK] Test scenario complete.")


if __name__ == "__main__":
    main()

`


---

### Module: app.py
**Description:** Integrated Real-Time Terminal & Web Dashboard Runner

`python
"""
app.py -- Dual-Output Web Dashboard & Terminal Server for GTAE-IDS + ATRA.

Executes the pipeline and broadcasts all logs simultaneously to:
  1. Terminal (Standard Output in real-time)
  2. Web Dashboard (Server-Sent Events streaming in real-time)

Access:
  http://127.0.0.1:5000
"""

import sys
import os
import subprocess
import threading
import queue
import time
import json
import csv
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config import DATA_CONFIG, ATRA_CONFIG, EVALUATION_CONFIG, LOGS_DIR, RESULTS_DIR

app = Flask(__name__, template_folder="templates", static_folder="static")

# Shared log streaming queue and ring buffer
log_subscribers = []
log_history = []
MAX_HISTORY = 500
is_pipeline_running = False
pipeline_lock = threading.Lock()


def emit_log(line: str):
    """Print to terminal AND broadcast to all connected web clients."""
    # 1. Output to terminal
    sys.stdout.write(line)
    sys.stdout.flush()

    # 2. Store in memory buffer
    timestamped_line = line
    if len(log_history) > MAX_HISTORY:
        log_history.pop(0)
    log_history.append(timestamped_line)

    # 3. Broadcast to Web SSE subscribers
    dead_subs = []
    for sub in log_subscribers:
        try:
            sub.put(line)
        except Exception:
            dead_subs.append(sub)
    for d in dead_subs:
        if d in log_subscribers:
            log_subscribers.remove(d)


def run_process_async(cmd: list, cwd: str = None, name: str = "Task"):
    """Run a subprocess and stream output simultaneously to terminal and web."""
    global is_pipeline_running
    with pipeline_lock:
        is_pipeline_running = True

    def worker():
        global is_pipeline_running
        env = os.environ.copy()
        src_dir = str(PROJECT_ROOT / "src")
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        emit_log(f"\n{'=' * 70}\n")
        emit_log(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: {name}\n")
        emit_log(f"Command: {' '.join(cmd)}\n")
        emit_log(f"{'=' * 70}\n\n")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=cwd or str(PROJECT_ROOT),
                env=env,
            )

            for line in proc.stdout:
                emit_log(line)

            proc.wait()
            status = "COMPLETED SUCCESSFULLY" if proc.returncode == 0 else f"FAILED (Code {proc.returncode})"
            emit_log(f"\n[{datetime.now().strftime('%H:%M:%S')}] {name} {status}.\n")
            emit_log(f"{'=' * 70}\n\n")

        except Exception as e:
            emit_log(f"\n[ERROR] Failed to run {name}: {str(e)}\n")
        finally:
            with pipeline_lock:
                is_pipeline_running = False

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


@app.route("/")
def index():
    """Main Cyber Security SOC Dashboard."""
    return render_template("index.html")


@app.route("/api/stream-logs")
def stream_logs():
    """Server-Sent Events endpoint streaming live terminal output to the web UI."""
    def event_stream():
        q = queue.Queue()
        # First send existing history to newly opened browser tab
        for line in log_history[-100:]:
            yield f"data: {json.dumps({'line': line})}\n\n"

        log_subscribers.append(q)
        try:
            while True:
                try:
                    line = q.get(timeout=20)
                    yield f"data: {json.dumps({'line': line})}\n\n"
                except queue.Empty:
                    # Keep-alive heartbeat comment
                    yield ": ping\n\n"
        except GeneratorExit:
            if q in log_subscribers:
                log_subscribers.remove(q)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/stats")
def get_stats():
    """Return latest metrics, risk distribution, and action counts."""
    # Read evaluation results if available
    eval_path = Path(EVALUATION_CONFIG["results_path"])
    eval_data = {}
    if eval_path.exists():
        try:
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
        except Exception:
            pass

    # Read attack log CSV to compute fresh counts
    log_path = Path(ATRA_CONFIG["attack_log_path"])
    total_flagged = 0
    risk_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    action_counts = {"Generate Log": 0, "Alert Admin": 0, "Blacklist IP": 0, "Block IP + Alert + Log": 0}
    attack_type_counts = {}

    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_flagged += 1
                    rl = row.get("Risk Level", "Unknown")
                    if rl in risk_counts:
                        risk_counts[rl] += 1

                    act = row.get("Action Taken", "Unknown")
                    action_counts[act] = action_counts.get(act, 0) + 1

                    atype = row.get("Attack Type", "Unknown")
                    attack_type_counts[atype] = attack_type_counts.get(atype, 0) + 1
        except Exception:
            pass

    # Read blacklist count
    bl_path = Path(ATRA_CONFIG["blacklist_path"])
    bl_count = 0
    if bl_path.exists():
        try:
            with open(bl_path, "r", encoding="utf-8") as f:
                bl_count = sum(1 for line in f if line.strip())
        except Exception:
            pass

    metrics = eval_data.get("metrics", {})

    return jsonify({
        "status": "Running" if is_pipeline_running else "Idle",
        "total_flows": 500,
        "total_flagged": total_flagged,
        "total_benign": 500 - total_flagged if total_flagged <= 500 else 343,
        "blacklisted_ips_count": bl_count,
        "tpr": metrics.get("tpr", 1.0) * 100,
        "fpr": metrics.get("fpr", 0.14) * 100,
        "f1": metrics.get("f1", 0.78),
        "precision": metrics.get("precision", 0.64) * 100,
        "risk_counts": risk_counts,
        "action_counts": action_counts,
        "attack_type_counts": attack_type_counts,
        "is_pipeline_running": is_pipeline_running,
    })


@app.route("/api/logs")
def get_attack_logs():
    """Return parsed attack log entries for the data table."""
    log_path = Path(ATRA_CONFIG["attack_log_path"])
    records = []
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
        except Exception as e:
            return jsonify({"error": str(e), "records": []})

    # Return newest first
    records.reverse()
    return jsonify({"records": records, "count": len(records)})


@app.route("/api/blacklist")
def get_blacklist():
    """Return active blacklisted IPs."""
    bl_path = Path(ATRA_CONFIG["blacklist_path"])
    entries = []
    if bl_path.exists():
        try:
            with open(bl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(",")
                        ip = parts[0]
                        timestamp = parts[1] if len(parts) > 1 else "Active"
                        entries.append({"ip": ip, "timestamp": timestamp})
        except Exception as e:
            return jsonify({"error": str(e), "blacklist": []})

    entries.reverse()
    return jsonify({"blacklist": entries, "count": len(entries)})


@app.route("/api/run-pipeline", methods=["POST"])
def trigger_pipeline():
    """Trigger the IDS pipeline execution (streams to both terminal & web)."""
    if is_pipeline_running:
        return jsonify({"success": False, "message": "A process is already running in background."}), 400

    data = request.get_json() or {}
    skip_data = data.get("skip_data", False)
    use_synthetic = data.get("synthetic", True)

    cmd = [sys.executable, "-u", "main.py"]
    if skip_data:
        cmd.append("--skip-data")
    if use_synthetic:
        cmd.append("--synthetic")

    run_process_async(cmd, name="Full IDS Pipeline")
    return jsonify({"success": True, "message": "Pipeline started. Logs are streaming to terminal and web."})


@app.route("/api/run-evaluation", methods=["POST"])
def trigger_evaluation():
    """Trigger evaluate.py to refresh metrics and charts."""
    if is_pipeline_running:
        return jsonify({"success": False, "message": "A process is already running in background."}), 400

    cmd = [sys.executable, "-u", "evaluate.py"]
    run_process_async(cmd, name="Evaluation & Reporting")
    return jsonify({"success": True, "message": "Evaluation started."})


@app.route("/api/clear-logs", methods=["POST"])
def clear_logs():
    """Clear memory log buffer."""
    global log_history
    log_history = []
    return jsonify({"success": True})


@app.route("/results/<path:filename>")
def serve_result_image(filename):
    """Serve generated plots directly into the browser."""
    return send_from_directory(str(PROJECT_ROOT / "results"), filename)


def start_server(host="127.0.0.1", port=5000):
    """Start the dual-output terminal + web server."""
    print("=" * 72)
    print("  GTAE-IDS + ATRA SECURITY OPERATIONS CENTER (SOC) WEB DASHBOARD")
    print("=" * 72)
    print(f"  [>] Terminal Engine : ACTIVE (All logs print here in real-time)")
    print(f"  [>] Web Dashboard   : http://{host}:{port}")
    print(f"  [>] Status          : READY")
    print("=" * 72)
    print("Press Ctrl+C in terminal to stop the web server.\n")

    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    start_server()


`
