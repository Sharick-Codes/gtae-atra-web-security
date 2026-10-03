# GTAE-ATRA-NIDS: Web Security Monitoring Edition

**Graph Transformer Autoencoder + Adaptive Threat Response Algorithm**  
Real-Time Web Application Security Monitoring System

---

## What This System Does

This system monitors HTTP traffic to a web application and detects anomalous behaviour using a Graph Transformer Autoencoder (GTAE) + ensemble anomaly detector. When anomalies are found, the Adaptive Threat Response Algorithm (ATRA) computes a risk score and determines the appropriate response (ALLOW / MONITOR / ALERT / SIMULATED_BLOCK).

```
Browser / Test Script
       ↓
Node.js Express Demo App (port 3000)
       ↓  [metadata only, no passwords/payloads]
Request Monitoring Middleware (monitor.js)
       ↓  [HTTP POST /telemetry, batched per IP]
Python Web Monitor Server (port 8765)
       ↓
Web Feature Extractor (20-dim vector)
       ↓
GTAE Encoder → Reconstruction Error → Anomaly Score
       ↓
Ensemble (IF + OCSVM + HBOS) → Anomaly/Normal
       ↓
ATRA Risk Scorer → Risk Level → Action
       ↓        ↘
JSONL Log     Socket.IO → React Dashboard (port 3001)
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm 9+

### Step 1: Install Python dependencies

```powershell
python -m pip install flask flask-socketio flask-cors torch scikit-learn joblib
```

### Step 2: Train the ML models

```powershell
python src/web_train.py
```

This will:
1. Generate 800 synthetic benign + 200 attack web sessions
2. Build the IP→endpoint graph with Laplacian PE
3. Train the GTAE autoencoder (benign-only)
4. Train the anomaly detector ensemble (IF + OCSVM + HBOS)
5. Train the attack-type classifier (RandomForest)
6. Save all models to `models/`

### Step 3: Start the Python Monitor Server

```powershell
python src/web_monitor_server.py
```

Listens on `http://127.0.0.1:8765`

### Step 4: Start the Demo Web Application

```powershell
cd webapp
npm install
npm start
```

Demo app runs on `http://localhost:3000`

### Step 5: Start the React Security Dashboard

```powershell
cd dashboard
npm install
npm start
```

Dashboard runs on `http://localhost:3001`

### Step 6: Generate Test Traffic

**Normal traffic:**
```powershell
python tests/traffic/generate_normal_traffic.py --duration 60 --rate 2
```

**Suspicious traffic (one scenario):**
```powershell
python tests/traffic/generate_attack_traffic.py --scenario brute_force
python tests/traffic/generate_attack_traffic.py --scenario rate_abuse
python tests/traffic/generate_attack_traffic.py --scenario path_traversal
python tests/traffic/generate_attack_traffic.py --scenario sqli
python tests/traffic/generate_attack_traffic.py --scenario sensitive_scan
```

**All scenarios:**
```powershell
python tests/traffic/generate_attack_traffic.py --scenario all
```

---

## Architecture

### Python Modules (src/)

| File | Purpose | Status |
|------|---------|--------|
| `gtae_model.py` | Graph Transformer Autoencoder | ✅ Reused |
| `atra.py` | Adaptive Threat Response | ✅ Reused + Extended |
| `anomaly_detector.py` | IF + OCSVM + HBOS ensemble | ✅ Reused |
| `attack_classifier.py` | RF attack type classifier | ✅ Reused |
| `graph_builder.py` | Graph construction + LPE | ✅ Reused |
| `utils.py` | Logging, metrics, plots | ✅ Reused |
| `config.py` | Central configuration | 🔄 Extended |
| `web_feature_extractor.py` | HTTP → 20-dim features | 🆕 New |
| `web_data_generator.py` | Synthetic web traffic | 🆕 New |
| `web_graph_builder.py` | Web graph construction | 🆕 New |
| `web_train.py` | Web ML training pipeline | 🆕 New |
| `web_monitor_server.py` | Real-time inference server | 🆕 New |

### Web Features (20-dimensional)

| # | Feature | Description |
|---|---------|-------------|
| 0 | request_rate | Requests per minute |
| 1 | unique_endpoint_count | Distinct paths visited |
| 2 | failed_login_count | POST /login with 401/403 |
| 3 | status_4xx_rate | Fraction of 4xx responses |
| 4 | status_5xx_rate | Fraction of 5xx responses |
| 5 | avg_response_time_ms | Mean latency |
| 6 | avg_path_length | Mean URL path length |
| 7 | avg_query_length | Mean query string length |
| 8 | post_ratio | Fraction of POST requests |
| 9 | get_ratio | Fraction of GET requests |
| 10 | sensitive_endpoint_count | Hits on /admin, /.env, etc. |
| 11 | repeated_endpoint_ratio | Most common endpoint / total |
| 12 | suspicious_pattern_count | SQLi/XSS/traversal patterns |
| 13 | abnormal_method_count | TRACE/PROPFIND/CONNECT etc. |
| 14 | avg_request_size | Mean request body size |
| 15 | max_request_rate_burst | Peak req/s in 5s sub-window |
| 16 | error_404_count | 404 error count |
| 17 | error_403_count | 403 error count |
| 18 | user_agent_entropy | Shannon entropy of user-agents |
| 19 | req_per_minute | Raw requests / window minutes |

### Attack Scenarios Covered

| Type | Example |
|------|---------|
| BruteForce | 40+ failed POST /login from one IP |
| RateLimitAbuse | 120+ rapid GETs from one IP |
| PathTraversal | Requests with ../../etc/passwd patterns |
| SQLInjection | URL queries with UNION SELECT etc. |
| XSS | URL queries with script/onerror= patterns |
| SensitiveEndpoint | Systematic scanning of /admin, /.env, etc. |
| BotScan | Systematic scanning for backup.sql, .git, etc. |
| AbnormalMethod | TRACE, PROPFIND, MOVE requests |
| LargePayload | POST with 50KB+ request bodies |
| Combined | Multi-vector attack pattern |

### ATRA Risk Actions

| Risk Level | Threshold | Action |
|-----------|-----------|--------|
| Low | 0–35 | ALLOW |
| Medium | 35–60 | MONITOR |
| High | 60–80 | ALERT |
| Critical | 80–100 | SIMULATED_BLOCK |

> ⚠️ **Disclaimer:** All blocking is simulated for demonstration purposes only. No real network blocking is performed.

---

## Ethical Note

- All test traffic is directed to the **local demo app only**
- No real attack tools, malware, or credential theft code is used
- No user data or real network traffic is collected
- The system never claims 100% detection accuracy
- Results are from controlled local tests only

---

## File Structure

```
IDS-Project/
├── src/
│   ├── gtae_model.py             (Graph Transformer Autoencoder)
│   ├── anomaly_detector.py       (IF + OCSVM + HBOS ensemble)
│   ├── atra.py                   (Adaptive Threat Response)
│   ├── attack_classifier.py      (RF attack type classifier)
│   ├── graph_builder.py          (Graph + LPE construction)
│   ├── config.py                 (Central configuration)
│   ├── utils.py                  (Logging, metrics, plots)
│   ├── web_feature_extractor.py  ← NEW
│   ├── web_data_generator.py     ← NEW
│   ├── web_graph_builder.py      ← NEW
│   ├── web_train.py              ← NEW
│   └── web_monitor_server.py     ← NEW
├── webapp/
│   ├── server.js                 (Express demo app)
│   ├── middleware/monitor.js     (HTTP telemetry middleware)
│   └── package.json
├── dashboard/
│   ├── src/App.js                (React security dashboard)
│   ├── src/hooks/useSocket.js    (Socket.IO hook)
│   └── package.json
├── tests/traffic/
│   ├── generate_normal_traffic.py
│   └── generate_attack_traffic.py
├── models/                       (trained model artifacts)
├── logs/                         (security event logs)
└── processed/                    (processed data)
```
