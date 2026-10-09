# System Performance & Academic Evaluation Report

**Project Title:** GTAE-IDS + ATRA: Real-Time Intrusion Detection & Adaptive Threat Response  
**Evaluation Timestamp:** 2026-10-08 05:22:48  
**Detection Model:** Graph Transformer Autoencoder (GTAE) + Multi-Detector Ensemble (IF + OCSVM + HBOS)  
**Response Model:** Adaptive Threat Response Algorithm (ATRA)  

---

## 1. Executive Summary

This report documents the performance of the GTAE-IDS detection engine coupled with the Adaptive Threat Response Algorithm (ATRA). The system was evaluated on network flow traffic, achieving robust separation of malicious activities with automated, tiered response escalation.

---

## 2. Core Detection Performance

| Metric | Measured Value | Target Standard | Status |
|---|:---:|:---:|:---:|
| **True Positive Rate (TPR / Recall)** | **99.52%** | > 90% | EXCELLENT |
| **False Positive Rate (FPR)** | **0.04%** | < 15% | WITHIN SPEC |
| **Precision** | **99.96%** | > 60% | ROBUST |
| **F1 Score** | **0.997** | > 0.75 | PASS |

### Confusion Matrix Counts
- **True Positives (TP)**: 5541
- **False Positives (FP)**: 2
- **True Negatives (TN)**: 4998
- **False Negatives (FN)**: 27

---

## 3. Detection Breakdown by Attack Type

| Attack Category | Detection Rate (Detected / Total) |
|---|---|
| Bot                       | 497/500 ( 99.4%) |
| DDoS                      | 500/500 (100.0%) |
| DoS GoldenEye             | 498/500 ( 99.6%) |
| DoS Hulk                  | 500/500 (100.0%) |
| DoS Slowhttptest          | 500/500 (100.0%) |
| DoS slowloris             | 498/500 ( 99.6%) |
| FTP-Patator               | 498/500 ( 99.6%) |
| Heartbleed                | 11/11 (100.0%) |
| Infiltration              | 24/36 ( 66.7%) |
| PortScan                  | 499/500 ( 99.8%) |
| SSH-Patator               | 497/500 ( 99.4%) |
| Web Attack - Brute Force  | 500/500 (100.0%) |
| Web Attack - Sql Injection | 20/21 ( 95.2%) |
| Web Attack - XSS          | 499/500 ( 99.8%) |


---

## 4. ATRA Adaptive Response Distribution

ATRA applies dynamic exponential time decay ($S = \sum e^{-\lambda \Delta t}$) to escalate repeat offender IPs from benign logging up to immediate blacklisting and active packet dropping.

### Response Tiers
- **Low Risk**: 369 flows
- **Medium Risk**: 1803 flows
- **High Risk**: 8 flows
- **Critical Risk**: 0 flows


### Key Generated Artifacts
- **Confusion Matrix**: `results/confusion_matrix.png`
- **Risk Score Distribution**: `results/risk_score_distribution.png`
- **Response Actions Taken**: `results/action_distribution.png`
- **Attack Event Log**: `logs/attack_log.csv`

---

## 5. Architectural Highlights

1. **Leak-Free Training**: All normalization scalers and Laplacian Positional Encodings (LPE) are fitted strictly on benign subgraphs.
2. **Tri-Detector Ensemble**: Uses Isolation Forest, One-Class SVM, and Histogram-Based Outlier Score (HBOS).
3. **Proportional Risk Response**: Lower confidence alerts land in Low/Medium tiers (Generate Log / Alert Admin), while persistent attack patterns trigger High/Critical tiers (Blacklist IP / Block).
