"""
demo_realtime.py -- Interactive Real-Time Flow Streaming Demo for GTAE-IDS + ATRA.

Streams network flows in real-time, executing full model inference:
  1. Forward pass through Graph Transformer Autoencoder (GTAE)
  2. Latent space scaling & Anomaly Detection (IF + OCSVM + HBOS ensemble)
  3. Attack-type classification (Random Forest)
  4. ATRA dynamic risk scoring with exponential time-decay history tracking
  5. Live automated defense response (Log -> Rate Limit -> Quarantine -> Block IP)
"""

import argparse
import os
import sys
import time
import pickle
import csv
from datetime import datetime
from pathlib import Path

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Reconfigure stdout for safe Windows terminal encoding
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


import torch
import numpy as np
import joblib

from config import DATA_CONFIG, GTAE_CONFIG, ATRA_CONFIG, MODELS_DIR, LOGS_DIR
from gtae_model import GTAE
from anomaly_detector import HBOSDetector, predict_all
from atra import (
    compute_risk_score,
    get_risk_level,
    normalize,
    IPHistoryTracker,
    RESPONSE_ACTIONS,
    ATTACK_SEVERITY,
    ATTACK_TYPE_WEIGHT,
)

# Terminal ANSI colors for rich real-time display
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_CYAN = "\033[96m"
COLOR_MAGENTA = "\033[95m"
COLOR_DIM = "\033[2m"


def load_artifacts():
    """Load trained models and dataset for real-time inference."""
    print(f"{COLOR_CYAN}{COLOR_BOLD}[INIT]{COLOR_RESET} Loading trained pipeline models...")

    # Load Graph Data
    graph_path = DATA_CONFIG.get("graph_data_path", "data/processed/graph_data.pkl")
    if not os.path.exists(graph_path):
        print(f"{COLOR_RED}[ERROR] Graph data not found at {graph_path}. Run main.py first.{COLOR_RESET}")
        sys.exit(1)

    with open(graph_path, "rb") as f:
        graph_data = pickle.load(f)

    # Load GTAE Model
    model_path = GTAE_CONFIG.get("model_path", "data/models/gtae_model.pt")
    in_features = graph_data["edge_attr"].shape[1]
    pe_dim = graph_data["edge_pe"].shape[1]
    d_model = GTAE_CONFIG.get("d_model", 64)
    num_heads = GTAE_CONFIG.get("num_heads", 4)
    num_layers = GTAE_CONFIG.get("num_layers", 3)

    gtae = GTAE(in_features=in_features, pe_dim=pe_dim, d_model=d_model, num_heads=num_heads, num_layers=num_layers)
    gtae.load_state_dict(torch.load(model_path, map_location="cpu"))
    gtae.eval()

    # Load Detectors and Scaler
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    iso_forest = joblib.load(os.path.join(MODELS_DIR, "iso_forest.joblib"))
    ocsvm = joblib.load(os.path.join(MODELS_DIR, "ocsvm.joblib"))
    hbos = joblib.load(os.path.join(MODELS_DIR, "hbos.joblib"))

    detectors = {"IF": iso_forest, "OCSVM": ocsvm, "HBOS": hbos}

    # Load Attack Classifier
    clf_path = os.path.join(MODELS_DIR, "attack_classifier.joblib")
    classifier = joblib.load(clf_path) if os.path.exists(clf_path) else None

    print(f"{COLOR_GREEN}[OK]{COLOR_RESET} GTAE model, 3 detectors (IF, OCSVM, HBOS), scaler, and classifier loaded.\n")
    return graph_data, gtae, scaler, detectors, classifier


def prepare_stream_queue(graph_data, count: int, burst: bool):
    """
    Build a realistic stream of flows with a balanced mix of benign traffic,
    various cyber attacks, and optional coordinated bursts to demonstrate escalation.
    """
    edge_attr = graph_data["edge_attr"]
    edge_pe = graph_data["edge_pe"]
    edge_index = graph_data["edge_index"]
    labels = graph_data["edge_labels"].numpy() if hasattr(graph_data["edge_labels"], "numpy") else np.array(graph_data["edge_labels"])
    attack_types = graph_data["attack_types"]
    node_to_idx = graph_data["node_to_idx"]
    idx_to_node = {v: k for k, v in node_to_idx.items()}

    benign_indices = np.where(labels == 0)[0]
    attack_indices = np.where(labels == 1)[0]

    np.random.seed(42)
    queue = []

    det_path = DATA_CONFIG.get("detection_results_path", "data/processed/detection_results.pkl")
    confirmed_anomalies = []
    if os.path.exists(det_path):
        try:
            with open(det_path, "rb") as f:
                det = pickle.load(f)
            confirmed_anomalies = [x["flow_id"] for x in det if x.get("ensemble_verdict") == "anomaly"]
        except Exception:
            confirmed_anomalies = []

    if confirmed_anomalies:
        ddos_candidates = [i for i in confirmed_anomalies if "DDoS" in attack_types[i] or "DoS" in attack_types[i]]
        portscan_candidates = [i for i in confirmed_anomalies if "PortScan" in attack_types[i]]
    else:
        ddos_candidates = [i for i in attack_indices if "DDoS" in attack_types[i] or "DoS" in attack_types[i]]
        portscan_candidates = [i for i in attack_indices if "PortScan" in attack_types[i]]

    if burst:
        # Construct an active threat escalation scenario:
        # Normal traffic -> PortScan probe -> DDoS flood burst (triggers escalation) -> Normal traffic
        burst_attacker_ip = "192.168.10.150"
        target_ip = "172.16.0.1"

        # 3 normal flows
        for idx in np.random.choice(benign_indices, size=min(3, len(benign_indices)), replace=False):
            queue.append((idx, None, None))

        # 2 initial PortScan probes
        if portscan_candidates:
            for idx in portscan_candidates[:2]:
                queue.append((idx, burst_attacker_ip, target_ip))

        # 8 rapid DDoS attacks from the same IP (demonstrates exponential escalation to CRITICAL / BLOCK_IP)
        if ddos_candidates:
            for idx in ddos_candidates[:8]:
                queue.append((idx, burst_attacker_ip, target_ip))

        # Fill remaining with interleaved normal and diverse attacks
        remaining = count - len(queue)
        if remaining > 0:
            for i in range(remaining):
                if i % 3 == 0:
                    idx = np.random.choice(attack_indices)
                else:
                    idx = np.random.choice(benign_indices)
                queue.append((idx, None, None))
    else:
        # Standard balanced realistic stream
        num_flows = count if count > 0 else 100
        for i in range(num_flows):
            # ~30% attacks, ~70% benign (realistic enterprise mix)
            if i % 3 == 0 and len(attack_indices) > 0:
                idx = np.random.choice(attack_indices)
            else:
                idx = np.random.choice(benign_indices)
            queue.append((idx, None, None))

    return queue, idx_to_node


def format_action_badge(action: str) -> str:
    """Colorize ATRA defensive response action."""
    action_upper = action.upper()
    if "BLOCK" in action_upper or "DROP" in action_upper:
        return f"{COLOR_RED}{COLOR_BOLD}[{action}]{COLOR_RESET}"
    elif "BLACKLIST" in action_upper or "QUARANTINE" in action_upper:
        return f"{COLOR_RED}[{action}]{COLOR_RESET}"
    elif "ALERT" in action_upper or "THROTTLE" in action_upper or "RATE_LIMIT" in action_upper:
        return f"{COLOR_YELLOW}{COLOR_BOLD}[{action}]{COLOR_RESET}"
    elif "LOG" in action_upper:
        return f"{COLOR_MAGENTA}[{action}]{COLOR_RESET}"
    else:
        return f"{COLOR_GREEN}[{action}]{COLOR_RESET}"


def format_risk_badge(level: str, score: float) -> str:
    """Colorize risk level badge."""
    if level == "Critical":
        return f"{COLOR_RED}{COLOR_BOLD}{level:<8s} ({score:4.1f}){COLOR_RESET}"
    elif level == "High":
        return f"{COLOR_RED}{level:<8s} ({score:4.1f}){COLOR_RESET}"
    elif level == "Medium":
        return f"{COLOR_YELLOW}{level:<8s} ({score:4.1f}){COLOR_RESET}"
    elif level == "Low":
        return f"{COLOR_MAGENTA}{level:<8s} ({score:4.1f}){COLOR_RESET}"
    else:
        return f"{COLOR_GREEN}{level:<8s} ({score:4.1f}){COLOR_RESET}"


def run_streaming_demo(speed: float = 5.0, count: int = 50, burst: bool = False):
    """Run the live real-time network flow streaming demonstration."""
    graph_data, gtae, scaler, detectors, classifier = load_artifacts()

    edge_attr_all = graph_data["edge_attr"]
    edge_pe_all = graph_data["edge_pe"]
    edge_index_all = graph_data["edge_index"]
    attack_types_all = graph_data["attack_types"]

    queue, idx_to_node = prepare_stream_queue(graph_data, count, burst)

    tracker = IPHistoryTracker(half_life_seconds=300)
    delay = 1.0 / speed if speed > 0 else 0.0

    # Ensure log output exists
    os.makedirs(LOGS_DIR, exist_ok=True)
    csv_log_path = os.path.join(LOGS_DIR, "realtime_stream.csv")
    csv_file = open(csv_log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "flow_id", "src_ip", "dst_ip", "verdict",
        "attack_type", "reconstruction_error", "risk_score", "risk_level", "action"
    ])

    print("=" * 112)
    print(f"{COLOR_BOLD}{COLOR_CYAN}  GTAE-IDS + ATRA: LIVE REAL-TIME FLOW STREAMING DEMONSTRATION{COLOR_RESET}")
    print(f"  Streaming Speed: {speed if speed > 0 else 'MAX'} flows/sec | Mode: {'BURST (Threat Escalation)' if burst else 'Standard Stream'}")
    print(f"  Live CSV Log   : {csv_log_path}")
    print("=" * 112)
    print(f"{COLOR_BOLD}{'TIME':<10s} {'SOURCE IP -> DESTINATION':<32s} {'VERDICT':<10s} {'CLASSIFICATION':<16s} {'RISK SCORE':<18s} {'RESPONSE ACTION'}{COLOR_RESET}")
    print("-" * 112)

    total_processed = 0
    benign_count = 0
    anomaly_count = 0
    blocked_count = 0
    total_latency_ms = 0.0

    try:
        for seq_id, (flow_idx, override_src, override_dst) in enumerate(queue):
            t_start = time.perf_counter()
            current_time = datetime.now()

            # Source and Destination IPs
            src_node_idx = edge_index_all[0][flow_idx].item()
            dst_node_idx = edge_index_all[1][flow_idx].item()
            src_node = idx_to_node.get(src_node_idx, f"node_{src_node_idx}")
            dst_node = idx_to_node.get(dst_node_idx, f"node_{dst_node_idx}")

            src_ip = str(override_src if override_src else src_node.split(":")[0]).encode("ascii", "replace").decode("ascii")
            dst_ip = str(override_dst if override_dst else dst_node.split(":")[0]).encode("ascii", "replace").decode("ascii")
            flow_endpoint = f"{src_ip} -> {dst_ip}"

            # Step 1: GTAE Forward Pass
            attr = edge_attr_all[flow_idx:flow_idx + 1]
            pe = edge_pe_all[flow_idx:flow_idx + 1]
            dst = torch.tensor([0], dtype=torch.long)

            with torch.no_grad():
                x_hat, z = gtae(attr, pe, dst)
                recon_err = float(torch.mean((x_hat - attr) ** 2).item())

            # Step 2: Anomaly Detection Ensemble
            z_np = z.cpu().numpy()
            z_scaled = scaler.transform(z_np)
            preds = predict_all(detectors, z_scaled)
            is_anomaly = bool(preds["ensemble"][0] == 1)

            # Step 3: Attack Classification & ATRA Response
            if not is_anomaly:
                verdict_str = f"{COLOR_GREEN}BENIGN{COLOR_RESET}"
                attack_type = "None"
                risk_score = 0.0
                risk_level = "None"
                action = "Pass"
                benign_count += 1
            else:
                verdict_str = f"{COLOR_RED}{COLOR_BOLD}ANOMALY{COLOR_RESET}"
                anomaly_count += 1

                if classifier is not None:
                    attack_type = str(classifier.predict(z_np)[0])
                else:
                    attack_type = attack_types_all[flow_idx]
                    if attack_type == "None":
                        attack_type = "Unknown"

                attack_type = attack_type.replace("\ufffd", "-").encode("ascii", "replace").decode("ascii")

                # ATRA scoring
                confidence = normalize(recon_err, 0.0001, 0.03, out_max=10.0)
                attack_weight = ATTACK_TYPE_WEIGHT.get(attack_type, 7.0)
                severity = ATTACK_SEVERITY.get(attack_type, 8.0)

                history_raw = tracker.get_decayed_history_score(src_ip, current_time)
                history_score = min(history_raw * 2.0, 10.0)
                freq_count = tracker.get_frequency_last_window(src_ip, current_time)
                frequency_score = min(freq_count * 2.5, 10.0)

                risk_score = compute_risk_score(attack_weight, severity, confidence, history_score, frequency_score)
                risk_level = get_risk_level(risk_score)
                action = RESPONSE_ACTIONS.get(risk_level, "Generate Log")

                tracker.record_attack(src_ip, current_time)
                if "BLOCK" in action.upper() or "BLACKLIST" in action.upper():
                    blocked_count += 1

            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            total_latency_ms += t_elapsed_ms
            total_processed += 1

            # Log to CSV
            csv_writer.writerow([
                current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                flow_idx, src_ip, dst_ip,
                "Anomaly" if is_anomaly else "Benign",
                attack_type, f"{recon_err:.5f}", f"{risk_score:.2f}",
                risk_level, action
            ])
            csv_file.flush()

            # Live Terminal Output
            time_str = current_time.strftime("%H:%M:%S")
            risk_badge = format_risk_badge(risk_level, risk_score)
            action_badge = format_action_badge(action)

            print(
                f"{time_str:<10s} "
                f"{flow_endpoint:<32s} "
                f"{verdict_str:<19s} "
                f"{attack_type:<16s} "
                f"{risk_badge:<26s} "
                f"{action_badge}"
            )

            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        print(f"\n{COLOR_YELLOW}[INTERRUPT] Streaming stopped by user.{COLOR_RESET}")

    finally:
        csv_file.close()

    # Final summary statistics
    avg_latency = total_latency_ms / total_processed if total_processed > 0 else 0.0
    print("-" * 112)
    print(f"{COLOR_BOLD}{COLOR_CYAN}STREAMING SESSION SUMMARY:{COLOR_RESET}")
    print(f"  Total Flows Evaluated: {total_processed}")
    print(f"  Benign Flows Passed  : {COLOR_GREEN}{benign_count}{COLOR_RESET}")
    print(f"  Anomalies Detected   : {COLOR_RED}{anomaly_count}{COLOR_RESET}")
    print(f"  Active IP Blocks     : {COLOR_RED}{COLOR_BOLD}{blocked_count}{COLOR_RESET}")
    print(f"  Avg Inference Latency: {COLOR_BOLD}{avg_latency:.2f} ms / flow{COLOR_RESET}")
    print(f"  Session Log File     : {csv_log_path}")
    print("=" * 112)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Network Flow Streaming Demo")
    parser.add_argument("--speed", type=float, default=5.0,
                        help="Flows per second (e.g. 5.0, 10.0, or 0 for max throughput)")
    parser.add_argument("--count", type=int, default=30,
                        help="Total number of flows to stream (default: 30)")
    parser.add_argument("--burst", action="store_true",
                        help="Inject a coordinated attack burst to demonstrate ATRA exponential threat escalation")
    args = parser.parse_args()

    run_streaming_demo(speed=args.speed, count=args.count, burst=args.burst)


if __name__ == "__main__":
    main()
