"""Real-time packet capture, flow collection, and inference."""
import csv
import json
import os
import pickle
import numpy as np
import time
import joblib
import torch
from datetime import datetime
from scapy.all import sniff, IP, TCP, get_if_list
from collections import defaultdict
import signal

from src.config import DATA_CONFIG, GTAE_CONFIG, MODELS_DIR, LOGS_DIR
from src.gtae_model import GTAE
from src.anomaly_detector import HBOSDetector, predict_all
from src.atra import (
    ATTACK_SEVERITY,
    ATTACK_TYPE_WEIGHT,
    IPHistoryTracker,
    RESPONSE_ACTIONS,
    compute_risk_score,
    get_risk_level,
    normalize,
)
from src.utils import get_logger

logger = get_logger(__name__)

FEATURES_PER_PACKET = 25
PACKETS_PER_DIRECTION = 8
LIVE_FEATURE_DIM = FEATURES_PER_PACKET * PACKETS_PER_DIRECTION * 2

_stop_requested = False

def _handle_sigint(signum, frame):
    global _stop_requested
    _stop_requested = True


def is_stop_requested() -> bool:
    """Return whether the current capture should stop."""
    return _stop_requested


def list_interfaces() -> list[str]:
    """Return capture interfaces reported by Scapy/Npcap."""
    return list(get_if_list())

signal.signal(signal.SIGINT, _handle_sigint)

class FlowCollector:
    """Accumulates packets into flows during a capture window."""
    def __init__(self):
        self.flows = defaultdict(list)
        self.flow_first_seen = {}

    def handle_packet(self, pkt):
        """Process incoming packet."""
        if IP not in pkt or TCP not in pkt:
            return
        src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
        src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport

        key = (src_ip, src_port, dst_ip, dst_port)
        rev_key = (dst_ip, dst_port, src_ip, src_port)

        if rev_key in self.flows:
            direction = "bwd"
            flow_key = rev_key
        else:
            direction = "fwd"
            flow_key = key
            if flow_key not in self.flow_first_seen:
                self.flow_first_seen[flow_key] = (src_ip, src_port, dst_ip, dst_port)

        timestamp = float(getattr(pkt, "time", time.time()))
        tcp_flags = str(pkt[TCP].flags)
        packet_length = int(len(pkt))
        payload_length = int(len(bytes(pkt[TCP].payload)))
        self.flows[flow_key].append({
            "direction": direction,
            "timestamp": timestamp,
            "length": packet_length,
            "payload_length": payload_length,
            "tcp_flags": tcp_flags,
        })

    def get_completed_flows(self):
        """Return flows with at least 2 packets."""
        results = []
        for flow_key, packets in self.flows.items():
            if len(packets) < 2:
                continue
            src_ip, src_port, dst_ip, dst_port = self.flow_first_seen.get(flow_key, flow_key)
            packets = sorted(self.flows[flow_key], key=lambda packet: packet["timestamp"])
            forward_packets = [p for p in packets if p["direction"] == "fwd"]
            backward_packets = [p for p in packets if p["direction"] == "bwd"]
            results.append({
                "src_ip": src_ip, "src_port": src_port,
                "dst_ip": dst_ip, "dst_port": dst_port,
                "packets": packets,
                "first_seen": packets[0]["timestamp"],
                "last_seen": packets[-1]["timestamp"],
                "duration": max(0.0, packets[-1]["timestamp"] - packets[0]["timestamp"]),
                "forward_packet_count": len(forward_packets),
                "backward_packet_count": len(backward_packets),
                "forward_bytes": sum(p["length"] for p in forward_packets),
                "backward_bytes": sum(p["length"] for p in backward_packets),
            })
        return results


def _packet_feature_vector(packet: dict, flow_start: float) -> list[float]:
    """Encode one captured packet into the live 25-value packet schema."""
    flag_names = ("F", "S", "R", "P", "A", "U", "E", "C")
    flags = packet["tcp_flags"]
    elapsed = max(0.0, packet["timestamp"] - flow_start)
    values = [
        min(packet["length"] / 1500.0, 1.0),
        min(packet["payload_length"] / 1500.0, 1.0),
        min(elapsed / 10.0, 1.0),
    ]
    values.extend(1.0 if name in flags else 0.0 for name in flag_names)
    values.extend([
        min(packet["length"] / 65535.0, 1.0),
        min(packet["payload_length"] / 65535.0, 1.0),
        min(elapsed / 60.0, 1.0),
    ])
    values.extend([0.0] * (FEATURES_PER_PACKET - len(values)))
    return values


def encode_flow_features(flow: dict) -> np.ndarray:
    """Encode a captured flow into the 400-value online feature tensor.

    The current synthetic trainer uses anonymous 25-value packet slots. This
    deterministic adapter preserves the tensor contract, but live predictions
    require retraining on these features before production use.
    """
    flow_start = float(flow["first_seen"])
    ordered_packets = {
        "fwd": [p for p in flow["packets"] if p["direction"] == "fwd"],
        "bwd": [p for p in flow["packets"] if p["direction"] == "bwd"],
    }
    values = []
    for direction in ("fwd", "bwd"):
        packets = ordered_packets[direction][:PACKETS_PER_DIRECTION]
        for packet in packets:
            values.extend(_packet_feature_vector(packet, flow_start))
        missing = PACKETS_PER_DIRECTION - len(packets)
        values.extend([0.0] * (missing * FEATURES_PER_PACKET))
    return np.asarray(values, dtype=np.float32)


def _flow_statistics(flow: dict) -> dict:
    """Build CIC-style aggregate statistics available from captured packets."""
    packets = flow["packets"]
    forward = [p for p in packets if p["direction"] == "fwd"]
    backward = [p for p in packets if p["direction"] == "bwd"]
    all_lengths = [p["length"] for p in packets]
    fwd_lengths = [p["length"] for p in forward]
    bwd_lengths = [p["length"] for p in backward]
    timestamps = [p["timestamp"] for p in packets]
    iats = np.diff(timestamps) if len(timestamps) > 1 else np.array([], dtype=float)
    duration_us = flow["duration"] * 1_000_000.0

    def stats(values):
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        return {
            "min": float(np.min(values)), "max": float(np.max(values)),
            "mean": float(np.mean(values)), "std": float(np.std(values)),
        }

    fwd_stats = stats(fwd_lengths)
    bwd_stats = stats(bwd_lengths)
    all_stats = stats(all_lengths)
    iat_stats = stats(iats.tolist())
    fwd_iats = np.diff([p["timestamp"] for p in forward]) if len(forward) > 1 else np.array([])
    bwd_iats = np.diff([p["timestamp"] for p in backward]) if len(backward) > 1 else np.array([])
    fwd_iat_stats = stats(fwd_iats.tolist())
    bwd_iat_stats = stats(bwd_iats.tolist())
    flag_count = lambda flag, values: sum(flag in packet["tcp_flags"] for packet in values)
    total_bytes = sum(all_lengths)
    total_packets = len(packets)
    duration_s = flow["duration"]
    return {
        "Destination Port": flow["dst_port"], "Flow Duration": duration_us,
        "Total Fwd Packets": len(forward), "Total Backward Packets": len(backward),
        "Total Length of Fwd Packets": sum(fwd_lengths), "Total Length of Bwd Packets": sum(bwd_lengths),
        "Fwd Packet Length Max": fwd_stats["max"], "Fwd Packet Length Min": fwd_stats["min"],
        "Fwd Packet Length Mean": fwd_stats["mean"], "Fwd Packet Length Std": fwd_stats["std"],
        "Bwd Packet Length Max": bwd_stats["max"], "Bwd Packet Length Min": bwd_stats["min"],
        "Bwd Packet Length Mean": bwd_stats["mean"], "Bwd Packet Length Std": bwd_stats["std"],
        "Flow Bytes/s": total_bytes / duration_s if duration_s else 0.0,
        "Flow Packets/s": total_packets / duration_s if duration_s else 0.0,
        "Flow IAT Mean": iat_stats["mean"] * 1_000_000.0,
        "Flow IAT Std": iat_stats["std"] * 1_000_000.0,
        "Flow IAT Max": iat_stats["max"] * 1_000_000.0,
        "Flow IAT Min": iat_stats["min"] * 1_000_000.0,
        "Fwd IAT Total": sum(fwd_iats) * 1_000_000.0 if len(fwd_iats) else 0.0,
        "Fwd IAT Mean": fwd_iat_stats["mean"] * 1_000_000.0,
        "Fwd IAT Std": fwd_iat_stats["std"] * 1_000_000.0,
        "Fwd IAT Max": fwd_iat_stats["max"] * 1_000_000.0,
        "Fwd IAT Min": fwd_iat_stats["min"] * 1_000_000.0,
        "Bwd IAT Total": sum(bwd_iats) * 1_000_000.0 if len(bwd_iats) else 0.0,
        "Bwd IAT Mean": bwd_iat_stats["mean"] * 1_000_000.0,
        "Bwd IAT Std": bwd_iat_stats["std"] * 1_000_000.0,
        "Bwd IAT Max": bwd_iat_stats["max"] * 1_000_000.0,
        "Bwd IAT Min": bwd_iat_stats["min"] * 1_000_000.0,
        "Fwd PSH Flags": flag_count("P", forward), "Bwd PSH Flags": flag_count("P", backward),
        "Fwd URG Flags": flag_count("U", forward), "Bwd URG Flags": flag_count("U", backward),
        "Fwd Header Length": len(forward) * 20, "Bwd Header Length": len(backward) * 20,
        "Fwd Packets/s": len(forward) / duration_s if duration_s else 0.0,
        "Bwd Packets/s": len(backward) / duration_s if duration_s else 0.0,
        "Min Packet Length": all_stats["min"], "Max Packet Length": all_stats["max"],
        "Packet Length Mean": all_stats["mean"], "Packet Length Std": all_stats["std"],
        "Packet Length Variance": all_stats["std"] ** 2,
        "FIN Flag Count": flag_count("F", packets), "SYN Flag Count": flag_count("S", packets),
        "RST Flag Count": flag_count("R", packets), "PSH Flag Count": flag_count("P", packets),
        "ACK Flag Count": flag_count("A", packets), "URG Flag Count": flag_count("U", packets),
        "CWE Flag Count": flag_count("C", packets), "ECE Flag Count": flag_count("E", packets),
        "Down/Up Ratio": len(backward) / len(forward) if forward else 0.0,
        "Average Packet Size": all_stats["mean"], "Avg Fwd Segment Size": fwd_stats["mean"],
        "Avg Bwd Segment Size": bwd_stats["mean"],
        "Subflow Fwd Packets": len(forward), "Subflow Fwd Bytes": sum(fwd_lengths),
        "Subflow Bwd Packets": len(backward), "Subflow Bwd Bytes": sum(bwd_lengths),
        "act_data_pkt_fwd": sum(p["payload_length"] > 0 for p in forward),
        "min_seg_size_forward": fwd_stats["min"],
    }


def encode_cic_flow_features(flow: dict, schema_path: str) -> np.ndarray:
    """Encode a live flow in the persisted CIC-IDS2017 feature order."""
    with open(schema_path, "r", encoding="utf-8") as file:
        schema = json.load(file)
    statistics = _flow_statistics(flow)
    values = []
    for name in schema["feature_cols"]:
        raw_value = statistics.get(name, 0.0)
        minimum = schema["benign_min"].get(name, 0.0)
        maximum = schema["benign_max"].get(name, minimum + 1.0)
        values.append(float(np.clip((raw_value - minimum) / (maximum - minimum or 1.0), 0.0, 1.0)))
    return np.asarray(values, dtype=np.float32)

def capture_window(duration_seconds: int = 10, iface: str | None = None):
    """Capture live traffic for duration_seconds."""
    global _stop_requested
    _stop_requested = False
    collector = FlowCollector()
    logger.info(f"Capturing live traffic for {duration_seconds}s...")
    try:
        sniff(
            prn=collector.handle_packet,
            iface=iface,
            timeout=duration_seconds,
            store=False,
            stop_filter=lambda pkt: _stop_requested,
        )
    except PermissionError as exc:
        raise RuntimeError(
            "Packet capture permission denied. Run PowerShell as Administrator "
            "and verify that Npcap is installed."
        ) from exc
    except OSError as exc:
        if iface:
            raise RuntimeError(f"Unable to capture on interface {iface!r}: {exc}") from exc
        raise RuntimeError(
            "Unable to start packet capture. Verify that Npcap is installed."
        ) from exc
    flows = collector.get_completed_flows()
    return flows

class RTInferenceEngine:
    def __init__(self, no_response: bool = False):
        """Load trained artifacts for one-flow-at-a-time live inference."""
        graph_path = DATA_CONFIG["graph_data_path"]
        model_path = GTAE_CONFIG["model_path"]
        if not os.path.exists(graph_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                "Trained graph/model artifacts are missing. Run "
                "'python main.py' or 'python main.py --synthetic' first."
            )
        with open(graph_path, "rb") as file:
            graph_data = pickle.load(file)
        edge_features = int(graph_data["edge_attr"].shape[1])
        self.edge_features = edge_features
        self.schema_path = os.path.join(MODELS_DIR, "cic_feature_schema.json")
        if edge_features == 78 and not os.path.exists(self.schema_path):
            raise FileNotFoundError(
                "CIC live feature schema is missing. Retrain with 'python main.py' "
                "to create data/models/cic_feature_schema.json."
            )
        pe_dim = int(graph_data["edge_pe"].shape[1])
        self.model = GTAE(
            in_features=edge_features,
            pe_dim=pe_dim,
            d_model=GTAE_CONFIG.get("d_model", 64),
            num_heads=GTAE_CONFIG.get("num_heads", 4),
            num_layers=GTAE_CONFIG.get("num_layers", 3),
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        import __main__
        __main__.HBOSDetector = HBOSDetector
        self.detectors = {
            "IF": joblib.load(os.path.join(MODELS_DIR, "iso_forest.joblib")),
            "OCSVM": joblib.load(os.path.join(MODELS_DIR, "ocsvm.joblib")),
            "HBOS": joblib.load(os.path.join(MODELS_DIR, "hbos.joblib")),
        }
        classifier_path = os.path.join(MODELS_DIR, "attack_classifier.joblib")
        self.classifier = joblib.load(classifier_path) if os.path.exists(classifier_path) else None
        self.pe_dim = pe_dim
        self.tracker = IPHistoryTracker(half_life_seconds=300)
        self.no_response = no_response
        self.log_path = os.path.join(LOGS_DIR, "realtime_stream.csv")
        self._ensure_log()
        logger.info("Loaded models for real-time inference.")

    def _ensure_log(self):
        """Create the live log with a stable header when it does not exist."""
        if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0:
            return
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(self.log_path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                "timestamp", "src_ip", "src_port", "dst_ip", "dst_port",
                "verdict", "attack_type", "reconstruction_error", "risk_score",
                "risk_level", "action",
            ])

    def encode_flow(self, flow: dict) -> np.ndarray:
        """Encode flow packets into a fixed-dimensional array."""
        if self.edge_features == 78:
            return encode_cic_flow_features(flow, self.schema_path)
        return encode_flow_features(flow)

    def _process_with_models(self, flows: list):
        """Run GTAE, anomaly detection, classification, ATRA, and logging."""
        if not flows:
            logger.info("No flows this window.")
            return []
        results = []
        for flow in flows:
            current_time = datetime.now()
            attr = torch.from_numpy(self.encode_flow(flow)).reshape(1, -1)
            pe = torch.zeros((1, self.pe_dim), dtype=torch.float32)
            dst = torch.zeros(1, dtype=torch.long)
            with torch.no_grad():
                reconstructed, embedding = self.model(attr, pe, dst)
                reconstruction_error = float(torch.mean((reconstructed - attr) ** 2).item())
            embedding_np = embedding.cpu().numpy()
            predictions = predict_all(self.detectors, self.scaler.transform(embedding_np))
            is_anomaly = bool(predictions["ensemble"][0] == 1)
            attack_type = "None"
            risk_score = 0.0
            risk_level = "None"
            action = "Pass"
            if is_anomaly:
                attack_type = str(self.classifier.predict(embedding_np)[0]) if self.classifier is not None else "Unknown"
                confidence = normalize(reconstruction_error, 0.0001, 0.03, out_max=10.0)
                history_score = min(self.tracker.get_decayed_history_score(flow["src_ip"], current_time) * 2.0, 10.0)
                frequency_score = min(self.tracker.get_frequency_last_window(flow["src_ip"], current_time) * 2.5, 10.0)
                risk_score = compute_risk_score(
                    ATTACK_TYPE_WEIGHT.get(attack_type, 7.0),
                    ATTACK_SEVERITY.get(attack_type, 8.0),
                    confidence, history_score, frequency_score,
                )
                risk_level = get_risk_level(risk_score)
                action = "Generate Log" if self.no_response else RESPONSE_ACTIONS.get(risk_level, "Generate Log")
                self.tracker.record_attack(flow["src_ip"], current_time)
            result = {
                "timestamp": current_time.isoformat(timespec="milliseconds"),
                "src_ip": flow["src_ip"], "src_port": flow["src_port"],
                "dst_ip": flow["dst_ip"], "dst_port": flow["dst_port"],
                "verdict": "Anomaly" if is_anomaly else "Benign",
                "attack_type": attack_type,
                "reconstruction_error": reconstruction_error,
                "risk_score": risk_score, "risk_level": risk_level, "action": action,
            }
            with open(self.log_path, "a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(result.values())
            results.append(result)
            print(
                f"[{current_time.strftime('%H:%M:%S')}] {flow['src_ip']} -> {flow['dst_ip']} | "
                f"{result['verdict']} {attack_type} | {risk_level} ({risk_score:.1f}) | {action}"
            )
        return results

    def process_window(self, flows: list):
        """Process a batch of completed flows."""
        return self._process_with_models(flows)
        if not flows:
            logger.info("No flows this window.")
            return
        
        logger.info(f"Processing {len(flows)} flows...")
        # Add processing logic, ATRA scoring, and verdicts here.
        for flow in flows:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {flow['src_ip']} -> {flow['dst_ip']} | Processed")

