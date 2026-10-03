"""
src/data_loader_synthetic.py
Generates synthetic network traffic flows mimicking the GTAE-IDS structure.
Self-contained: Requires NO external dataset download!

- Produces 400 benign flows and 100 attack flows across 3 distinct attack profiles:
    * DDoS: high-volume burst traffic
    * DoS Hulk: sustained heavy payload
    * PortScan: sparse probing packets
- Includes repeat offender IPs to exercise ATRA time-decay and escalation.
- Saves directly to DATA_CONFIG["benign_train_path"] and DATA_CONFIG["attack_test_path"].
"""

import sys
from pathlib import Path
import random
import numpy as np
import pickle

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_CONFIG
from utils import get_logger

logger = get_logger(__name__)

NUM_BENIGN_FLOWS = 400
NUM_ATTACK_FLOWS = 100
PACKETS_PER_FLOW = 8
FEATURES_PER_PACKET = 25
ATTACK_TYPES = ["DoS Hulk", "PortScan", "DDoS"]

random.seed(42)
np.random.seed(42)


def random_ip():
    return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def random_port():
    return random.randint(1024, 65535)


def generate_packet_features(is_attack: bool, attack_type: str = None):
    if not is_attack:
        base = np.random.uniform(0.05, 0.45, size=FEATURES_PER_PACKET)
        noise = np.random.normal(0, 0.05, size=FEATURES_PER_PACKET)
    elif attack_type == "DDoS":
        base = np.random.uniform(0.60, 0.95, size=FEATURES_PER_PACKET)
        noise = np.random.normal(0, 0.15, size=FEATURES_PER_PACKET)
    elif attack_type == "DoS Hulk":
        base = np.random.uniform(0.70, 0.98, size=FEATURES_PER_PACKET)
        noise = np.random.normal(0, 0.08, size=FEATURES_PER_PACKET)
        base[:8] = np.random.uniform(0.85, 1.0, size=8)
    elif attack_type == "PortScan":
        base = np.random.uniform(0.40, 0.65, size=FEATURES_PER_PACKET)
        noise = np.random.normal(0, 0.06, size=FEATURES_PER_PACKET)
        mask = np.random.choice(FEATURES_PER_PACKET, size=15, replace=False)
        base[mask] = np.random.uniform(0.02, 0.15, size=len(mask))
    else:
        base = np.random.uniform(0.55, 0.90, size=FEATURES_PER_PACKET)
        noise = np.random.normal(0, 0.12, size=FEATURES_PER_PACKET)

    features = np.clip(base + noise, 0, 1)
    return features


def generate_flow_features(is_attack: bool, attack_type: str = None):
    feats = []
    for _ in range(PACKETS_PER_FLOW * 2):  # fwd and bwd
        feats.extend(generate_packet_features(is_attack, attack_type))
    return np.array(feats, dtype=np.float32)


def generate_records():
    benign_records = []
    attack_records = []
    flow_id = 0

    # 1. Benign flows
    for _ in range(NUM_BENIGN_FLOWS):
        src = f"{random_ip()}:{random_port()}"
        dst = f"{random_ip()}:{random_port()}"
        feats = generate_flow_features(is_attack=False)
        benign_records.append({
            "flow_id": flow_id,
            "src_node": src,
            "dst_node": dst,
            "edge_features": feats,
            "label": "benign",
            "attack_type": "None",
        })
        flow_id += 1

    # 2. Repeat offender IPs for ATRA testing
    NUM_REPEAT_OFFENDERS = 3
    repeat_offenders = [(random_ip(), random_port()) for _ in range(NUM_REPEAT_OFFENDERS)]

    for _ in range(NUM_ATTACK_FLOWS):
        atk_type = random.choice(ATTACK_TYPES)
        if random.random() < 0.4:
            src_ip, src_port = random.choice(repeat_offenders)
            src = f"{src_ip}:{src_port}"
        else:
            src = f"{random_ip()}:{random_port()}"

        dst = f"{random_ip()}:{random_port()}"
        feats = generate_flow_features(is_attack=True, attack_type=atk_type)

        attack_records.append({
            "flow_id": flow_id,
            "src_node": src,
            "dst_node": dst,
            "edge_features": feats,
            "label": "attack",
            "attack_type": atk_type,
        })
        flow_id += 1

    return benign_records, attack_records


def main():
    logger.info("Generating self-contained synthetic traffic dataset...")
    benign_records, attack_records = generate_records()

    benign_path = Path(DATA_CONFIG["benign_train_path"])
    attack_path = Path(DATA_CONFIG["attack_test_path"])

    benign_path.parent.mkdir(parents=True, exist_ok=True)
    attack_path.parent.mkdir(parents=True, exist_ok=True)

    with open(benign_path, "wb") as f:
        pickle.dump(benign_records, f)
    with open(attack_path, "wb") as f:
        pickle.dump(attack_records, f)

    print(f"\n[OK] Generated {len(benign_records)} benign records -> {benign_path}")
    print(f"[OK] Generated {len(attack_records)} attack records -> {attack_path}")
    print(f"Edge feature length: {benign_records[0]['edge_features'].shape[0]} dims")

    types, counts = np.unique([r["attack_type"] for r in attack_records], return_counts=True)
    for t, c in zip(types, counts):
        print(f"  {t:<20}: {c}")


if __name__ == "__main__":
    main()

