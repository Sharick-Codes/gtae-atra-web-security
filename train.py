import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import DATA_CONFIG, GTAE_CONFIG, MODEL_CONFIG
from utils import get_logger
import gtae_model
import anomaly_detector
import attack_classifier

logger = get_logger(__name__)

def main():
    """Train the GTAE model, anomaly detectors, and attack classifier."""
    parser = argparse.ArgumentParser(description="Standalone training script.")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train GTAE")
    parser.add_argument("--lr", type=float, help="Learning rate for GTAE")
    args = parser.parse_args()

    graph_path = DATA_CONFIG.get("graph_data_path", "data/processed/graph_data.pkl")
    if not os.path.exists(graph_path):
        print(f"Error: Run main.py first to preprocess data. File not found: {graph_path}")
        sys.exit(1)

    if args.epochs is not None:
        GTAE_CONFIG["epochs"] = args.epochs
    if args.lr is not None:
        GTAE_CONFIG["learning_rate"] = args.lr

    logger.info("=== STEP 1/3: Training GTAE Autoencoder ===")
    gtae_model.main()

    logger.info("=== STEP 2/3: Training Anomaly Detectors ===")
    anomaly_detector.main()

    logger.info("=== STEP 3/3: Training Attack Classifier ===")
    attack_classifier.main()

    print("\nTraining summary:")
    print("GTAE Model trained successfully.")
    print("Anomaly detectors trained successfully.")
    print("Attack classifier trained successfully.")

if __name__ == "__main__":
    main()
