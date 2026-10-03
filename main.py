"""
main.py -- Single entry point for the IDS-Project pipeline.

Runs all 7 pipeline steps in order via subprocess:
  1. Data loading & preprocessing  (Synthetic or CIC-IDS2017)
  2. Graph construction            (with benign-only LPE)
  3. GTAE training & inference     (autoencoder embeddings)
  4. Anomaly detection             (IF + OCSVM + HBOS ensemble)
  5. Attack type classification    (RandomForest)
  6. ATRA risk scoring & response  (adaptive threat response)
  7. Attack logging                (CSV + JSON summary)

Usage:
    python main.py                 # Full pipeline (auto-detects real or synthetic)
    python main.py --synthetic     # Force synthetic traffic generator (no dataset needed)
    python main.py --skip-data     # Re-use existing processed data
"""

import argparse
import subprocess
import sys
import os
import pickle
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import print_config, DATA_CONFIG, RAW_DATA_DIR


def check_raw_dataset_exists():
    """Check if CIC-IDS2017 raw CSV files exist in data/raw."""
    raw_path = Path(RAW_DATA_DIR)
    if not raw_path.exists():
        return False
    csvs = list(raw_path.glob("*.csv"))
    return len(csvs) > 0


def run_step(description: str, script_path: str):
    """Run a pipeline step as a subprocess with proper PYTHONPATH."""
    print(f"\n{'=' * 70}", flush=True)
    print(f"STEP: {description}", flush=True)
    print(f"{'=' * 70}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    env = os.environ.copy()
    src_dir = str(Path(__file__).parent / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(
        [sys.executable, "-u", script_path],
        env=env,
        cwd=str(Path(__file__).parent),
    )
    sys.stdout.flush()
    sys.stderr.flush()
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {script_path}", flush=True)
        sys.exit(1)


def print_final_summary():
    """Load ATRA results and print a clean pipeline summary."""
    atra_path = DATA_CONFIG["atra_results_path"]
    try:
        with open(atra_path, "rb") as f:
            atra_results = pickle.load(f)
    except FileNotFoundError:
        print(f"[WARN] Could not load ATRA results from {atra_path}")
        return

    total = len(atra_results)
    benign = sum(1 for r in atra_results if r.get("risk_level") in (None, "None"))
    anomalous = total - benign

    print(f"\n{'=' * 70}")
    print("==== IDS-PROJECT PIPELINE SUMMARY ====")
    print(f"{'=' * 70}")
    print(f"Total flows processed: {total:,}")
    print(f"Benign: {benign:,}  |  Anomalous: {anomalous:,}")
    for level in ["Low", "Medium", "High", "Critical"]:
        count = sum(1 for r in atra_results if r.get("risk_level") == level)
        print(f"  {level:10s} risk: {count:,}")
    print(f"\nOutputs:")
    print(f"  - logs/attack_log.csv")
    print(f"  - results/evaluation_results.json")
    print(f"  - data/models/gtae_model.pt")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Run the full IDS pipeline.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic data generator (works with zero dataset files)")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip Step 1 (re-use existing processed data)")
    args = parser.parse_args()

    print_config()

    # Determine data loader script
    has_raw = check_raw_dataset_exists()
    if args.synthetic or not has_raw:
        data_script = "src/data_loader_synthetic.py"
        data_desc = "Generating synthetic traffic flows (self-contained, zero external dataset required)"
    else:
        data_script = "src/data_loader.py"
        data_desc = "Loading & preprocessing CIC-IDS2017 raw CSV data"

    steps = [
        (data_desc,                               data_script),
        ("Constructing graph",                    "src/graph_builder.py"),
        ("Training GTAE (autoencoder)",           "src/gtae_model.py"),
        ("Running anomaly detectors (ensemble)",   "src/anomaly_detector.py"),
        ("Training attack-type classifier",        "src/attack_classifier.py"),
        ("Running ATRA (risk scoring + response)", "src/atra.py"),
        ("Writing attack log",                     "src/attack_logger.py"),
    ]

    if args.skip_data:
        print("\n[INFO] Skipping data generation/loading (--skip-data)")
        steps = steps[1:]

    for desc, script in steps:
        run_step(desc, script)

    print_final_summary()


if __name__ == "__main__":
    main()
