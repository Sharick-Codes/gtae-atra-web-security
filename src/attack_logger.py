
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
