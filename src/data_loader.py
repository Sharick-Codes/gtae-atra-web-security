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

