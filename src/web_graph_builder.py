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
