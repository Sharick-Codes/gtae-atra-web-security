"""
src/graph_builder.py
Converts processed flow records into a PyTorch-geometric-style graph
representation for the GTAE encoder.

Key design:
  - Nodes  = unique "IP:Port" style group identifiers (from data_loader)
  - Edges  = individual network flows, with feature vectors as edge_attr
  - LPE    = Laplacian Positional Encoding computed on BENIGN-ONLY subgraph
              to prevent attack topology from leaking into training features.
              Attack-only nodes receive zero-padded PEs automatically.
"""

import os
import pickle
import sys
import numpy as np
import torch
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_CONFIG, GRAPH_CONFIG
from utils import get_logger

logger = get_logger(__name__)


# ============================================================
# NODE INDEX
# ============================================================

def build_node_index(records: list) -> dict:
    """
    Build a string -> integer mapping for all unique node identifiers.

    Args:
        records: List of flow-record dicts with 'src_node' and 'dst_node'

    Returns:
        node_to_idx: dict mapping node string -> integer index
    """
    nodes = set()
    for r in records:
        nodes.add(r["src_node"])
        nodes.add(r["dst_node"])
    node_to_idx = {node: idx for idx, node in enumerate(sorted(nodes))}
    logger.info(f"Node index: {len(node_to_idx):,} unique nodes")
    return node_to_idx


# ============================================================
# EDGE TENSORS
# ============================================================

def build_edge_tensors(records: list, node_to_idx: dict) -> tuple:
    """
    Build PyTorch tensors from the flow records.

    Returns:
        edge_index   : LongTensor  [2, E]   -- (src_idx, dst_idx) per edge
        edge_attr    : FloatTensor [E, F]   -- feature vector per edge
        edge_labels  : LongTensor  [E]      -- 0=benign, 1=attack
        attack_types : list[str]   [E]      -- attack type label per edge
    """
    src_list, dst_list = [], []
    feat_list, label_list, atk_list = [], [], []

    for r in records:
        src_list.append(node_to_idx[r["src_node"]])
        dst_list.append(node_to_idx[r["dst_node"]])
        feat_list.append(r["edge_features"])
        label_list.append(1 if r["label"] == "attack" else 0)
        atk_list.append(r["attack_type"])

    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr   = torch.tensor(np.stack(feat_list), dtype=torch.float32)
    edge_labels = torch.tensor(label_list, dtype=torch.long)

    return edge_index, edge_attr, edge_labels, atk_list


# ============================================================
# LAPLACIAN POSITIONAL ENCODING
# ============================================================

def compute_laplacian_pe(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int,
) -> torch.Tensor:
    """
    Compute k-dimensional Laplacian Positional Encodings via eigen-
    decomposition of the normalized graph Laplacian.

    CRITICAL: Always called with benign-only edge_index to prevent
    topological leakage. Nodes absent from this subgraph receive
    zero-padded PEs.

    Args:
        edge_index: LongTensor [2, E_benign] -- benign edges only
        num_nodes:  Total node count (including attack-only nodes)
        k:          Number of eigenvectors to use

    Returns:
        node_pe: FloatTensor [num_nodes, k]
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from([(int(u), int(v)) for u, v in edges])

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    L = nx.normalized_laplacian_matrix(G).toarray().astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(L)   # Sorted ascending

    # Skip the trivial eigenvector (eigenvalue ? 0), take next k
    pe = eigvecs[:, 1 : k + 1]

    # Zero-pad if graph has fewer than k non-trivial eigenvectors
    if pe.shape[1] < k:
        pad = np.zeros((num_nodes, k - pe.shape[1]), dtype=np.float64)
        pe = np.concatenate([pe, pad], axis=1)

    return torch.tensor(pe, dtype=torch.float32)


# ============================================================
# ATTACH PE TO EDGES
# ============================================================

def attach_pe_to_edges(
    edge_index: torch.Tensor,
    node_pe: torch.Tensor,
) -> torch.Tensor:
    """
    Map node-level PEs onto edges by concatenating src and dst PEs.

    Args:
        edge_index: LongTensor [2, E]
        node_pe:    FloatTensor [num_nodes, k]

    Returns:
        edge_pe: FloatTensor [E, 2k]
    """
    src_pe = node_pe[edge_index[0]]   # [E, k]
    dst_pe = node_pe[edge_index[1]]   # [E, k]
    return torch.cat([src_pe, dst_pe], dim=1)   # [E, 2k]


# ============================================================
# MAIN BUILD FUNCTION
# ============================================================

def build_graph(records: list) -> dict:
    """
    Full graph construction from flow records.

    Args:
        records: Combined list of benign + attack flow-record dicts

    Returns:
        graph_data dict with keys:
            node_to_idx, num_nodes, edge_index, edge_attr,
            edge_pe, edge_labels, attack_types
    """
    node_to_idx = build_node_index(records)
    num_nodes   = len(node_to_idx)

    edge_index, edge_attr, edge_labels, attack_types = build_edge_tensors(
        records, node_to_idx
    )
    logger.info(f"edge_index : {tuple(edge_index.shape)}")
    logger.info(f"edge_attr  : {tuple(edge_attr.shape)}")

    k = GRAPH_CONFIG["pos_enc_dim"]

    # Compute LPE on BENIGN-ONLY subgraph (no topological leakage)
    benign_mask       = edge_labels == 0
    benign_edge_index = edge_index[:, benign_mask]
    logger.info(f"Computing LPE on {int(benign_mask.sum()):,} benign edges "
                f"(k={k}, num_nodes={num_nodes:,})")
    node_pe  = compute_laplacian_pe(benign_edge_index, num_nodes, k)
    edge_pe  = attach_pe_to_edges(edge_index, node_pe)

    logger.info(f"edge_pe    : {tuple(edge_pe.shape)}")

    benign_count = int((edge_labels == 0).sum())
    attack_count = int((edge_labels == 1).sum())
    print(f"\nGraph built: {num_nodes:,} nodes | "
          f"{benign_count:,} benign edges | {attack_count:,} attack edges")

    return {
        "node_to_idx": node_to_idx,
        "num_nodes":   num_nodes,
        "edge_index":  edge_index,
        "edge_attr":   edge_attr,
        "edge_pe":     edge_pe,
        "edge_labels": edge_labels,
        "attack_types": attack_types,
    }


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    """Load processed records, build graph, save graph_data.pkl."""
    benign_path = DATA_CONFIG["benign_train_path"]
    attack_path = DATA_CONFIG["attack_test_path"]

    if not Path(benign_path).exists():
        logger.error(
            f"Processed data not found: {benign_path}\n"
            "Run 'python src/data_loader.py' first."
        )
        return

    records = []
    with open(benign_path, "rb") as f:
        records.extend(pickle.load(f))
    with open(attack_path, "rb") as f:
        records.extend(pickle.load(f))

    logger.info(f"Loaded {len(records):,} total records "
                f"({sum(1 for r in records if r['label']=='benign'):,} benign, "
                f"{sum(1 for r in records if r['label']=='attack'):,} attack)")

    graph_data = build_graph(records)

    out_path = DATA_CONFIG["graph_data_path"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(graph_data, f)

    print(f"[OK] Saved graph_data -> {out_path}")


if __name__ == "__main__":
    main()

