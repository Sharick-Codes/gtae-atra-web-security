
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import pickle
import os

from config import DATA_CONFIG, GTAE_CONFIG, MODELS_DIR
from utils import get_logger

logger = get_logger(__name__)

class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer Layer with attention grouped by shared destination node.
    """
    def __init__(self, d_model: int, num_heads: int, ffn_hidden_mult: int = 2):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_hidden_mult),
            nn.ReLU(),
            nn.Linear(d_model * ffn_hidden_mult, d_model),
        )

    def forward(self, e: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
        """
        e: [E, d_model] edge features
        dst_idx: [E] destination node index for each edge, used to group
                 edges that share a destination node for attention (Eq. 4-5)
        """
        E = e.size(0)
        Q = self.q_proj(e).view(E, self.num_heads, self.d_k)
        K = self.k_proj(e).view(E, self.num_heads, self.d_k)
        V = self.v_proj(e).view(E, self.num_heads, self.d_k)

        # Group edge indices by shared destination node
        groups = defaultdict(list)
        for edge_i, d in enumerate(dst_idx.tolist()):
            groups[d].append(edge_i)

        attn_out = torch.zeros(E, self.num_heads, self.d_k, device=e.device)

        for _, edge_ids in groups.items():
            idx = torch.tensor(edge_ids, dtype=torch.long, device=e.device)
            q_group = Q[idx]              # [n, heads, d_k]
            k_group = K[idx]              # [n, heads, d_k]
            v_group = V[idx]              # [n, heads, d_k]

            # scores: [n_query, n_key, heads]
            scores = torch.einsum("qhd,khd->qkh", q_group, k_group) / (self.d_k ** 0.5)
            attn_weights = F.softmax(scores, dim=1)  # softmax over "key" edges in the group

            # weighted sum of values -> [n_query, heads, d_k]
            out = torch.einsum("qkh,khd->qhd", attn_weights, v_group)
            attn_out[idx] = out

        attn_out = attn_out.reshape(E, self.d_model)
        attn_out = self.out_proj(attn_out)

        # Residual + norm (Eq. 6)
        e = self.norm1(e + attn_out)

        # FFN + residual + norm (Eq. 7-8)
        ffn_out = self.ffn(e)
        e = self.norm2(e + ffn_out)

        return e


class GraphTransformerEncoder(nn.Module):
    """
    Encoder stack of Graph Transformer layers.
    """
    def __init__(self, in_features: int, pe_dim: int, d_model: int = 64,
                 num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)   # Eq. 2
        self.pe_proj = nn.Linear(pe_dim, d_model)            # Eq. 3

        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_model, num_heads) for _ in range(num_layers)
        ])

    def forward(self, edge_attr: torch.Tensor, edge_pe: torch.Tensor,
                dst_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass to obtain bottleneck embeddings z."""
        e0 = self.input_proj(edge_attr)      # Eq. 2
        pe0 = self.pe_proj(edge_pe)          # Eq. 3
        e = e0 + pe0                         # combined input (only at input layer)

        for layer in self.layers:
            e = layer(e, dst_idx)

        return e  # [E, d_model] -> this is the bottleneck z


class DNNDecoder(nn.Module):
    """
    3-layer MLP decoder taking hidden dims from GTAE_CONFIG.
    """
    def __init__(self, d_model: int, out_features: int):
        super().__init__()
        hidden_dims = GTAE_CONFIG.get("decoder_hidden", [128, 256])
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], out_features),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the bottleneck z back to edge features."""
        return self.net(z)


class GTAE(nn.Module):
    """
    Graph Transformer Autoencoder.
    """
    def __init__(self, in_features: int, pe_dim: int, d_model: int = 64,
                 num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.encoder = GraphTransformerEncoder(
            in_features=in_features, pe_dim=pe_dim,
            d_model=d_model, num_heads=num_heads, num_layers=num_layers,
        )
        self.decoder = DNNDecoder(d_model=d_model, out_features=in_features)

    def forward(self, edge_attr: torch.Tensor, edge_pe: torch.Tensor, dst_idx: torch.Tensor):
        """Forward pass to reconstruct edge attributes."""
        z = self.encoder(edge_attr, edge_pe, dst_idx)   # bottleneck
        x_hat = self.decoder(z)                          # reconstruction
        return x_hat, z


def train_gtae(model, edge_attr, edge_pe, dst_idx, edge_labels):
    """
    Train GTAE on BENIGN-ONLY flows (edge_labels == 0).
    """
    epochs = GTAE_CONFIG.get("epochs", 100)
    lr = GTAE_CONFIG.get("lr", 0.001)
    patience = GTAE_CONFIG.get("early_stopping_patience", 10)
    checkpoint_every = GTAE_CONFIG.get("checkpoint_every", 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    benign_mask = (edge_labels == 0)
    edge_attr_benign = edge_attr[benign_mask]
    edge_pe_benign = edge_pe[benign_mask]
    dst_idx_benign = dst_idx[benign_mask]

    logger.info(f"Training on {edge_attr_benign.shape[0]} benign flows only.")

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        x_hat, _ = model(edge_attr_benign, edge_pe_benign, dst_idx_benign)
        loss = loss_fn(x_hat, edge_attr_benign)
        loss.backward()
        optimizer.step()

        val_loss = loss.item() # use benign as val for unsupervised early stopping 

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{epochs} | Reconstruction MSE Loss: {val_loss:.6f}")
        
        if epoch % checkpoint_every == 0:
            ckpt_path = os.path.join(MODELS_DIR, f'gtae_checkpoint_epoch{epoch}.pt')
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def compute_reconstruction_errors(model, edge_attr, edge_pe, dst_idx):
    """
    Run in eval mode with torch.no_grad() and compute errors.
    """
    model.eval()
    with torch.no_grad():
        x_hat, z = model(edge_attr, edge_pe, dst_idx)
        errors = torch.mean((x_hat - edge_attr) ** 2, dim=1)  # [E]
    return errors, z


def load_gtae(path, in_features, pe_dim):
    """
    Load saved model weights from path.
    """
    d_model = GTAE_CONFIG.get("d_model", 64)
    num_heads = GTAE_CONFIG.get("num_heads", 4)
    num_layers = GTAE_CONFIG.get("num_layers", 3)
    model = GTAE(in_features=in_features, pe_dim=pe_dim, d_model=d_model, num_heads=num_heads, num_layers=num_layers)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def main():
    """Main execution function for GTAE training and evaluation."""
    data_path = DATA_CONFIG["graph_data_path"]
    with open(data_path, "rb") as f:
        graph_data = pickle.load(f)

    edge_attr = graph_data["edge_attr"]
    edge_pe = graph_data["edge_pe"]
    dst_idx = graph_data["edge_index"][1]
    edge_labels = graph_data["edge_labels"]
    attack_types = graph_data["attack_types"]
    
    in_features = edge_attr.shape[1]
    pe_dim = edge_pe.shape[1]
    
    d_model = GTAE_CONFIG.get("d_model", 64)
    num_heads = GTAE_CONFIG.get("num_heads", 4)
    num_layers = GTAE_CONFIG.get("num_layers", 3)

    model = GTAE(in_features=in_features, pe_dim=pe_dim, d_model=d_model, num_heads=num_heads, num_layers=num_layers)
    
    # Train
    model = train_gtae(model, edge_attr, edge_pe, dst_idx, edge_labels)
    
    # Evaluate all flows
    errors, embeddings = compute_reconstruction_errors(model, edge_attr, edge_pe, dst_idx)
    
    benign_mask = (edge_labels == 0)
    attack_mask = (edge_labels == 1)
    
    benign_errors = errors[benign_mask]
    attack_errors = errors[attack_mask]
    
    logger.info("--- Reconstruction Error Summary ---")
    logger.info(f"Benign flows  -> mean error: {benign_errors.mean():.6f} | std: {benign_errors.std():.6f}")
    logger.info(f"Attack flows  -> mean error: {attack_errors.mean():.6f} | std: {attack_errors.std():.6f}")
    
    model_path = GTAE_CONFIG["model_path"]
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved trained model to {model_path}")
    
    results = {
        "reconstruction_errors": errors,
        "embeddings": embeddings,
        "edge_labels": edge_labels,
        "attack_types": attack_types,
    }
    
    results_path = DATA_CONFIG["gtae_results_path"]
    os.makedirs(os.path.dirname(results_path) or '.', exist_ok=True)
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved results to {results_path}")

if __name__ == "__main__":
    main()
