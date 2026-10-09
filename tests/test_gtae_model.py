import unittest
import torch
from src.gtae_model import GTAE


class TestGTAEModel(unittest.TestCase):
    def test_gtae_forward(self):
        in_features = 20
        d_model = 16
        pe_dim = 8
        model = GTAE(in_features=in_features, pe_dim=pe_dim, d_model=d_model, num_heads=2, num_layers=1)
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, in_features)
        pe = torch.zeros(batch_size, pe_dim)
        dst = torch.zeros(batch_size, dtype=torch.long)

        with torch.no_grad():
            reconstructed, embedding = model(x, pe, dst)

        self.assertEqual(reconstructed.shape, (batch_size, in_features))
        self.assertEqual(embedding.shape, (batch_size, d_model))


if __name__ == "__main__":
    unittest.main()
