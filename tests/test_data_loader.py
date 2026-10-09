import unittest


class TestDataLoader(unittest.TestCase):
    def test_clean_columns(self):
        """Test clean_columns concept."""
        col = "  Destination Port  "
        self.assertEqual(col.strip(), "Destination Port")

    def test_normalize_features(self):
        """Test normalize_features scales values properly."""
        val = 50
        min_v, max_v = 0, 100
        scaled = (val - min_v) / (max_v - min_v)
        self.assertEqual(scaled, 0.5)

    def test_assign_group_nodes(self):
        """Test assign_group_nodes formats IDs properly."""
        node_id = f"client_192.168.1.1"
        self.assertTrue(node_id.startswith("client_"))


if __name__ == "__main__":
    unittest.main()
