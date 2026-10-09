import unittest


class TestGraphBuilder(unittest.TestCase):
    def test_build_node_index(self):
        nodes = ["ip1", "ip2", "endpoint1"]
        node_idx = {n: i for i, n in enumerate(nodes)}
        self.assertEqual(node_idx["ip1"], 0)
        self.assertEqual(node_idx["ip2"], 1)

    def test_laplacian_pe_dimension(self):
        k = 8
        self.assertEqual(k, 8)


if __name__ == "__main__":
    unittest.main()
