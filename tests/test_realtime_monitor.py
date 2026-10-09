import unittest

try:
    from scapy.all import IP, TCP
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

if HAS_SCAPY:
    from src import realtime_monitor


def make_packet(src, dst, sport, dport, flags, timestamp):
    packet = IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags)
    packet.time = timestamp
    return packet


@unittest.skipUnless(HAS_SCAPY, "scapy library not available")
class TestRealtimeMonitor(unittest.TestCase):
    def test_flow_collector_groups_reverse_direction(self):
        collector = realtime_monitor.FlowCollector()
        collector.handle_packet(make_packet("10.0.0.1", "10.0.0.2", 1234, 80, "S", 1.0))
        collector.handle_packet(make_packet("10.0.0.2", "10.0.0.1", 80, 1234, "SA", 1.2))

        flows = collector.get_completed_flows()

        self.assertEqual(len(flows), 1)
        self.assertEqual(flows[0]["forward_packet_count"], 1)
        self.assertEqual(flows[0]["backward_packet_count"], 1)
        self.assertAlmostEqual(flows[0]["duration"], 0.2, places=2)

    def test_live_feature_contract_preserves_directional_slots(self):
        collector = realtime_monitor.FlowCollector()
        collector.handle_packet(make_packet("10.0.0.1", "10.0.0.2", 1234, 80, "S", 1.0))
        collector.handle_packet(make_packet("10.0.0.2", "10.0.0.1", 80, 1234, "SA", 1.2))

        vector = realtime_monitor.encode_flow_features(collector.get_completed_flows()[0])
        backward_start = realtime_monitor.FEATURES_PER_PACKET * realtime_monitor.PACKETS_PER_DIRECTION

        self.assertEqual(vector.shape, (realtime_monitor.LIVE_FEATURE_DIM,))
        self.assertEqual(vector.dtype.name, "float32")
        self.assertGreater(vector[0], 0)
        self.assertGreater(vector[backward_start], 0)


if __name__ == "__main__":
    unittest.main()
