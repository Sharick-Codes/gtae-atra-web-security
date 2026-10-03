from scapy.all import IP, TCP
import pytest

from src import realtime_monitor


def make_packet(src, dst, sport, dport, flags, timestamp):
    packet = IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags)
    packet.time = timestamp
    return packet


def test_flow_collector_groups_reverse_direction():
    collector = realtime_monitor.FlowCollector()
    collector.handle_packet(make_packet("10.0.0.1", "10.0.0.2", 1234, 80, "S", 1.0))
    collector.handle_packet(make_packet("10.0.0.2", "10.0.0.1", 80, 1234, "SA", 1.2))

    flows = collector.get_completed_flows()

    assert len(flows) == 1
    assert flows[0]["forward_packet_count"] == 1
    assert flows[0]["backward_packet_count"] == 1
    assert flows[0]["duration"] == pytest.approx(0.2)


def test_live_feature_contract_preserves_directional_slots():
    collector = realtime_monitor.FlowCollector()
    collector.handle_packet(make_packet("10.0.0.1", "10.0.0.2", 1234, 80, "S", 1.0))
    collector.handle_packet(make_packet("10.0.0.2", "10.0.0.1", 80, 1234, "SA", 1.2))

    vector = realtime_monitor.encode_flow_features(collector.get_completed_flows()[0])
    backward_start = realtime_monitor.FEATURES_PER_PACKET * realtime_monitor.PACKETS_PER_DIRECTION

    assert vector.shape == (realtime_monitor.LIVE_FEATURE_DIM,)
    assert vector.dtype.name == "float32"
    assert vector[0] > 0
    assert vector[backward_start] > 0


def test_capture_window_forwards_interface(monkeypatch):
    captured = {}

    def fake_sniff(**kwargs):
        captured.update(kwargs)
        kwargs["prn"](make_packet("10.0.0.1", "10.0.0.2", 1234, 80, "S", 1.0))
        kwargs["prn"](make_packet("10.0.0.2", "10.0.0.1", 80, 1234, "SA", 1.2))

    monkeypatch.setattr(realtime_monitor, "sniff", fake_sniff)
    flows = realtime_monitor.capture_window(1, iface="test-interface")

    assert captured["iface"] == "test-interface"
    assert len(flows) == 1
