import unittest
from datetime import datetime, timedelta

from src.atra import (
    IPHistoryTracker,
    compute_risk_score,
    get_risk_level,
    normalize,
    ATTACK_TYPE_WEIGHT,
    ATTACK_SEVERITY,
    RESPONSE_ACTIONS,
)


class TestATRA(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize(5.0, 0.0, 10.0, 10.0), 5.0)
        self.assertEqual(normalize(0.0, 0.0, 10.0, 10.0), 0.0)
        self.assertEqual(normalize(10.0, 0.0, 10.0, 10.0), 10.0)
        # Clamping
        self.assertEqual(normalize(15.0, 0.0, 10.0, 10.0), 10.0)
        self.assertEqual(normalize(-5.0, 0.0, 10.0, 10.0), 0.0)

    def test_compute_risk_score(self):
        # All zeros -> 0
        score_zero = compute_risk_score(0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(score_zero, 0.0)

        # All max (10.0) -> 100.0
        score_max = compute_risk_score(10.0, 10.0, 10.0, 10.0, 10.0)
        self.assertAlmostEqual(score_max, 100.0, places=2)

    def test_get_risk_level(self):
        self.assertEqual(get_risk_level(15.0), "Low")
        self.assertEqual(get_risk_level(35.0), "Medium")
        self.assertEqual(get_risk_level(55.0), "High")
        self.assertEqual(get_risk_level(75.0), "Critical")

    def test_ip_history_tracker_record_and_decay(self):
        tracker = IPHistoryTracker(half_life_seconds=300)
        now = datetime(2026, 10, 9, 12, 0, 0)
        tracker.record_attack("198.51.100.10", now)

        # Immediate check (score = 1.0)
        score_now = tracker.get_decayed_history_score("198.51.100.10", now)
        self.assertAlmostEqual(score_now, 1.0, places=2)

        # Half-life check at +300 seconds (score should be 0.5)
        after_half_life = now + timedelta(seconds=300)
        score_decayed = tracker.get_decayed_history_score("198.51.100.10", after_half_life)
        self.assertAlmostEqual(score_decayed, 0.5, places=2)

    def test_ip_history_frequency_window(self):
        tracker = IPHistoryTracker(half_life_seconds=300)
        now = datetime(2026, 10, 9, 12, 0, 0)
        tracker.record_attack("198.51.100.20", now - timedelta(seconds=100))
        tracker.record_attack("198.51.100.20", now - timedelta(seconds=50))
        tracker.record_attack("198.51.100.20", now)
        # Older attack beyond window
        tracker.record_attack("198.51.100.20", now - timedelta(seconds=700))

        freq = tracker.get_frequency_last_window("198.51.100.20", now, window_seconds=600)
        self.assertEqual(freq, 3)


if __name__ == "__main__":
    unittest.main()
