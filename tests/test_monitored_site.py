import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from monitored_site import WebRequestMonitor


class TestMonitoredSite(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp_dir.name) / "web-events.csv"
        self.monitor = WebRequestMonitor(self.log_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normal_request_is_benign_and_logged(self):
        result = self.monitor.inspect("192.168.31.164", "GET", "/about", 200)

        self.assertEqual(result["verdict"], "Benign")
        self.assertEqual(result["response_decision"], "No Action (Benign)")
        self.assertEqual(result["enforcement"], "decision-only; no traffic blocked")
        self.assertEqual(len(self.log_path.read_text(encoding="utf-8").splitlines()), 2)

    def test_sensitive_path_probe_gets_atra_assessment(self):
        result = self.monitor.inspect("192.168.31.164", "GET", "/.env", 404)

        self.assertEqual(result["verdict"], "Suspicious")
        self.assertEqual(result["signal"], "Sensitive path probe")
        self.assertIn(result["risk_level"], {"Low", "Medium", "High", "Critical"})
        self.assertNotEqual(result["response_decision"], "No Action (Benign)")

    def test_repeated_404_requests_are_flagged(self):
        start = datetime(2026, 9, 28, 12, 0, 0)

        for index in range(self.monitor.NOT_FOUND_THRESHOLD - 1):
            result = self.monitor.inspect(
                "192.168.31.164", "GET", f"/missing-{index}", 404,
                now=start + timedelta(seconds=index),
            )
            self.assertEqual(result["verdict"], "Benign")

        result = self.monitor.inspect(
            "192.168.31.164", "GET", "/missing-final", 404,
            now=start + timedelta(seconds=self.monitor.NOT_FOUND_THRESHOLD),
        )

        self.assertEqual(result["verdict"], "Suspicious")
        self.assertEqual(result["signal"], "Repeated not-found requests")

    def test_active_blocking_enforcement(self):
        # Repeated attacks escalate to Block
        now = datetime(2026, 9, 28, 12, 0, 0)
        for i in range(5):
            res = self.monitor.inspect("10.0.0.99", "GET", "/.env", 404, now=now + timedelta(seconds=i*2))
        
        # After escalation, IP is tracked in blocked_ips
        self.assertTrue(self.monitor.is_blocked("10.0.0.99"))


if __name__ == "__main__":
    unittest.main()
