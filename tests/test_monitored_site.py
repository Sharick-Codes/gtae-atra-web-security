from datetime import datetime, timedelta

from monitored_site import WebRequestMonitor


def test_normal_request_is_benign_and_logged(tmp_path):
    log_path = tmp_path / "web-events.csv"
    monitor = WebRequestMonitor(log_path)

    result = monitor.inspect("192.168.31.164", "GET", "/about", 200)

    assert result["verdict"] == "Benign"
    assert result["response_decision"] == "No Action (Benign)"
    assert result["enforcement"] == "decision-only; no traffic blocked"
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 2


def test_sensitive_path_probe_gets_atra_assessment(tmp_path):
    monitor = WebRequestMonitor(tmp_path / "web-events.csv")

    result = monitor.inspect("192.168.31.164", "GET", "/.env", 404)

    assert result["verdict"] == "Suspicious"
    assert result["signal"] == "Sensitive path probe"
    assert result["risk_level"] in {"Low", "Medium", "High", "Critical"}
    assert result["response_decision"] != "No Action (Benign)"


def test_repeated_404_requests_are_flagged(tmp_path):
    monitor = WebRequestMonitor(tmp_path / "web-events.csv")
    start = datetime(2026, 9, 28, 12, 0, 0)

    for index in range(monitor.NOT_FOUND_THRESHOLD - 1):
        result = monitor.inspect(
            "192.168.31.164", "GET", f"/missing-{index}", 404,
            now=start + timedelta(seconds=index),
        )
        assert result["verdict"] == "Benign"

    result = monitor.inspect(
        "192.168.31.164", "GET", "/missing-final", 404,
        now=start + timedelta(seconds=monitor.NOT_FOUND_THRESHOLD),
    )

    assert result["verdict"] == "Suspicious"
    assert result["signal"] == "Repeated not-found requests"
