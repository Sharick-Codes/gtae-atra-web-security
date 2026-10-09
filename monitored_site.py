"""Small authorized web target with server-side request monitoring."""

import argparse
import csv
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request

from src.atra import (
    ATTACK_SEVERITY,
    ATTACK_TYPE_WEIGHT,
    IPHistoryTracker,
    RESPONSE_ACTIONS,
    compute_risk_score,
    get_risk_level,
)
from src.config import LOGS_DIR

app = Flask(__name__)
MONITOR = None


class WebRequestMonitor:
    """Record requests and score simple, explainable web-abuse signals."""

    WINDOW_SECONDS = 10
    BURST_THRESHOLD = 25
    NOT_FOUND_THRESHOLD = 5
    PROBE_MARKERS = ("/.env", "/.git", "/wp-admin", "/phpmyadmin", "../", "%2e%2e")
    ALLOWED_METHODS = {"GET", "HEAD", "POST", "OPTIONS"}

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.events_by_ip = defaultdict(deque)
        self.history = IPHistoryTracker(half_life_seconds=300)
        self.blocked_ips = set()
        self.lock = threading.Lock()
        self._ensure_log()

    def _ensure_log(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path.exists() and self.log_path.stat().st_size:
            return
        with self.log_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                "timestamp", "client_ip", "method", "path", "status_code",
                "verdict", "signal", "requests_in_window", "risk_score",
                "risk_level", "response_decision", "enforcement",
            ])

    def inspect(self, client_ip: str, method: str, path: str, status_code: int, now=None) -> dict:
        """Classify one server request and calculate its ATRA response decision."""
        now = now or datetime.now()
        now_seconds = time.time() if now is None else now.timestamp()
        normalized_path = path.lower()

        with self.lock:
            events = self.events_by_ip[client_ip]
            events.append((now_seconds, method.upper(), normalized_path, status_code))
            cutoff = now_seconds - self.WINDOW_SECONDS
            while events and events[0][0] < cutoff:
                events.popleft()

            recent_count = len(events)
            recent_404s = sum(event[3] == 404 for event in events)
            signal = None
            confidence = 0.0
            attack_type = "Unknown"

            if any(marker in normalized_path for marker in self.PROBE_MARKERS):
                signal = "Sensitive path probe"
                attack_type = "Web Attack"
                confidence = 8.0
            elif method.upper() not in self.ALLOWED_METHODS:
                signal = "Unusual HTTP method"
                attack_type = "Web Attack"
                confidence = 7.0
            elif recent_404s >= self.NOT_FOUND_THRESHOLD:
                signal = "Repeated not-found requests"
                attack_type = "PortScan"
                confidence = 6.0
            elif recent_count >= self.BURST_THRESHOLD:
                signal = "HTTP request burst"
                attack_type = "DoS Hulk"
                confidence = 6.0

            suspicious = signal is not None
            risk_score = 0.0
            risk_level = "None"
            decision = "No Action (Benign)"
            if suspicious:
                history_score = min(self.history.get_decayed_history_score(client_ip, now) * 2.0, 10.0)
                frequency_score = min(self.history.get_frequency_last_window(client_ip, now) * 2.5, 10.0)
                risk_score = compute_risk_score(
                    ATTACK_TYPE_WEIGHT.get(attack_type, ATTACK_TYPE_WEIGHT["Unknown"]),
                    ATTACK_SEVERITY.get(attack_type, ATTACK_SEVERITY["Unknown"]),
                    confidence,
                    history_score,
                    frequency_score,
                )
                risk_level = get_risk_level(risk_score)
                decision = RESPONSE_ACTIONS.get(risk_level, "Generate Log")
                self.history.record_attack(client_ip, now)

            enforcement = "decision-only; no traffic blocked"
            if decision in ("Block (Drop/Blacklist)", "BLOCK") or risk_level == "Critical":
                self.blocked_ips.add(client_ip)
                enforcement = "active; traffic blocked (403 Forbidden)"

            result = {
                "timestamp": now.isoformat(timespec="milliseconds"),
                "client_ip": client_ip,
                "method": method.upper(),
                "path": path,
                "status_code": status_code,
                "verdict": "Suspicious" if suspicious else "Benign",
                "signal": signal or "",
                "requests_in_window": recent_count,
                "risk_score": round(risk_score, 2),
                "risk_level": risk_level,
                "response_decision": decision,
                "enforcement": enforcement,
            }
            with self.log_path.open("a", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=result.keys()).writerow(result)
            if suspicious:
                print(
                    f"[WEB IDS] {client_ip} {method.upper()} {path} | {signal} | "
                    f"{risk_level} ({risk_score:.1f}) | ATRA: {decision} | Enforcement: {enforcement}"
                )
            return result

    def is_blocked(self, client_ip: str) -> bool:
        return client_ip in getattr(self, "blocked_ips", set())


def _render_blocked_page(ip: str, path: str):
    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>403 Forbidden - GTAE-ATRA Active Defense</title>
<style>
:root {{ --bg: #0a0e17; --card: #131c2e; --border: #f43f5e; --text: #f1f5f9; --sub: #94a3b8; }}
[data-theme="light"] {{ --bg: #fff1f2; --card: #ffffff; --border: #e11d48; --text: #0f172a; --sub: #64748b; }}
body {{ font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; align-items: center; justify-content: center; margin: 0; padding: 20px; }}
.card {{ background: var(--card); border: 2px solid var(--border); border-radius: 16px; padding: 32px; max-width: 540px; text-align: center; box-shadow: 0 10px 40px rgba(244,63,94,0.3); }}
.shield {{ font-size: 54px; margin-bottom: 12px; }}
h1 {{ margin: 0 0 12px; font-size: 24px; color: #f43f5e; }}
p {{ color: var(--sub); line-height: 1.6; margin-bottom: 20px; }}
.info {{ background: rgba(244,63,94,0.08); border-radius: 8px; padding: 12px; font-family: monospace; font-size: 14px; text-align: left; margin-bottom: 20px; }}
.btn {{ display: inline-block; background: #f43f5e; color: #fff; text-decoration: none; padding: 10px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; border: none; }}
.toggle {{ position: absolute; top: 20px; right: 20px; background: var(--card); border: 1px solid var(--border); color: var(--text); padding: 6px 14px; border-radius: 20px; cursor: pointer; }}
</style>
</head>
<body>
<button class="toggle" onclick="t=document.documentElement;t.dataset.theme=t.dataset.theme==='light'?'dark':'light'">🌓 Theme</button>
<div class="card">
<div class="shield">🛡️</div>
<h1>Access Denied (HTTP 403)</h1>
<p>Your requests triggered the <strong>GTAE-ATRA Autonomous Intrusion Defense System</strong>. Malicious activity was flagged and your IP address is actively blocked.</p>
<div class="info">
<div><strong>Blocked IP:</strong> {ip}</div>
<div><strong>Target URL:</strong> {path}</div>
<div><strong>Action:</strong> BLOCK (Drop/Blacklist)</div>
<div><strong>Engine:</strong> Graph Transformer Autoencoder & ATRA</div>
</div>
<button class="btn" onclick="location.reload()">Retry Connection</button>
</div>
</body></html>"""


@app.before_request
def ensure_monitor():
    """Initialize monitor lazily and enforce blocking."""
    global MONITOR
    if MONITOR is None:
        MONITOR = WebRequestMonitor(Path(LOGS_DIR) / "web_request_events.csv")
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    if request.path != "/health" and MONITOR.is_blocked(client_ip):
        return _render_blocked_page(client_ip, request.path), 403


@app.after_request
def monitor_request(response):
    """Observe requests reaching this owned demo site, not dashboard APIs."""
    if request.path != "/health":
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
        result = MONITOR.inspect(
            client_ip,
            request.method,
            request.path,
            response.status_code,
        )
        response.headers["X-IDS-Verdict"] = result["verdict"]
        response.headers["X-ATRA-Risk"] = str(result["risk_level"])
    return response


@app.get("/")
def home():
    return """<!doctype html>
<html lang="en"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IDS Monitored Test Site</title>
<style>body{font:17px system-ui;max-width:760px;margin:12vh auto;padding:24px;color:#18212b}
a{color:#087e8b}main{border-left:4px solid #087e8b;padding:4px 24px}</style>
<main><h1>Monitored Test Website</h1>
<p>This site is hosted on the IDS laptop. Requests to it are logged and scored.</p>
<p><a href="/about">Open the about page</a></p>
<p><a href="/health">Check site health</a></p></main></html>"""


@app.get("/about")
def about():
    return "<h1>About</h1><p>Normal test-site page.</p><a href='/'>Home</a>"


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "not found"}), 404


def main():
    parser = argparse.ArgumentParser(description="Run the IDS-monitored demo website.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: all interfaces)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    global MONITOR
    MONITOR = WebRequestMonitor(Path(LOGS_DIR) / "web_request_events.csv")
    print(f"Monitored test site listening at http://{args.host}:{args.port}")
    print(f"Request events: {MONITOR.log_path}")
    print("ATRA is decision-only in this demo; it does not block requests.")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
