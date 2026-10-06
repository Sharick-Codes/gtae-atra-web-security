"""
src/web_monitor_server.py
Real-time web security monitoring server for GTAE-ATRA.

Receives HTTP request telemetry from the Node.js webapp middleware,
runs GTAE + ensemble anomaly detection, applies ATRA risk scoring,
emits security events via Socket.IO to the React dashboard.

Architecture:
  Node.js webapp  -->  POST /telemetry  -->  This server
  This server     -->  Socket.IO        -->  React dashboard
  This server     -->  JSONL log        -->  logs/web_security_events.jsonl

Usage:
  python src/web_monitor_server.py

Prerequisites:
  python src/web_train.py   (must be run first to generate trained models)
"""

import sys
import os
import json
import time
import logging
import threading
from pathlib import Path
from collections import defaultdict
from datetime import datetime

try:
    import requests as _requests_lib
    _PROXY_AVAILABLE = True
except ImportError:
    _PROXY_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

import torch
import numpy as np
import joblib

from config import WEB_CONFIG, GTAE_CONFIG, ATRA_CONFIG, MODELS_DIR, LOGS_DIR
from gtae_model import GTAE
from anomaly_detector import HBOSDetector, predict_all
from atra import (
    ATTACK_SEVERITY, ATTACK_TYPE_WEIGHT, RESPONSE_ACTIONS,
    IPHistoryTracker, compute_risk_score, get_risk_level, normalize,
)
from web_feature_extractor import extract_features, build_reason, get_feature_dict
from utils import get_logger

logger = get_logger(__name__)

# ============================================================
# SERVER SETUP
# ============================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "gtae-atra-web-security-2026")
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ============================================================
# GLOBALS
# ============================================================

# Per-IP sliding window: IP -> list of request dicts
_ip_windows: dict = defaultdict(list)
_window_lock = threading.Lock()

# ATRA history tracker
_ip_tracker = IPHistoryTracker(half_life_seconds=ATRA_CONFIG["history_half_life_seconds"])

# Stats counters
_stats = {
    "total_requests": 0,
    "total_alerts":   0,
    "total_blocked":  0,
    "total_normal":   0,
    "requests_per_second": 0.0,
    "last_events":    [],         # ring buffer of 50 most recent events
    "blocked_ips":    set(),
    "anomaly_scores": [],         # last 200 anomaly score data points
    "request_times":  [],         # for req/s calculation
}
_stats_lock = threading.Lock()

# Global inference engine (loaded once on startup)
_engine = None
_engine_ready = False
_anomaly_threshold = None


# ============================================================
# INFERENCE ENGINE
# ============================================================

class WebInferenceEngine:
    """Loads trained GTAE + ensemble and runs inference per request window."""

    def __init__(self):
        schema_path = WEB_CONFIG["web_schema_path"]
        model_path  = WEB_CONFIG["web_model_path"]
        models_dir  = Path(MODELS_DIR)

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}\n"
                "Run 'python src/web_train.py' first."
            )

        # Load schema
        with open(schema_path) as f:
            schema = json.load(f)
        self.in_features = schema["in_features"]
        self.pe_dim      = schema["pe_dim"]

        # Load GTAE model
        self.model = GTAE(
            in_features=self.in_features,
            pe_dim=self.pe_dim,
            d_model=GTAE_CONFIG.get("d_model", 64),
            num_heads=GTAE_CONFIG.get("num_heads", 4),
            num_layers=GTAE_CONFIG.get("num_layers", 3),
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        logger.info(f"Loaded GTAE model from {model_path}")

        # Load scaler + detectors
        import __main__
        __main__.HBOSDetector = HBOSDetector
        self.scaler    = joblib.load(models_dir / "web_scaler.joblib")
        self.detectors = {
            "IF":    joblib.load(models_dir / "web_iso_forest.joblib"),
            "OCSVM": joblib.load(models_dir / "web_ocsvm.joblib"),
            "HBOS":  joblib.load(models_dir / "web_hbos.joblib"),
        }

        # Load attack classifier (optional)
        clf_path = models_dir / "web_attack_classifier.joblib"
        self.attack_classifier = joblib.load(clf_path) if clf_path.exists() else None

        # Load training results for threshold
        results_path = models_dir / "web_training_results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            self.anomaly_threshold = results.get("anomaly_threshold", 0.01)
        else:
            self.anomaly_threshold = 0.01

        logger.info(
            f"Inference engine ready | in_features={self.in_features} "
            f"| threshold={self.anomaly_threshold:.6f}"
        )

    def infer(self, source_ip: str, requests: list, site_id: str = "default") -> dict:
        """
        Run full GTAE + ATRA inference on one IP's request window.

        Returns a security event dict.
        """
        t_start = time.monotonic()

        # Feature extraction
        feats = extract_features(requests, window_seconds=WEB_CONFIG["window_seconds"])
        feat_tensor = torch.from_numpy(feats).reshape(1, -1)
        pe_tensor   = torch.zeros((1, self.pe_dim), dtype=torch.float32)
        dst_tensor  = torch.zeros(1, dtype=torch.long)

        # GTAE inference
        with torch.no_grad():
            reconstructed, embedding = self.model(feat_tensor, pe_tensor, dst_tensor)
            reconstruction_error = float(torch.mean((reconstructed - feat_tensor) ** 2).item())

        embedding_np = embedding.cpu().numpy()

        # Smooth 0-100 anomaly score (calibrated to avoid flat-lining at 200.0)
        ratio = reconstruction_error / max(self.anomaly_threshold, 1e-9)
        if ratio <= 1.0:
            anomaly_score = ratio * 50.0
        else:
            # Smoothly scales from 50.0 to 100.0 based on severity
            anomaly_score = 50.0 + 50.0 * float(np.tanh((ratio - 1.0) / 4.0))
        anomaly_score = round(float(np.clip(anomaly_score, 0.0, 100.0)), 1)

        # Ensemble anomaly detection
        scaled_emb   = self.scaler.transform(embedding_np)
        predictions  = predict_all(self.detectors, scaled_emb)

        # ── Concrete Threat Evidence Corroboration ────────────────────
        # Thresholds are intentionally conservative to avoid false-positives
        # from legitimate users interacting with AI (large POSTs, streaming, etc.)
        feat_dict = get_feature_dict(feats)

        # SQLi/XSS must be explicit – any match fires
        has_sqli_evidence      = (feat_dict.get("suspicious_pattern_count", 0.0) > 0.0)
        # Brute force: only explicit failed logins count
        has_brute_evidence     = (feat_dict.get("failed_login_count", 0.0) > 0.0)
        # Rate abuse: very high rate (>0.50 normalized = 150 req/min) OR extreme burst (>0.40 = 40 req/5s)
        has_rate_evidence      = (
            feat_dict.get("request_rate", 0.0) > 0.50 or
            feat_dict.get("max_request_rate_burst", 0.0) > 0.40
        )
        # Sensitive endpoint: must visit 3+ sensitive endpoints in one window
        has_sensitive_evidence = (feat_dict.get("sensitive_endpoint_count", 0.0) > 0.03)
        # Abnormal HTTP methods (e.g. TRACE, CONNECT, custom junk)
        has_method_evidence    = (feat_dict.get("abnormal_method_count", 0.0) > 0.0)
        # High 4xx: >70% of requests are client errors (not just an odd 404)
        has_error_evidence     = (feat_dict.get("status_4xx_rate", 0.0) > 0.70)
        # Scanning: sustained 404/403 storm – >15% each
        has_scanning_evidence  = (
            feat_dict.get("error_404_count", 0.0) > 0.15 and
            feat_dict.get("error_403_count", 0.0) > 0.15
        )

        threat_signals = [
            has_sqli_evidence, has_brute_evidence, has_rate_evidence,
            has_sensitive_evidence, has_method_evidence, has_error_evidence,
            has_scanning_evidence
        ]
        # Require ≥2 independent threat signals for anomaly (prevents single-feature FPs)
        has_threat_evidence = sum(threat_signals) >= 2 or has_sqli_evidence or has_brute_evidence

        # Anomaly requires both ML anomaly signal AND actual threat evidence
        is_ml_anomaly = bool(predictions["ensemble"][0] == 1) or (reconstruction_error > self.anomaly_threshold * 2.0)
        is_anomaly    = bool(is_ml_anomaly and has_threat_evidence)

        # Normal traffic without threat indicators stays safely in normal range
        if not is_anomaly:
            anomaly_score = min(anomaly_score * 0.25, 20.0)

        t_feature_done = time.monotonic()

        # Attack type classification grounded in evidence
        attack_type = "None"
        if is_anomaly:
            if has_sqli_evidence:
                attack_type = "SQLInjection"
            elif has_brute_evidence:
                attack_type = "BruteForce"
            elif has_rate_evidence:
                attack_type = "RateLimitAbuse"
            elif has_sensitive_evidence:
                attack_type = "SensitiveEndpoint"
            elif has_method_evidence:
                attack_type = "AbnormalMethod"
            elif has_scanning_evidence:
                attack_type = "BotScan"
            elif self.attack_classifier is not None:
                try:
                    attack_type = str(self.attack_classifier.predict(scaled_emb)[0])
                except Exception:
                    attack_type = "Unknown"
            else:
                attack_type = "Unknown"

        # ATRA risk scoring
        current_time = datetime.now()
        risk_score   = 0.0
        risk_level   = "Low"
        action       = "ALLOW"
        reason       = "Normal traffic"

        if is_anomaly:
            atk_weight   = ATTACK_TYPE_WEIGHT.get(attack_type, 5)
            severity     = ATTACK_SEVERITY.get(attack_type, 6)
            confidence   = normalize(reconstruction_error, 0, self.anomaly_threshold * 5, out_max=10.0)
            history_raw  = _ip_tracker.get_decayed_history_score(source_ip, current_time)
            history_score = min(history_raw * 2.0, 10.0)
            freq_raw     = _ip_tracker.get_frequency_last_window(source_ip, current_time)
            freq_score   = min(freq_raw * 2.5, 10.0)

            risk_score = compute_risk_score(atk_weight, severity, confidence, history_score, freq_score)
            risk_level = get_risk_level(risk_score)
            action     = RESPONSE_ACTIONS.get(risk_level, "ALERT")
            reason     = build_reason(feats)

            _ip_tracker.record_attack(source_ip, current_time)

        t_atra_done = time.monotonic()

        # Build security event
        event = {
            "timestamp":            current_time.isoformat(timespec="milliseconds"),
            "source_ip":            source_ip,
            "site_id":              site_id,
            "endpoint":             requests[-1].get("path", "/") if requests else "/",
            "method":               requests[-1].get("method", "GET") if requests else "GET",
            "request_count":        len(requests),
            "anomaly_score":        round(anomaly_score, 3),
            "reconstruction_error": round(reconstruction_error, 6),
            "is_anomaly":           is_anomaly,
            "attack_type":          attack_type,
            "risk_score":           round(risk_score, 2),
            "risk_level":           risk_level,
            "action":               action,
            "reason":               reason,
            "features":             get_feature_dict(feats),
            "timing": {
                "feature_ms":  round((t_feature_done - t_start) * 1000, 2),
                "atra_ms":     round((t_atra_done - t_feature_done) * 1000, 2),
                "total_ms":    round((t_atra_done - t_start) * 1000, 2),
            },
        }

        return event


# ============================================================
# TELEMETRY ENDPOINT (called by Node.js middleware)
# ============================================================

@app.route("/telemetry", methods=["POST"])
def receive_telemetry():
    """
    Receive a batch of HTTP request records from the Node.js webapp.
    Payload: { "source_ip": "...", "requests": [...] }
    """
    global _stats

    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    source_ip = data.get("source_ip", request.remote_addr or "unknown")
    site_id   = data.get("site_id", "default")
    requests_batch = data.get("requests", [])

    # Filter out internal telemetry / health / blocklist pings / external services to prevent false alarms
    clean_batch = []
    for r in requests_batch:
        raw_p = str(r.get("path", "") or r.get("endpoint", ""))
        path = raw_p.lower()
        if any(ign in path for ign in ("/telemetry", "/api/blocklist", "ipify", "/api/security", "firestore", "googleapis.com")):
            continue
        # Clean up full URLs to relative paths if any slipped through
        if "://" in raw_p:
            try:
                from urllib.parse import urlparse
                r["path"] = urlparse(raw_p).path or "/"
                r["endpoint"] = r["path"]
            except Exception:
                pass
        clean_batch.append(r)
    requests_batch = clean_batch

    if not requests_batch:
        return jsonify({"status": "ok", "skipped": True}), 200

    # Update stats
    with _stats_lock:
        _stats["total_requests"] += len(requests_batch)
        if "monitored_sites" not in _stats:
            _stats["monitored_sites"] = set()
        _stats["monitored_sites"].add(site_id)
        now_ts = time.time()
        _stats["request_times"].append(now_ts)
        # Keep only last 60 seconds for req/s calculation
        _stats["request_times"] = [t for t in _stats["request_times"] if now_ts - t < 60]

    if not _engine_ready:
        return jsonify({"status": "ok", "inference": "engine_not_ready"}), 200

    # Accumulate window for this IP
    with _window_lock:
        _ip_windows[source_ip].extend(requests_batch)
        window_s = WEB_CONFIG["window_seconds"]
        cutoff_ms = (time.time() - window_s) * 1000
        _ip_windows[source_ip] = [
            r for r in _ip_windows[source_ip]
            if float(r.get("timestamp_ms", 0)) >= cutoff_ms
        ]
        current_window = list(_ip_windows[source_ip])

    # Only run inference if we have enough data
    if len(current_window) < WEB_CONFIG["min_requests_per_window"]:
        return jsonify({"status": "ok", "skipped": True}), 200

    # Run ML inference
    try:
        event = _engine.infer(source_ip, current_window, site_id=site_id)
    except Exception as exc:
        logger.error(f"Inference error for {source_ip}: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500

    # Update stats and push event
    _update_stats_and_emit(event, source_ip)

    with _stats_lock:
        is_blocked = (event.get("action") in ("BLOCK", "SIMULATED_BLOCK")) or (source_ip in _stats["blocked_ips"])

    return jsonify({"status": "ok", "event": event, "blocked": is_blocked}), 200


_site_stats = defaultdict(lambda: {
    "total_requests": 0,
    "total_alerts": 0,
    "total_blocked": 0,
    "total_normal": 0,
    "blocked_ips": set(),
})

def _update_stats_and_emit(event: dict, source_ip: str):
    """Update in-memory stats and emit to connected dashboard clients."""
    site_id = event.get("site_id", "default")
    with _stats_lock:
        if "monitored_sites" not in _stats:
            _stats["monitored_sites"] = set()
        _stats["monitored_sites"].add(site_id)

        s_stats = _site_stats[site_id]

        if event["is_anomaly"]:
            _stats["total_alerts"] += 1
            s_stats["total_alerts"] += 1
            if event["action"] in ("BLOCK", "SIMULATED_BLOCK"):
                # NEVER block localhost loopback IPs (development protection)
                if source_ip not in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"):
                    _stats["total_blocked"] += 1
                    _stats["blocked_ips"].add(source_ip)
                    s_stats["total_blocked"] += 1
                    s_stats["blocked_ips"].add(source_ip)
        else:
            _stats["total_normal"] += 1
            s_stats["total_normal"] += 1

        # Ring buffer of recent events
        _stats["last_events"].insert(0, event)
        _stats["last_events"] = _stats["last_events"][:100]

        # Anomaly score history (for graphs)
        _stats["anomaly_scores"].append({
            "timestamp": event["timestamp"],
            "score":     event["anomaly_score"],
            "ip":        source_ip,
            "level":     event["risk_level"],
            "site_id":   site_id,
        })
        _stats["anomaly_scores"] = _stats["anomaly_scores"][-200:]

        # req/s calculation
        now = time.time()
        recent = [t for t in _stats["request_times"] if now - t < 10]
        _stats["requests_per_second"] = round(len(recent) / 10.0, 2)

    # Log to JSONL
    _log_event(event)

    # Emit to Socket.IO dashboard
    if event["is_anomaly"]:
        socketio.emit("security_alert", event)
    socketio.emit("stats_update", _build_stats_payload())


def _log_event(event: dict):
    """Append event to the JSONL security log."""
    try:
        log_path = Path(WEB_CONFIG["web_events_log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to write event log: {exc}")


def _build_stats_payload(site_id: str = None) -> dict:
    with _stats_lock:
        now = time.time()
        recent_10s = [t for t in _stats["request_times"] if now - t < 10]
        all_sites = list(_stats.get("monitored_sites", set()))

        if site_id and site_id != "all":
            s = _site_stats.get(site_id, {
                "total_requests": 0, "total_alerts": 0, "total_blocked": 0, "total_normal": 0, "blocked_ips": set()
            })
            site_events = [e for e in _stats["last_events"] if e.get("site_id") == site_id]
            site_scores = [pt for pt in _stats["anomaly_scores"] if pt.get("site_id") == site_id]
            return {
                "site_id":              site_id,
                "total_requests":       s["total_requests"],
                "total_alerts":         s["total_alerts"],
                "total_blocked":        s["total_blocked"],
                "total_normal":         s["total_normal"],
                "requests_per_second":  round(len(recent_10s) / 10.0, 2),
                "monitored_sites":      all_sites,
                "site_count":           len(all_sites),
                "blocked_ips":          list(s["blocked_ips"])[-20:],
                "recent_events":        site_events[:30],
                "anomaly_score_history":site_scores[-50:],
            }

        return {
            "site_id":              "all",
            "total_requests":       _stats["total_requests"],
            "total_alerts":         _stats["total_alerts"],
            "total_blocked":        _stats["total_blocked"],
            "total_normal":         _stats["total_normal"],
            "requests_per_second":  round(len(recent_10s) / 10.0, 2),
            "monitored_sites":      all_sites,
            "site_count":           len(all_sites),
            "blocked_ips":          list(_stats["blocked_ips"])[-20:],
            "recent_events":        _stats["last_events"][:30],
            "anomaly_score_history":_stats["anomaly_scores"][-50:],
        }


# ============================================================
# REST API ENDPOINTS (polled by dashboard)
# ============================================================

@app.route("/api/security/stats", methods=["GET"])
def api_stats():
    site_id = request.args.get("site")
    return jsonify(_build_stats_payload(site_id=site_id))


@app.route("/api/security/events", methods=["GET"])
def api_events():
    limit = int(request.args.get("limit", 50))
    site_id = request.args.get("site")
    with _stats_lock:
        evs = _stats["last_events"]
        if site_id and site_id != "all":
            evs = [e for e in evs if e.get("site_id") == site_id]
        return jsonify({"events": evs[:limit], "total": len(evs)})


@app.route("/api/security/alerts", methods=["GET"])
def api_alerts():
    site_id = request.args.get("site")
    with _stats_lock:
        alerts = [e for e in _stats["last_events"] if e.get("is_anomaly")]
        if site_id and site_id != "all":
            alerts = [e for e in alerts if e.get("site_id") == site_id]
    return jsonify({"alerts": alerts[:50], "count": len(alerts)})


@app.route("/api/security/blocked", methods=["GET"])
def api_blocked():
    with _stats_lock:
        return jsonify({"blocked_ips": list(_stats["blocked_ips"]), "count": len(_stats["blocked_ips"])})


# ============================================================
# BLOCKLIST ENDPOINTS (queried by security middleware)
# ============================================================

@app.route("/api/blocklist/check", methods=["GET"])
def check_blocklist():
    """Checked by security middleware on every incoming request for real-time enforcement."""
    ip = request.args.get("ip", "").strip()
    if ip in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"):
        return jsonify({"ip": ip, "blocked": False})
    with _stats_lock:
        is_blocked = ip in _stats["blocked_ips"]
    return jsonify({"ip": ip, "blocked": is_blocked})


@app.route("/api/blocklist/list", methods=["GET"])
def list_blocklist():
    """List all currently blocked IPs."""
    with _stats_lock:
        return jsonify({"blocked_ips": list(_stats["blocked_ips"]), "count": len(_stats["blocked_ips"])})


@app.route("/api/blocklist/unblock", methods=["GET", "POST"])
def unblock_ip():
    """Unblock a specific IP address."""
    ip = request.args.get("ip", "").strip() or (request.get_json(silent=True) or {}).get("ip", "").strip()
    if not ip:
        return jsonify({"error": "Missing IP parameter"}), 400
    with _stats_lock:
        if ip in _stats["blocked_ips"]:
            _stats["blocked_ips"].remove(ip)
        for s_stats in _site_stats.values():
            if ip in s_stats["blocked_ips"]:
                s_stats["blocked_ips"].remove(ip)
    with _window_lock:
        if ip in _ip_windows:
            _ip_windows[ip].clear()
    socketio.emit("stats_update", _build_stats_payload())
    return jsonify({"status": "ok", "unblocked": ip, "message": f"IP {ip} unblocked successfully"})


@app.route("/api/blocklist/clear", methods=["GET", "POST"])
def clear_blocklist():
    """Clear all blocked IPs and reset event stats."""
    with _stats_lock:
        _stats["blocked_ips"].clear()
        _stats["total_blocked"] = 0
        _stats["total_alerts"] = 0
        _stats["total_requests"] = 0
        _stats["total_normal"] = 0
        _stats["last_events"] = []
        _stats["anomaly_scores"] = []
        for s_stats in _site_stats.values():
            s_stats["blocked_ips"].clear()
            s_stats["total_blocked"] = 0
            s_stats["total_alerts"] = 0
            s_stats["total_requests"] = 0
            s_stats["total_normal"] = 0
    with _window_lock:
        _ip_windows.clear()
    socketio.emit("stats_update", _build_stats_payload())
    return jsonify({"status": "ok", "message": "All blocked IPs and stats cleared successfully"})


# ============================================================
# REVERSE PROXY LAYER
# Allows monitoring ANY website with ZERO code changes.
# Developer just registers their site + changes DNS CNAME.
# ============================================================

# Site registry: site_id -> { "origin": "https://real-server.com", "site_id": "..." }
_SITE_REGISTRY_PATH = Path(LOGS_DIR) / "proxy_sites.json"
_site_registry: dict = {}
_registry_lock = threading.Lock()


def _load_site_registry():
    """Load registered proxy sites from disk."""
    global _site_registry
    if _SITE_REGISTRY_PATH.exists():
        try:
            with open(_SITE_REGISTRY_PATH, encoding="utf-8") as f:
                _site_registry = json.load(f)
            logger.info(f"Loaded {len(_site_registry)} proxy site(s) from registry.")
        except Exception as exc:
            logger.warning(f"Could not load site registry: {exc}")
            _site_registry = {}


def _save_site_registry():
    """Persist site registry to disk."""
    try:
        _SITE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SITE_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(_site_registry, f, indent=2)
    except Exception as exc:
        logger.warning(f"Could not save site registry: {exc}")


@app.route("/api/proxy/register", methods=["POST"])
def register_proxy_site():
    """
    Register a website for zero-code-change reverse proxy monitoring.

    Body: {
      "site_id": "my-site",
      "origin":  "https://my-real-server.com",
      "domain":  "my-site.com"          ← optional: friend's real domain
    }

    After registration:
      Friend changes DNS CNAME: their-domain.com → this server
      Users visit their-domain.com → flows through our IDS → forwarded to origin
      Their code: UNTOUCHED. Their server: UNTOUCHED.
    """
    data = request.get_json(silent=True) or {}
    site_id = str(data.get("site_id", "")).strip()
    origin  = str(data.get("origin",  "")).strip().rstrip("/")
    domain  = str(data.get("domain",  "")).strip().lower()   # e.g. "john-site.com"

    if not site_id or not origin:
        return jsonify({"error": "site_id and origin are required"}), 400

    if not origin.startswith("http"):
        return jsonify({"error": "origin must start with http:// or https://"}), 400

    with _registry_lock:
        _site_registry[site_id] = {
            "site_id":    site_id,
            "origin":     origin,
            "domain":     domain,   # friend's real domain e.g. "john-site.com"
            "registered": datetime.now().isoformat(),
        }
        _save_site_registry()

    logger.info(f"Proxy site registered: {site_id} → {origin} (domain: {domain or 'none'})")
    return jsonify({
        "status":   "ok",
        "site_id":  site_id,
        "origin":   origin,
        "domain":   domain or None,
        "message":  (
            f"Site registered! "
            + (f"Change DNS: {domain} CNAME → gtae-atra-security.onrender.com" if domain
               else f"Use proxy URL: /proxy/{site_id}/")
        ),
        "proxy_url": f"https://gtae-atra-security.onrender.com/proxy/{site_id}/",
        "dns_setup": {
            "type":  "CNAME",
            "name":  domain or "your-domain.com",
            "value": "gtae-atra-security.onrender.com",
            "note":  "Also add domain in Render dashboard for SSL support"
        } if domain else None
    }), 201


@app.route("/api/proxy/sites", methods=["GET"])
def list_proxy_sites():
    """List all registered proxy sites."""
    with _registry_lock:
        return jsonify({"sites": list(_site_registry.values()), "count": len(_site_registry)})


@app.route("/api/proxy/unregister/<site_id>", methods=["DELETE"])
def unregister_proxy_site(site_id: str):
    """Remove a site from proxy monitoring."""
    with _registry_lock:
        removed = _site_registry.pop(site_id, None)
        if removed:
            _save_site_registry()
    if removed:
        return jsonify({"status": "ok", "removed": site_id})
    return jsonify({"error": "Site not found"}), 404


def _find_site_by_host(host: str) -> dict | None:
    """
    Find registered site by incoming Host header.
    Used for domain-based routing (friend's domain DNS → our server).
    """
    host_clean = host.split(":")[0].lower().strip()  # strip port
    with _registry_lock:
        for site in _site_registry.values():
            registered_domain = (site.get("domain") or "").lower().strip()
            if registered_domain and registered_domain == host_clean:
                return site
    return None


@app.route("/proxy/<site_id>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
@app.route("/proxy/<site_id>/", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
@app.route("/proxy/<site_id>/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
def reverse_proxy(site_id: str, subpath: str = ""):
    """
    Reverse proxy endpoint.

    Flow:
      Visitor → our-server.com/proxy/<site_id>/path
              → IDS analysis (IP check, telemetry)
              → Forward to registered origin
              → Return response to visitor

    Developer's server: completely untouched.
    Developer's code:   completely untouched.
    """
    if not _PROXY_AVAILABLE:
        return jsonify({"error": "Proxy unavailable: 'requests' library not installed"}), 503

    with _registry_lock:
        site = _site_registry.get(site_id)

    if not site:
        return jsonify({
            "error": f"Site '{site_id}' not registered.",
            "hint":  "POST /api/proxy/register with site_id + origin to register."
        }), 404

    origin = site["origin"]
    source_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()

    # ── Block check ────────────────────────────────────────────
    with _stats_lock:
        if source_ip in _stats["blocked_ips"]:
            return jsonify({
                "error":   "Forbidden",
                "message": "Your IP is blocked by GTAE-ATRA Security Engine.",
                "code":    "GTAE_ATRA_BLOCKED",
            }), 403

    # ── Build forward URL ──────────────────────────────────────
    path = "/" + subpath if subpath else "/"
    query = request.query_string.decode("utf-8")
    target_url = f"{origin}{path}" + (f"?{query}" if query else "")

    # ── Forward request ────────────────────────────────────────
    start_ms = time.time() * 1000
    try:
        # Strip hop-by-hop headers
        forward_headers = {
            k: v for k, v in request.headers
            if k.lower() not in ("host", "connection", "transfer-encoding")
        }
        forward_headers["X-Forwarded-For"] = source_ip
        forward_headers["X-Forwarded-Host"] = request.host
        forward_headers["X-GTAE-ATRA-Proxy"] = "1"

        resp = _requests_lib.request(
            method  = request.method,
            url     = target_url,
            headers = forward_headers,
            data    = request.get_data(),
            timeout = 30,
            allow_redirects = False,
            stream  = False,
        )
        latency_ms = time.time() * 1000 - start_ms

    except Exception as exc:
        logger.error(f"[Proxy] Forward failed for {site_id}: {exc}")
        return jsonify({"error": "Upstream unreachable", "detail": str(exc)}), 502

    # ── Telemetry: analyze this request ───────────────────────
    record = {
        "timestamp_ms":    int(start_ms),
        "method":          request.method,
        "path":            path,
        "query":           query,
        "status":          resp.status_code,
        "status_code":     resp.status_code,
        "response_time_ms":round(latency_ms, 1),
        "request_size":    int(request.headers.get("Content-Length", 0) or 0),
        "response_size":   int(resp.headers.get("Content-Length", 0) or 0),
        "user_agent":      request.headers.get("User-Agent", "")[:200],
    }

    with _window_lock:
        _ip_windows[source_ip].append(record)
        window_s  = WEB_CONFIG["window_seconds"]
        cutoff_ms = (time.time() - window_s) * 1000
        _ip_windows[source_ip] = [
            r for r in _ip_windows[source_ip]
            if float(r.get("timestamp_ms", 0)) >= cutoff_ms
        ]
        current_window = list(_ip_windows[source_ip])

    with _stats_lock:
        _stats["total_requests"] += 1

    # Run inference if window is ready
    if _engine_ready and len(current_window) >= WEB_CONFIG["min_requests_per_window"]:
        try:
            event = _engine.infer(source_ip, current_window, site_id=site_id)
            _update_stats_and_emit(event, source_ip)
            # Block if engine says so
            if event.get("action") in ("BLOCK", "SIMULATED_BLOCK"):
                with _stats_lock:
                    if source_ip not in ("127.0.0.1", "::1"):
                        _stats["blocked_ips"].add(source_ip)
        except Exception as exc:
            logger.error(f"[Proxy] Inference error: {exc}")

    # ── Return upstream response to visitor ───────────────────
    excluded_resp_headers = {"transfer-encoding", "connection", "content-encoding"}
    response_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in excluded_resp_headers
    }

    from flask import Response
    return Response(
        response = resp.content,
        status   = resp.status_code,
        headers  = response_headers,
    )


# ── Host-header based catch-all (Domain DNS approach) ─────────────────────────
# When friend's domain DNS points to our server:
#   User visits: john-site.com/about
#   Host header: john-site.com
#   We look up "john-site.com" in registry → find origin → forward
#   User sees normal content, never knows about proxy
@app.route("/<path:subpath>", methods=["GET","POST","PUT","DELETE","PATCH","HEAD","OPTIONS"])
@app.route("/", methods=["GET","POST","PUT","DELETE","PATCH","HEAD","OPTIONS"])
def host_based_proxy(subpath: str = ""):
    """
    Catch-all route for domain-based proxying.
    Only activates when the Host header matches a registered domain.
    All GTAE-ATRA internal API paths (/api/, /telemetry, /proxy/) take priority
    and are handled by their own routes first.
    """
    host = request.headers.get("Host", "")
    site = _find_site_by_host(host)

    # Not a registered domain → show normal server info (don't break existing routes)
    if not site:
        return jsonify({
            "status":  "online",
            "service": "GTAE-ATRA Cloud Security Engine",
            "hint":    "To monitor your site, POST /api/proxy/register"
        }), 200

    # Found registered domain → proxy the request
    if not _PROXY_AVAILABLE:
        return jsonify({"error": "Proxy unavailable"}), 503

    origin    = site["origin"]
    site_id   = site["site_id"]
    source_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()

    # Block check
    with _stats_lock:
        if source_ip in _stats["blocked_ips"]:
            return jsonify({
                "error":   "Forbidden",
                "message": "Your IP is blocked by GTAE-ATRA Security Engine.",
                "code":    "GTAE_ATRA_BLOCKED",
            }), 403

    path       = "/" + subpath if subpath else "/"
    query      = request.query_string.decode("utf-8")
    target_url = f"{origin}{path}" + (f"?{query}" if query else "")

    start_ms = time.time() * 1000
    try:
        fwd_headers = {
            k: v for k, v in request.headers
            if k.lower() not in ("host", "connection", "transfer-encoding")
        }
        fwd_headers["X-Forwarded-For"]   = source_ip
        fwd_headers["X-Forwarded-Host"]  = host
        fwd_headers["X-GTAE-ATRA-Proxy"] = "1"

        resp = _requests_lib.request(
            method=request.method, url=target_url,
            headers=fwd_headers, data=request.get_data(),
            timeout=30, allow_redirects=False, stream=False,
        )
        latency_ms = time.time() * 1000 - start_ms
    except Exception as exc:
        logger.error(f"[HostProxy] Forward failed: {exc}")
        return jsonify({"error": "Upstream unreachable"}), 502

    # Telemetry
    record = {
        "timestamp_ms": int(start_ms), "method": request.method,
        "path": path, "query": query,
        "status": resp.status_code, "status_code": resp.status_code,
        "response_time_ms": round(latency_ms, 1),
        "request_size": int(request.headers.get("Content-Length", 0) or 0),
        "response_size": int(resp.headers.get("Content-Length", 0) or 0),
        "user_agent": request.headers.get("User-Agent", "")[:200],
    }
    with _window_lock:
        _ip_windows[source_ip].append(record)
        cutoff_ms = (time.time() - WEB_CONFIG["window_seconds"]) * 1000
        _ip_windows[source_ip] = [r for r in _ip_windows[source_ip]
                                   if float(r.get("timestamp_ms", 0)) >= cutoff_ms]
        current_window = list(_ip_windows[source_ip])

    with _stats_lock:
        _stats["total_requests"] += 1

    if _engine_ready and len(current_window) >= WEB_CONFIG["min_requests_per_window"]:
        try:
            event = _engine.infer(source_ip, current_window, site_id=site_id)
            _update_stats_and_emit(event, source_ip)
            if event.get("action") in ("BLOCK", "SIMULATED_BLOCK"):
                with _stats_lock:
                    if source_ip not in ("127.0.0.1", "::1"):
                        _stats["blocked_ips"].add(source_ip)
        except Exception as exc:
            logger.error(f"[HostProxy] Inference error: {exc}")

    excluded = {"transfer-encoding", "connection", "content-encoding"}
    resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}
    from flask import Response
    return Response(response=resp.content, status=resp.status_code, headers=resp_headers)




@app.route("/api/security/anomaly", methods=["GET"])
def api_anomaly():
    with _stats_lock:
        return jsonify({"history": _stats["anomaly_scores"][-100:]})


@app.route("/api/traffic/stats", methods=["GET"])
def api_traffic_stats():
    return jsonify(_build_stats_payload())


@app.route("/", methods=["GET", "HEAD"])
def root_index():
    return jsonify({
        "status": "online",
        "service": "GTAE-ATRA Cloud Security Engine",
        "engine_ready": _engine_ready,
        "endpoints": {
            "health": "/api/health",
            "stats": "/api/security/stats",
            "telemetry": "/telemetry (POST)",
            "blocklist": "/api/blocklist/check?ip=<ip>"
        }
    }), 200


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "engine_ready": _engine_ready,
        "timestamp":    datetime.now().isoformat(),
    })


@app.route("/api/logs", methods=["GET"])
def api_logs():
    """Return recent log events from the JSONL file."""
    log_path = Path(WEB_CONFIG["web_events_log"])
    events = []
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines[-200:]):
                try:
                    events.append(json.loads(line.strip()))
                except Exception:
                    pass
        except Exception:
            pass
    return jsonify({"events": events[:100], "count": len(events)})


# ============================================================
# SOCKET.IO EVENTS
# ============================================================

@socketio.on("connect")
def on_connect():
    logger.info(f"Dashboard client connected: {request.sid}")
    # pyrefly: ignore [unexpected-keyword]
    socketio.emit("stats_update", _build_stats_payload(), room=request.sid)


@socketio.on("disconnect")
def on_disconnect():
    logger.info(f"Dashboard client disconnected: {request.sid}")


# ============================================================
# STARTUP
# ============================================================

def _load_engine():
    global _engine, _engine_ready, _anomaly_threshold
    _load_site_registry()
    try:
        _engine = WebInferenceEngine()
        _anomaly_threshold = _engine.anomaly_threshold
        _engine_ready = True
        logger.info("Inference engine loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning(f"Models not found: {exc}")
        logger.warning("Run 'python src/web_train.py' first, then restart the monitor server.")
        _engine_ready = False


def start_monitor_server(
    host: str = None,
    port: int = None,
    debug: bool = False,
):
    host = host or WEB_CONFIG["monitor_host"]
    port = port or WEB_CONFIG["monitor_port"]

    print("=" * 70)
    print("  GTAE-ATRA WEB SECURITY MONITOR SERVER")
    print("=" * 70)
    print(f"  Telemetry endpoint  : http://{host}:{port}/telemetry")
    print(f"  Dashboard API       : http://{host}:{port}/api/security/stats")
    print(f"  Health check        : http://{host}:{port}/api/health")
    print("=" * 70)

    _load_engine()
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    start_monitor_server()
