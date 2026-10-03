"""
src/web_feature_extractor.py
Converts raw HTTP request telemetry into normalized feature vectors
for the GTAE-ATRA web security monitoring system.

Input:  List of request dicts from Node.js middleware (one window per IP)
Output: numpy array [20 features] normalized to [0, 1]

Feature set (must match WEB_CONFIG["feature_names"]):
  0  request_rate               req/min in window
  1  unique_endpoint_count      distinct paths visited
  2  failed_login_count         POST /login with non-2xx response
  3  status_4xx_rate            fraction of 4xx responses
  4  status_5xx_rate            fraction of 5xx responses
  5  avg_response_time_ms       mean latency ms
  6  avg_path_length            mean URL path length (chars)
  7  avg_query_length           mean query-string length (chars)
  8  post_ratio                 fraction of POST requests
  9  get_ratio                  fraction of GET requests
  10 sensitive_endpoint_count   hits on sensitive endpoints
  11 repeated_endpoint_ratio    most-visited endpoint / total
  12 suspicious_pattern_count   URL pattern matches (SQLi/XSS/traversal)
  13 abnormal_method_count      non-GET/POST/PUT/DELETE/HEAD methods
  14 avg_request_size           mean request body size (bytes)
  15 max_request_rate_burst     peak req/s in any 5s sub-window
  16 error_404_count            404 response count
  17 error_403_count            403 response count
  18 user_agent_entropy         Shannon entropy of user-agent strings (0-1)
  19 req_per_minute             raw request count / window_minutes
"""

import sys
import re
import math
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from config import WEB_CONFIG
from utils import get_logger

logger = get_logger(__name__)

FEATURE_DIM = WEB_CONFIG["feature_dim"]
SENSITIVE_ENDPOINTS = set(WEB_CONFIG["sensitive_endpoints"])
SUSPICIOUS_PATTERNS = [re.compile(p, re.IGNORECASE)
                       for p in WEB_CONFIG["suspicious_url_patterns"]]
NORMAL_METHODS = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}

# Normalization bounds calibrated on benign traffic ranges
_RAW_BOUNDS = {
    "request_rate":             (0,   300),
    "unique_endpoint_count":    (0,   50),
    "failed_login_count":       (0,   100),
    "status_4xx_rate":          (0,   1.0),
    "status_5xx_rate":          (0,   1.0),
    "avg_response_time_ms":     (0,   30_000),
    "avg_path_length":          (0,   200),
    "avg_query_length":         (0,   500),
    "post_ratio":               (0,   1.0),
    "get_ratio":                (0,   1.0),
    "sensitive_endpoint_count": (0,   100),
    "repeated_endpoint_ratio":  (0,   1.0),
    "suspicious_pattern_count": (0,   50),
    "abnormal_method_count":    (0,   30),
    "avg_request_size":         (0,   2_000_000),
    "max_request_rate_burst":   (0,   100),
    "error_404_count":          (0,   200),
    "error_403_count":          (0,   200),
    "user_agent_entropy":       (0,   1.0),
    "req_per_minute":           (0,   300),
}


def _shannon_entropy(strings: list) -> float:
    """Normalized Shannon entropy of a list of strings (0-1 scale)."""
    if not strings:
        return 0.0
    counts = Counter(strings)
    total = len(strings)
    probs = [c / total for c in counts.values()]
    raw_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return raw_entropy / max_entropy if max_entropy > 0 else 0.0


def _count_suspicious_patterns(path: str, query: str) -> int:
    """Count how many suspicious patterns match in path+query."""
    text = (path or "") + "?" + (query or "")
    return sum(1 for pat in SUSPICIOUS_PATTERNS if pat.search(text))


def extract_features(requests: list, window_seconds: float = None) -> np.ndarray:
    """
    Extract a 20-dim feature vector from a list of request dicts.

    Args:
        requests: List of request dicts with keys:
            - timestamp_ms: epoch milliseconds
            - method: HTTP method string
            - path: URL path (no query)
            - query: query string (may be empty)
            - status: HTTP status code int
            - response_time_ms: float
            - request_size: int (body bytes)
            - response_size: int
            - user_agent: string
        window_seconds: duration of window; auto-computed if None

    Returns:
        np.ndarray shape [20] normalized to [0, 1]
    """
    if not requests:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    window_s = window_seconds or WEB_CONFIG["window_seconds"]
    window_min = window_s / 60.0

    n = len(requests)
    methods   = [r.get("method", "GET").upper() for r in requests]
    paths     = [r.get("path", "/") for r in requests]
    queries   = [r.get("query", "") or "" for r in requests]
    statuses  = [int(r.get("status", 200)) for r in requests]
    latencies = [float(r.get("response_time_ms", 0)) for r in requests]
    req_sizes = [int(r.get("request_size", 0)) for r in requests]
    agents    = [r.get("user_agent", "") or "" for r in requests]
    timestamps = [float(r.get("timestamp_ms", 0)) for r in requests]

    req_per_min  = n / window_min if window_min > 0 else 0.0
    request_rate = req_per_min
    unique_endpoints = len(set(paths))

    failed_logins = sum(
        1 for r in requests
        if r.get("method", "").upper() == "POST"
        and r.get("path", "").rstrip("/") in ("/login", "/api/login", "/api/auth")
        and int(r.get("status", 200)) not in range(200, 300)
    )

    count_4xx = sum(1 for s in statuses if 400 <= s < 500)
    count_5xx = sum(1 for s in statuses if 500 <= s < 600)
    status_4xx_rate = count_4xx / n if n > 0 else 0.0
    status_5xx_rate = count_5xx / n if n > 0 else 0.0

    avg_response_ms = float(np.mean(latencies)) if latencies else 0.0
    avg_path_len    = float(np.mean([len(p) for p in paths]))
    avg_query_len   = float(np.mean([len(q) for q in queries]))

    post_count = methods.count("POST")
    get_count  = methods.count("GET")
    post_ratio = post_count / n if n > 0 else 0.0
    get_ratio  = get_count  / n if n > 0 else 0.0

    sensitive_count = sum(
        1 for p in paths if any(p.startswith(s) for s in SENSITIVE_ENDPOINTS)
    )

    endpoint_counts = Counter(paths)
    most_common_count = endpoint_counts.most_common(1)[0][1] if endpoint_counts else 0
    # A single or pair of requests is normal browsing, not repetitive probing
    repeated_endpoint_ratio = (most_common_count / n) if n >= 5 else 0.0

    susp_count = sum(
        _count_suspicious_patterns(p, q) for p, q in zip(paths, queries)
    )

    abnormal_method_count = sum(1 for m in methods if m not in NORMAL_METHODS)
    avg_req_size = float(np.mean(req_sizes)) if req_sizes else 0.0

    # Max burst in any 5-second sub-window
    max_burst = 0.0
    if len(timestamps) >= 2:
        sorted_ts = sorted(timestamps)
        sub_window_ms = 5000.0
        left = 0
        for right in range(len(sorted_ts)):
            while sorted_ts[right] - sorted_ts[left] > sub_window_ms:
                left += 1
            burst_count = right - left + 1
            burst_rate  = burst_count / (sub_window_ms / 1000.0)
            max_burst   = max(max_burst, burst_rate)

    error_404  = sum(1 for s in statuses if s == 404)
    error_403  = sum(1 for s in statuses if s == 403)
    ua_entropy = _shannon_entropy(agents)

    raw = np.array([
        request_rate,
        unique_endpoints,
        failed_logins,
        status_4xx_rate,
        status_5xx_rate,
        avg_response_ms,
        avg_path_len,
        avg_query_len,
        post_ratio,
        get_ratio,
        sensitive_count,
        repeated_endpoint_ratio,
        susp_count,
        abnormal_method_count,
        avg_req_size,
        max_burst,
        error_404,
        error_403,
        ua_entropy,
        req_per_min,
    ], dtype=np.float64)

    feature_names = WEB_CONFIG["feature_names"]
    normalized = np.zeros(FEATURE_DIM, dtype=np.float32)
    for i, name in enumerate(feature_names):
        lo, hi = _RAW_BOUNDS[name]
        span = hi - lo if hi != lo else 1.0
        normalized[i] = float(np.clip((raw[i] - lo) / span, 0.0, 1.0))

    return normalized


def extract_features_from_json(json_str: str) -> np.ndarray:
    """Parse a JSON string of request list and extract features."""
    try:
        requests = json.loads(json_str)
        return extract_features(requests)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning(f"Feature extraction failed: {exc}")
        return np.zeros(FEATURE_DIM, dtype=np.float32)


def get_feature_dict(features: np.ndarray) -> dict:
    """Return a dict mapping feature names to their (normalized) values."""
    return {n: float(features[i])
            for i, n in enumerate(WEB_CONFIG["feature_names"])}


def build_reason(features: np.ndarray, threshold: float = 0.4) -> str:
    """Build a human-readable reason string from elevated feature values."""
    names    = WEB_CONFIG["feature_names"]
    readable = {
        "request_rate":             "High request rate",
        "failed_login_count":       "Multiple failed login attempts",
        "status_4xx_rate":          "High 4xx error rate",
        "status_5xx_rate":          "High 5xx error rate",
        "sensitive_endpoint_count": "Repeated access to sensitive endpoints",
        "suspicious_pattern_count": "Suspicious URL patterns detected (SQLi/XSS/traversal)",
        "abnormal_method_count":    "Unusual HTTP methods",
        "max_request_rate_burst":   "High burst request rate",
        "error_404_count":          "Multiple 404 errors (scanning behaviour)",
        "error_403_count":          "Multiple 403 errors (access denial violations)",
        "repeated_endpoint_ratio":  "Repetitive endpoint targeting (possible probing)",
        "avg_query_length":         "Abnormally long query strings",
    }
    reasons = [readable[n] for i, n in enumerate(names)
               if float(features[i]) >= threshold and n in readable]
    return "; ".join(reasons) if reasons else "Anomalous traffic pattern detected"


if __name__ == "__main__":
    import random, time
    synthetic = []
    now = time.time() * 1000
    for i in range(20):
        synthetic.append({
            "timestamp_ms":    now + i * 1500,
            "method":          random.choice(["GET", "POST"]),
            "path":            random.choice(["/", "/home", "/products", "/login"]),
            "query":           "",
            "status":          random.choice([200, 200, 200, 404, 500]),
            "response_time_ms":random.uniform(10, 200),
            "request_size":    random.randint(0, 500),
            "response_size":   random.randint(100, 5000),
            "user_agent":      "Mozilla/5.0",
        })
    feats = extract_features(synthetic, window_seconds=30)
    print("Feature vector shape:", feats.shape)
    for name, val in zip(WEB_CONFIG["feature_names"], feats):
        print(f"  {name:<30}: {val:.4f}")
    print("Reason:", build_reason(feats))
