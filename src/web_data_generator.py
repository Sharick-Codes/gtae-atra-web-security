"""
src/web_data_generator.py
Generates synthetic web traffic sessions for training the GTAE-ATRA
web security monitoring system.

Produces:
  - Benign sessions: normal user browsing patterns
  - Attack sessions: brute force, scanning, SQLi, XSS, path traversal,
                     rate abuse, bot scanning, and combined suspicious patterns

No real attack tools or destructive payloads are generated.
All data is synthetic and stays local.
"""

import sys
import pickle
import random
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import WEB_CONFIG
from web_feature_extractor import extract_features
from utils import get_logger

logger = get_logger(__name__)

random.seed(42)
np.random.seed(42)

BENIGN_ENDPOINTS = [
    ("/", "GET"), ("/index.html", "GET"), ("/home", "GET"), ("/products", "GET"),
    ("/about", "GET"), ("/contact", "GET"), ("/blog", "GET"),
    ("/api/products", "GET"), ("/api/categories", "GET"),
    ("/profile", "GET"), ("/login", "POST"), ("/api/search", "GET"),
    ("/api/explain", "POST"), ("/api/chat", "POST"), ("/dashboard", "GET"),
]

USER_AGENTS_BENIGN = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
]

USER_AGENTS_BOT = [
    "python-requests/2.28.0", "curl/7.88.1", "Go-http-client/1.1", "Nikto/2.1.6",
]

SUSPICIOUS_METHODS = ["TRACE", "OPTIONS", "CONNECT", "PROPFIND", "MOVE"]


def _ms_now():
    return time.time() * 1000


def _make_request(path, method="GET", status=200, latency_ms=None,
                  query="", req_size=0, user_agent=None, ts_offset=0):
    return {
        "timestamp_ms":     _ms_now() + ts_offset,
        "method":           method,
        "path":             path,
        "query":            query,
        "status":           status,
        "response_time_ms": latency_ms or random.uniform(20, 1500),
        "request_size":     req_size,
        "response_size":    random.randint(500, 8000),
        "user_agent":       user_agent or random.choice(USER_AGENTS_BENIGN),
    }


def _gen_benign_session(window_s=30):
    n = random.randint(3, 12)
    return [_make_request(
        *random.choice(BENIGN_ENDPOINTS),
        status=random.choices([200, 404, 302], weights=[94, 4, 2])[0],
        latency_ms=random.uniform(20, 2000),
        ts_offset=i * random.uniform(500, 4000),
        user_agent=random.choice(USER_AGENTS_BENIGN),
    ) for i in range(n)]


def _gen_single_page_visit(window_s=30):
    """User visiting just 1 or 2 pages or calling an endpoint (very common)."""
    n = random.choice([1, 1, 2])
    endpoint, method = random.choice(BENIGN_ENDPOINTS)
    req_size = random.randint(500, 150_000) if method == "POST" else 0
    return [_make_request(
        endpoint,
        method=method,
        status=200,
        latency_ms=random.uniform(50, 4000),
        req_size=req_size,
        ts_offset=i * random.uniform(1000, 5000),
        user_agent=random.choice(USER_AGENTS_BENIGN),
    ) for i in range(n)]


def _gen_api_session(window_s=30):
    """Normal user interacting with API endpoints (like AI research paper explain, chat, search)."""
    api_endpoints = [("/api/explain", "POST"), ("/api/chat", "POST"), ("/api/search", "GET"), ("/api/products", "GET")]
    n = random.randint(1, 4)
    reqs = []
    for i in range(n):
        ep, m = random.choice(api_endpoints)
        req_size = random.randint(2000, 250_000) if m == "POST" else 0
        latency = random.uniform(500, 8000) if m == "POST" else random.uniform(50, 600)
        reqs.append(_make_request(
            ep, method=m, status=200, latency_ms=latency, req_size=req_size,
            ts_offset=i * random.uniform(2000, 8000)
        ))
    return reqs


def _gen_logged_in_session(window_s=30):
    reqs = [_make_request("/login", "POST", 200, ts_offset=0)]
    pages = ["/profile", "/api/products", "/products", "/home", "/api/categories"]
    for i in range(random.randint(2, 6)):
        reqs.append(_make_request(random.choice(pages), "GET", 200,
                                  ts_offset=(i+1)*random.uniform(1000, 5000)))
    return reqs


def _gen_brute_force(window_s=30):
    return [_make_request("/login", "POST", 401,
                          latency_ms=random.uniform(100, 500),
                          req_size=random.randint(50, 200),
                          ts_offset=i * random.uniform(300, 1200),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i in range(random.randint(15, 60))]


def _gen_rate_abuse(window_s=30):
    ep = random.choice(["/", "/api/products", "/home"])
    return [_make_request(ep, "GET", random.choice([200, 503]),
                          latency_ms=random.uniform(5, 50),
                          ts_offset=i * random.uniform(50, 300),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i in range(random.randint(100, 250))]


def _gen_path_traversal(window_s=30):
    paths = [
        "/../../../etc/passwd", "/../../windows/win.ini",
        "/.env", "/config/database.yml", "/.git/config",
        "/%2e%2e%2f%2e%2e%2fetc/passwd",
    ]
    return [_make_request(p, "GET", random.choice([403, 404, 200]),
                          ts_offset=i * random.uniform(200, 1000),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i, p in enumerate(paths * random.randint(2, 5))]


def _gen_sqli_probe(window_s=30):
    queries = [
        "id=1' OR '1'='1", "id=1 UNION SELECT * FROM users",
        "search='; DROP TABLE users;--", "q=admin' AND 1=1--",
    ]
    return [_make_request("/api/products", "GET", random.choice([200, 400, 500]),
                          query=random.choice(queries),
                          ts_offset=i * random.uniform(500, 2000),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i in range(random.randint(5, 20))]


def _gen_xss_probe(window_s=30):
    payloads = [
        "<script>alert('xss')</script>", "javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
    ]
    return [_make_request("/api/search", "GET", random.choice([200, 400]),
                          query=f"q={random.choice(payloads)}",
                          ts_offset=i * random.uniform(300, 1500),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i in range(random.randint(5, 15))]


def _gen_sensitive_scan(window_s=30):
    targets = [
        ("/admin", "GET"), ("/wp-admin", "GET"), ("/phpmyadmin", "GET"),
        ("/.env", "GET"), ("/config", "GET"), ("/api/admin", "GET"),
        ("/api/users", "GET"), ("/api/keys", "GET"), ("/debug", "GET"),
    ]
    return [_make_request(p, m, random.choice([403, 404, 200]),
                          ts_offset=i * random.uniform(200, 800),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i, (p, m) in enumerate(targets * random.randint(1, 3))]


def _gen_bot_scan(window_s=30):
    paths = [f"/{w}" for w in [
        "robots.txt", "sitemap.xml", "backup.sql", "dump.sql", "data.zip",
        "secret", "private", "test", "wp-content", "uploads", "assets",
    ]]
    return [_make_request(p, "GET", random.choice([404, 403, 200]),
                          ts_offset=i * random.uniform(100, 600),
                          user_agent=random.choice(USER_AGENTS_BOT))
            for i, p in enumerate(paths * random.randint(1, 2))]


def _gen_abnormal_methods(window_s=30):
    return [_make_request(random.choice(["/", "/api/products", "/admin"]),
                          random.choice(SUSPICIOUS_METHODS),
                          status=random.choice([405, 200, 501]),
                          ts_offset=i * random.uniform(500, 2000))
            for i in range(random.randint(5, 20))]


def _gen_large_payload(window_s=30):
    return [_make_request("/api/submit", "POST", random.choice([200, 413, 500]),
                          req_size=random.randint(50_000, 200_000),
                          ts_offset=i * random.uniform(500, 3000))
            for i in range(random.randint(3, 10))]


def _gen_combined_attack(window_s=30):
    reqs = (_gen_brute_force()[:10] + _gen_sensitive_scan()[:8] + _gen_sqli_probe()[:5])
    random.shuffle(reqs)
    return reqs


ATTACK_GENERATORS = {
    "BruteForce":        _gen_brute_force,
    "RateLimitAbuse":    _gen_rate_abuse,
    "PathTraversal":     _gen_path_traversal,
    "SQLInjection":      _gen_sqli_probe,
    "XSS":               _gen_xss_probe,
    "SensitiveEndpoint": _gen_sensitive_scan,
    "BotScan":           _gen_bot_scan,
    "AbnormalMethod":    _gen_abnormal_methods,
    "LargePayload":      _gen_large_payload,
    "Combined":          _gen_combined_attack,
}

BENIGN_GENERATORS = [
    _gen_benign_session,
    _gen_single_page_visit,
    _gen_api_session,
    _gen_logged_in_session,
]


def _make_record(flow_id, src_ip, dst_node, requests, label, attack_type):
    feats = extract_features(requests, window_seconds=WEB_CONFIG["window_seconds"])
    return {
        "flow_id":       flow_id,
        "src_node":      f"client:{src_ip}",
        "dst_node":      dst_node,
        "edge_features": feats,
        "label":         label,
        "attack_type":   attack_type,
        "raw_requests":  requests,
        "source_ip":     src_ip,
    }


def generate_web_records(num_benign=None, num_attack=None) -> tuple:
    """Generate (benign_records, attack_records) in graph-edge format."""
    num_benign = num_benign or WEB_CONFIG["num_benign_sessions"]
    num_attack = num_attack or WEB_CONFIG["num_attack_sessions"]
    server_node = "endpoint:webserver"
    flow_id = 0
    benign_records, attack_records = [], []

    logger.info(f"Generating {num_benign} benign web sessions...")
    for _ in range(num_benign):
        ip = f"10.0.{random.randint(1,10)}.{random.randint(1,254)}"
        reqs = random.choice(BENIGN_GENERATORS)()
        benign_records.append(_make_record(flow_id, ip, server_node, reqs, "benign", "None"))
        flow_id += 1

    attack_types = list(ATTACK_GENERATORS.keys())
    per_type = max(1, num_attack // len(attack_types))
    remainder = num_attack - per_type * len(attack_types)

    logger.info(f"Generating {num_attack} attack web sessions ({len(attack_types)} types)...")
    for atk_type, gen_fn in ATTACK_GENERATORS.items():
        count = per_type + (1 if remainder > 0 else 0)
        remainder -= 1
        for _ in range(count):
            ip = f"172.16.{random.randint(1,5)}.{random.randint(1,254)}"
            reqs = gen_fn()
            attack_records.append(
                _make_record(flow_id, ip, server_node, reqs, "attack", atk_type))
            flow_id += 1

    logger.info(f"Generated {len(benign_records)} benign + {len(attack_records)} attack records")
    return benign_records, attack_records


def main():
    logger.info("=" * 60)
    logger.info("WEB DATA GENERATOR -- Synthetic Web Traffic")
    logger.info("=" * 60)

    benign_records, attack_records = generate_web_records()

    benign_path = Path(WEB_CONFIG["web_benign_path"])
    attack_path = Path(WEB_CONFIG["web_attack_path"])
    benign_path.parent.mkdir(parents=True, exist_ok=True)

    with open(benign_path, "wb") as f:
        pickle.dump(benign_records, f)
    with open(attack_path, "wb") as f:
        pickle.dump(attack_records, f)

    print(f"\n[OK] Saved {len(benign_records)} benign records -> {benign_path}")
    print(f"[OK] Saved {len(attack_records)} attack records -> {attack_path}")
    print(f"     Feature dim: {benign_records[0]['edge_features'].shape[0]}")

    from collections import Counter
    print("\n--- Attack Type Distribution ---")
    for atype, cnt in sorted(Counter(r["attack_type"] for r in attack_records).items()):
        print(f"  {atype:<25}: {cnt}")


if __name__ == "__main__":
    main()
