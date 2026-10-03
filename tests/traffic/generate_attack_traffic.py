"""
tests/traffic/generate_attack_traffic.py
Controlled suspicious traffic simulation for GTAE-ATRA-NIDS testing.

IMPORTANT:
  - All traffic is directed to the LOCAL demo app (http://127.0.0.1:3000)
  - No real attack tools, malware, or destructive payloads are used
  - This is purely for evaluating the detection system locally
  - The X-Forwarded-For header spoofs IPs so the ML system sees distinct sources

Usage:
  python tests/traffic/generate_attack_traffic.py --scenario brute_force
  python tests/traffic/generate_attack_traffic.py --scenario all
  python tests/traffic/generate_attack_traffic.py --list
"""

import argparse
import random
import time
import sys
import requests
from datetime import datetime

BASE_URL = "http://127.0.0.1:3000"

ATTACK_IP_BASE = "172.16.1."  # simulated attacker subnet


def _req(session, method, path, data=None, params=None, headers_extra=None, attacker_ip="172.16.1.100"):
    url = f"{BASE_URL}{path}"
    headers = {
        "User-Agent":       random.choice([
            "python-requests/2.28.0", "curl/7.88.1", "Go-http-client/1.1",
        ]),
        "X-Forwarded-For":  attacker_ip,
        "Accept":           "*/*",
        **(headers_extra or {}),
    }
    try:
        if method == "GET":
            r = session.get(url, params=params, headers=headers, timeout=4, allow_redirects=False)
        else:
            r = session.post(url, data=data, json=data if isinstance(data, dict) else None,
                             headers=headers, timeout=4, allow_redirects=False)
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"  [{ts}] {method:4} {path:<40} -> {r.status_code}")
        return r.status_code
    except Exception as e:
        print(f"  [ERR] {path}: {e}")
        return None


# ---- Scenario definitions -----------------------------------------------

def scenario_brute_force(session):
    """Simulate repeated failed login attempts from one IP."""
    print("\n[SCENARIO] Brute Force Login Attack")
    ip = f"{ATTACK_IP_BASE}{random.randint(100,150)}"
    for i in range(40):
        _req(session, "POST", "/login",
             data={"username": "admin", "password": f"wrong{i}"},
             attacker_ip=ip)
        time.sleep(0.3)
    print("  Done: 40 failed login attempts")


def scenario_rate_abuse(session):
    """Simulate high-frequency requests from one IP."""
    print("\n[SCENARIO] Rate Limit Abuse / DoS Simulation")
    ip = f"{ATTACK_IP_BASE}{random.randint(50,80)}"
    for i in range(120):
        _req(session, "GET", "/api/products", attacker_ip=ip)
        time.sleep(0.1)
    print("  Done: 120 rapid requests")


def scenario_path_traversal(session):
    """Simulate path traversal probing."""
    print("\n[SCENARIO] Path Traversal Probing")
    ip = f"{ATTACK_IP_BASE}{random.randint(200,250)}"
    paths = [
        "/../../../etc/passwd", "/.env",
        "/config/database.yml", "/.git/config",
        "/%2e%2e%2f%2e%2e%2fetc/passwd", "/../../windows/win.ini",
        "/admin/../../../etc/shadow",
    ]
    for path in paths * 3:
        _req(session, "GET", path, attacker_ip=ip)
        time.sleep(0.4)
    print("  Done: Path traversal probes")


def scenario_sqli(session):
    """Simulate SQL injection-like query patterns (against local demo only)."""
    print("\n[SCENARIO] SQL Injection Probe Simulation")
    ip = f"{ATTACK_IP_BASE}{random.randint(10,30)}"
    sqli_queries = [
        "id=1' OR '1'='1",
        "id=1 UNION SELECT * FROM users",
        "search='; DROP TABLE users;--",
        "q=admin'--",
        "id=1 AND SLEEP(5)--",
    ]
    for q in sqli_queries * 4:
        _req(session, "GET", "/api/search", params={"q": q}, attacker_ip=ip)
        time.sleep(0.5)
    print("  Done: SQLi probe simulation")


def scenario_xss(session):
    """Simulate XSS-like URL patterns (against local demo only)."""
    print("\n[SCENARIO] XSS Probe Simulation")
    ip = f"{ATTACK_IP_BASE}{random.randint(30,50)}"
    xss_payloads = [
        "<script>alert('xss')</script>",
        "javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
    ]
    for payload in xss_payloads * 4:
        _req(session, "GET", "/api/search", params={"q": payload}, attacker_ip=ip)
        time.sleep(0.4)
    print("  Done: XSS probe simulation")


def scenario_sensitive_scan(session):
    """Simulate systematic scanning of sensitive endpoints."""
    print("\n[SCENARIO] Sensitive Endpoint Scan")
    ip = f"{ATTACK_IP_BASE}{random.randint(150,200)}"
    targets = [
        "/admin", "/admin/", "/wp-admin", "/phpmyadmin",
        "/.env", "/config", "/api/admin", "/api/users",
        "/api/keys", "/debug", "/actuator", "/metrics",
    ]
    for target in targets * 3:
        _req(session, "GET", target, attacker_ip=ip)
        time.sleep(0.3)
    print("  Done: Sensitive endpoint scan")


def scenario_bot_scan(session):
    """Simulate bot scanning of many endpoints."""
    print("\n[SCENARIO] Bot Scan")
    ip = f"{ATTACK_IP_BASE}{random.randint(80,100)}"
    scan_paths = [
        "/robots.txt", "/sitemap.xml", "/backup.sql", "/dump.sql",
        "/data.zip", "/secret", "/private", "/test", "/wp-content",
        "/uploads", "/assets", "/.htaccess", "/web.config",
    ]
    for path in scan_paths * 2:
        _req(session, "GET", path, attacker_ip=ip)
        time.sleep(0.25)
    print("  Done: Bot scan")


def scenario_abnormal_methods(session):
    """Simulate unusual HTTP methods."""
    print("\n[SCENARIO] Abnormal HTTP Methods")
    ip = f"{ATTACK_IP_BASE}{random.randint(250,254)}"
    methods_paths = [
        ("TRACE",    "/"),
        ("OPTIONS",  "/api/products"),
        ("PROPFIND", "/admin"),
        ("MOVE",     "/products"),
    ]
    for _ in range(5):
        for method, path in methods_paths:
            _req(session, method, path, attacker_ip=ip)
            time.sleep(0.5)
    print("  Done: Abnormal HTTP methods")


def scenario_combined(session):
    """Simulate a combined multi-vector attack pattern."""
    print("\n[SCENARIO] Combined Multi-Vector Attack")
    ip = f"{ATTACK_IP_BASE}{random.randint(1,10)}"
    print("  Phase 1: Reconnaissance (scanning)")
    for path in ["/admin", "/.env", "/api/admin", "/api/users"]:
        _req(session, "GET", path, attacker_ip=ip)
        time.sleep(0.3)
    print("  Phase 2: Brute force")
    for i in range(15):
        _req(session, "POST", "/login",
             data={"username": "admin", "password": f"pass{i}"},
             attacker_ip=ip)
        time.sleep(0.2)
    print("  Phase 3: SQLi probe")
    for q in ["id=1' OR '1'='1", "q=admin'--"]:
        _req(session, "GET", "/api/search", params={"q": q}, attacker_ip=ip)
        time.sleep(0.4)
    print("  Done: Combined attack simulation")


SCENARIOS = {
    "brute_force":     scenario_brute_force,
    "rate_abuse":      scenario_rate_abuse,
    "path_traversal":  scenario_path_traversal,
    "sqli":            scenario_sqli,
    "xss":             scenario_xss,
    "sensitive_scan":  scenario_sensitive_scan,
    "bot_scan":        scenario_bot_scan,
    "abnormal_methods":scenario_abnormal_methods,
    "combined":        scenario_combined,
}


def main():
    parser = argparse.ArgumentParser(description="Generate suspicious test traffic (local only)")
    parser.add_argument("--scenario", default="brute_force",
                        choices=list(SCENARIOS.keys()) + ["all"],
                        help="Attack scenario to simulate")
    parser.add_argument("--target", default="http://127.0.0.1:3000",
                        help="Target base URL (default: http://127.0.0.1:3000)")
    parser.add_argument("--list", action="store_true", help="List all scenarios")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.target.rstrip('/')

    if args.list:
        print("Available scenarios:")
        for name in SCENARIOS:
            print(f"  {name}")
        return

    print("=" * 70)
    print("  SUSPICIOUS TRAFFIC SIMULATOR (LOCAL TESTING ONLY)")
    print("=" * 70)
    print(f"  Target   : {BASE_URL}")
    print(f"  Scenario : {args.scenario}")
    print(f"  WARNING  : All traffic directed to LOCAL demo app only!")
    print("=" * 70)

    session = requests.Session()

    if args.scenario == "all":
        for name, fn in SCENARIOS.items():
            fn(session)
            time.sleep(2)
    else:
        SCENARIOS[args.scenario](session)

    print("\n[OK] Test scenario complete.")


if __name__ == "__main__":
    main()
