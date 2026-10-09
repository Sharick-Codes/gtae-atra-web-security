"""
scripts/cloud_live_test.py
Live Cloud Attack Simulation & Verification Script for GTAE-ATRA Web Security.
Sends live traffic scenarios directly to the production Render cloud backend:
https://gtae-atra-security.onrender.com
"""

import time
import json
import urllib.request
import urllib.error

RENDER_URL = "https://gtae-atra-security.onrender.com"
TELEMETRY_URL = f"{RENDER_URL}/telemetry"
STATS_URL = f"{RENDER_URL}/api/security/stats"
BLOCKLIST_URL = f"{RENDER_URL}/api/blocklist/list"

def send_telemetry(source_ip: str, requests_batch: list, site_id: str = "ai-research-paper-explainer"):
    payload = {
        "source_ip": source_ip,
        "site_id": site_id,
        "requests": requests_batch
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(TELEMETRY_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            return json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  [!] HTTP Error: {e.code} - {e.read().decode('utf-8')}")
        return None
    except Exception as e:
        print(f"  [!] Request Error: {e}")
        return None

def fetch_stats():
    try:
        with urllib.request.urlopen(STATS_URL, timeout=10) as res:
            return json.loads(res.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def fetch_blocklist():
    try:
        with urllib.request.urlopen(BLOCKLIST_URL, timeout=10) as res:
            return json.loads(res.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 70)
    print("  [*] GTAE-ATRA CLOUD LIVE SECURITY ATTACK SIMULATION")
    print(f"  Target Cloud Server: {RENDER_URL}")
    print("=" * 70)

    # 1. Benign Traffic Scenario
    print("\n[Phase 1] [NORMAL] Sending Normal Benign Traffic (Regular User browsing)...")
    now_ms = time.time() * 1000
    benign_ip = "122.164.88.10"  # Resident IP
    benign_batch = [
        {
            "method": "GET",
            "endpoint": "/index.html",
            "status_code": 200,
            "response_time_ms": 32.0,
            "request_size": 250,
            "response_size": 4200,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            "timestamp_ms": now_ms - 2000
        },
        {
            "method": "GET",
            "endpoint": "/api/docs",
            "status_code": 200,
            "response_time_ms": 48.0,
            "request_size": 180,
            "response_size": 12500,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            "timestamp_ms": now_ms - 1500
        },
        {
            "method": "POST",
            "endpoint": "/api/explain",
            "status_code": 200,
            "response_time_ms": 1400.0,
            "request_size": 3200,
            "response_size": 8900,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            "timestamp_ms": now_ms - 800
        },
        {
            "method": "GET",
            "endpoint": "/api/history",
            "status_code": 200,
            "response_time_ms": 55.0,
            "request_size": 310,
            "response_size": 1800,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            "timestamp_ms": now_ms
        }
    ]
    resp = send_telemetry(benign_ip, benign_batch)
    if resp and resp.get("event"):
        ev = resp["event"]
        print(f"  Result: Action={ev['action']} | Risk={ev['risk_level']} (Score={ev['risk_score']}) | Anomaly={ev['is_anomaly']}")
    else:
        print(f"  Result: {resp}")

    time.sleep(2)

    # 2. SQL Injection Attack Scenario
    print("\n[Phase 2] [ATTACK] Launching SQL Injection Attack (sqlmap tool probing)...")
    sqli_ip = "198.51.100.44"
    sqli_batch = [
        {
            "method": "POST",
            "endpoint": "/api/login",
            "status_code": 401,
            "response_time_ms": 110.0,
            "request_size": 850,
            "response_size": 220,
            "user_agent": "sqlmap/1.7#stable (https://sqlmap.org)",
            "payload": "' OR 1=1 --",
            "timestamp_ms": now_ms - 1200
        },
        {
            "method": "POST",
            "endpoint": "/api/login",
            "status_code": 401,
            "response_time_ms": 95.0,
            "request_size": 910,
            "response_size": 220,
            "user_agent": "sqlmap/1.7#stable (https://sqlmap.org)",
            "payload": "' UNION SELECT null, username, password FROM users --",
            "timestamp_ms": now_ms - 800
        },
        {
            "method": "POST",
            "endpoint": "/api/users",
            "status_code": 500,
            "response_time_ms": 140.0,
            "request_size": 1100,
            "response_size": 512,
            "user_agent": "sqlmap/1.7#stable (https://sqlmap.org)",
            "payload": "'; DROP TABLE sessions; --",
            "timestamp_ms": now_ms - 400
        },
        {
            "method": "POST",
            "endpoint": "/api/login",
            "status_code": 401,
            "response_time_ms": 105.0,
            "request_size": 960,
            "response_size": 220,
            "user_agent": "sqlmap/1.7#stable (https://sqlmap.org)",
            "payload": "admin' AND SLEEP(5)--",
            "timestamp_ms": now_ms
        }
    ]
    resp = send_telemetry(sqli_ip, sqli_batch)
    if resp and resp.get("event"):
        ev = resp["event"]
        print(f"  Result: Action={ev['action']} | Risk={ev['risk_level']} (Score={ev['risk_score']}) | Attack={ev['attack_type']} | Blocked={resp.get('blocked')}")
    else:
        print(f"  Result: {resp}")

    time.sleep(2)

    # 3. Sensitive Endpoint Reconnaissance Attack
    print("\n[Phase 3] [ATTACK] Probing Sensitive Endpoints & Path Traversal (.env, wp-admin, /etc/passwd)...")
    recon_ip = "203.0.113.89"
    recon_batch = [
        {
            "method": "GET",
            "endpoint": "/.env",
            "status_code": 404,
            "response_time_ms": 15.0,
            "request_size": 120,
            "response_size": 180,
            "user_agent": "Nikto/2.1.6",
            "timestamp_ms": now_ms - 1500
        },
        {
            "method": "GET",
            "endpoint": "/.git/config",
            "status_code": 404,
            "response_time_ms": 12.0,
            "request_size": 130,
            "response_size": 180,
            "user_agent": "Nikto/2.1.6",
            "timestamp_ms": now_ms - 1100
        },
        {
            "method": "GET",
            "endpoint": "/wp-login.php",
            "status_code": 404,
            "response_time_ms": 14.0,
            "request_size": 140,
            "response_size": 180,
            "user_agent": "Nikto/2.1.6",
            "timestamp_ms": now_ms - 800
        },
        {
            "method": "GET",
            "endpoint": "/admin/backup.sql",
            "status_code": 403,
            "response_time_ms": 18.0,
            "request_size": 160,
            "response_size": 210,
            "user_agent": "Nikto/2.1.6",
            "timestamp_ms": now_ms - 400
        },
        {
            "method": "GET",
            "endpoint": "/api/v1/../../etc/passwd",
            "status_code": 403,
            "response_time_ms": 22.0,
            "request_size": 180,
            "response_size": 210,
            "user_agent": "Nikto/2.1.6",
            "timestamp_ms": now_ms
        }
    ]
    resp = send_telemetry(recon_ip, recon_batch)
    if resp and resp.get("event"):
        ev = resp["event"]
        print(f"  Result: Action={ev['action']} | Risk={ev['risk_level']} (Score={ev['risk_score']}) | Attack={ev['attack_type']} | Blocked={resp.get('blocked')}")
    else:
        print(f"  Result: {resp}")

    time.sleep(2)

    # 4. Fetching Final Cloud Stats
    print("\n" + "=" * 70)
    print("  [STATS] FETCHING LIVE CLOUD STATS & ACTIVE BLOCKLIST")
    print("=" * 70)
    stats = fetch_stats()
    blocklist = fetch_blocklist()
    print(f"  Total Monitored Requests : {stats.get('total_requests')}")
    print(f"  Total Security Alerts    : {stats.get('total_alerts')}")
    print(f"  Total Blocked Attacks    : {stats.get('total_blocked')}")
    print(f"  Monitored Sites          : {stats.get('monitored_sites')}")
    print(f"  Active Blocked IPs       : {blocklist.get('blocked_ips')}")
    print("=" * 70)
    print("  [SUCCESS] SIMULATION COMPLETE! Check your Vercel Dashboard:")
    print("  >> https://gtae-atra-web-security.vercel.app")
    print("=" * 70)

if __name__ == "__main__":
    main()
