"""
scripts/cloud_realistic_test.py
Sends realistic attack telemetry with correct path and status fields
to test GTAE-ATRA Deep Learning & ATRA Risk Engine in the cloud.
"""

import time
import json
import urllib.request

RENDER_URL = "https://gtae-atra-security.onrender.com"
TELEMETRY_URL = f"{RENDER_URL}/telemetry"

def send(ip, site_id, batch):
    data = json.dumps({"source_ip": ip, "site_id": site_id, "requests": batch}).encode("utf-8")
    req = urllib.request.Request(TELEMETRY_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as res:
        return json.loads(res.read().decode("utf-8"))

now_ms = time.time() * 1000
mobile_ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"

# Scenario A: Hacker probing sensitive backend paths repeatedly
print("Sending Sensitive Path Reconnaissance (/.env, /admin)...")
recon_ip = "182.73.224.50"
recon_batch = [
    {
        "method": "GET",
        "path": "/.env",
        "endpoint": "/.env",
        "status": 404,
        "status_code": 404,
        "response_time_ms": 25.0,
        "request_size": 150,
        "response_size": 180,
        "user_agent": mobile_ua,
        "timestamp_ms": now_ms - 1000
    },
    {
        "method": "GET",
        "path": "/.env.local",
        "endpoint": "/.env.local",
        "status": 404,
        "status_code": 404,
        "response_time_ms": 20.0,
        "request_size": 160,
        "response_size": 180,
        "user_agent": mobile_ua,
        "timestamp_ms": now_ms - 800
    },
    {
        "method": "GET",
        "path": "/admin/config",
        "endpoint": "/admin/config",
        "status": 403,
        "status_code": 403,
        "response_time_ms": 30.0,
        "request_size": 200,
        "response_size": 220,
        "user_agent": mobile_ua,
        "timestamp_ms": now_ms - 600
    },
    {
        "method": "GET",
        "path": "/.git/config",
        "endpoint": "/.git/config",
        "status": 404,
        "status_code": 404,
        "response_time_ms": 22.0,
        "request_size": 170,
        "response_size": 180,
        "user_agent": mobile_ua,
        "timestamp_ms": now_ms - 400
    },
    {
        "method": "GET",
        "path": "/api/keys",
        "endpoint": "/api/keys",
        "status": 403,
        "status_code": 403,
        "response_time_ms": 28.0,
        "request_size": 210,
        "response_size": 220,
        "user_agent": mobile_ua,
        "timestamp_ms": now_ms
    }
]

resp = send(recon_ip, "ai-research-paper-explainer", recon_batch)
print("Recon Response:\n", json.dumps(resp, indent=2))

# Scenario B: High-rate Credential Stuffing / Brute Force
print("\nSending Credential Stuffing / Brute Force Burst...")
brute_ip = "103.21.244.15"
brute_batch = [
    {
        "method": "POST",
        "path": "/api/login",
        "endpoint": "/api/login",
        "status": 401,
        "status_code": 401,
        "response_time_ms": 40.0,
        "request_size": 450,
        "response_size": 180,
        "user_agent": mobile_ua,
        "timestamp_ms": now_ms - (i * 80)
    }
    for i in range(8)
]

resp2 = send(brute_ip, "fitfuel-store", brute_batch)
print("Brute Force Response:\n", json.dumps(resp2, indent=2))
