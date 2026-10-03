"""
tests/traffic/generate_normal_traffic.py
Generates realistic normal user traffic to the demo web application.

Usage:
  python tests/traffic/generate_normal_traffic.py
  python tests/traffic/generate_normal_traffic.py --duration 60 --rate 2

All requests are sent to the LOCAL demo app (http://127.0.0.1:3000).
This script does NOT generate any attack traffic.
"""

import argparse
import random
import time
import sys
import requests
from datetime import datetime

BASE_URL = "http://127.0.0.1:3000"

NORMAL_FLOWS = [
    # (method, path, data)
    ("GET",  "/",               None),
    ("GET",  "/home",           None),
    ("GET",  "/products",       None),
    ("GET",  "/about",          None),
    ("GET",  "/contact",        None),
    ("GET",  "/api/products",   None),
    ("GET",  "/api/categories", None),
    ("GET",  "/api/search?q=widget", None),
    ("POST", "/login",          {"username": "alice", "password": "Password123!"}),
    ("GET",  "/profile",        None),
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) AppleWebKit/605.1.15 Mobile",
]


def send_request(session, method, path, data=None, ip_suffix=None):
    url = f"{BASE_URL}{path}"
    headers = {
        "User-Agent":   random.choice(USER_AGENTS),
        "Accept":       "text/html,application/json,*/*;q=0.9",
        "X-Forwarded-For": f"10.0.1.{ip_suffix or random.randint(1, 50)}",
    }
    try:
        if method == "GET":
            r = session.get(url, headers=headers, timeout=5, allow_redirects=True)
        else:
            r = session.post(url, data=data, headers=headers, timeout=5, allow_redirects=True)
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {method:4} {path:<30} -> {r.status_code}")
        return r.status_code
    except Exception as e:
        print(f"  [ERROR] {method} {path}: {e}")
        return None


def run_normal_traffic(duration: int = 60, rate: float = 2.0, num_users: int = 3):
    """
    Simulate normal user browsing for `duration` seconds at `rate` req/s.
    Uses `num_users` simulated IPs for realistic diversity.
    """
    print("=" * 60)
    print("  NORMAL TRAFFIC GENERATOR")
    print("=" * 60)
    print(f"  Target      : {BASE_URL}")
    print(f"  Duration    : {duration}s")
    print(f"  Rate        : {rate} req/s")
    print(f"  Simulated users: {num_users}")
    print("=" * 60)

    session = requests.Session()
    end_time = time.time() + duration
    req_count = 0
    delay = 1.0 / rate

    while time.time() < end_time:
        # Pick a simulated user IP (1-50 range = normal users)
        user_ip = random.randint(1, 50)

        # Pick a realistic browsing sequence
        method, path, data = random.choice(NORMAL_FLOWS)
        send_request(session, method, path, data, ip_suffix=user_ip)
        req_count += 1

        time.sleep(max(0, delay + random.uniform(-delay * 0.3, delay * 0.3)))

    print(f"\n[OK] Normal traffic complete: {req_count} requests in {duration}s")


def main():
    parser = argparse.ArgumentParser(description="Generate normal web traffic")
    parser.add_argument("--duration", type=int, default=60,  help="Duration in seconds")
    parser.add_argument("--rate",     type=float, default=2.0, help="Requests per second")
    parser.add_argument("--users",    type=int, default=3,    help="Number of simulated users")
    args = parser.parse_args()
    run_normal_traffic(args.duration, args.rate, args.users)


if __name__ == "__main__":
    main()
