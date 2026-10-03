# IDS-Project

An advanced Intrusion Detection System utilizing Graph Transformers and Adaptive Threat Response.

## Installation
`pip install -r requirements.txt`

## Usage
- **Full Pipeline (7 Steps):** `python main.py`
- **Real-Time Flow Streaming Demo:** `python demo_realtime.py --speed 5 --burst`
- **Model Training:** `python train.py`
- **Evaluation & Report Generation:** `python evaluate.py`
- **Live Interface Packet Sniffing:** `python realtime.py`

### Authorized Web Request Monitoring Demo

To monitor visitors to a website you control, run the separate demo site on the
laptop. It records requests made to that site and applies transparent behavior
rules with ATRA risk scoring. It does not monitor visitors to unrelated public
websites, and its response decisions do not block requests.

```powershell
python monitored_site.py --host 0.0.0.0 --port 8080
```

From a phone on the same private Wi-Fi, open `http://<laptop-wifi-ip>:8080`.
In an Administrator PowerShell, allow inbound TCP port 8080 on the Private
network profile if Windows Firewall blocks the connection:

```powershell
New-NetFirewallRule -DisplayName "IDS Monitored Site 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow -Profile Private
```

The terminal prints suspicious requests, and structured events are saved to
`logs/web_request_events.csv`.

This web-request demo uses HTTP behavior rules; it does not feed web requests
into the CIC-IDS2017 flow model, whose feature schema is different. Begin with
authorized, low-rate tests and treat ATRA outcomes as logged decisions, not
firewall enforcement.

### Live Network Capture (Windows)

Install [Npcap](https://npcap.com/) before using live capture. Start PowerShell
as Administrator if Windows denies packet capture permissions. List available
adapters first:

```powershell
python realtime.py --list-interfaces
```

Capture the selected adapter in ten-second windows:

```powershell
python realtime.py --interface "<interface name>" --duration 10
```

Use `--no-response` to record detections without requesting ATRA response
actions:

```powershell
python realtime.py --interface "<interface name>" --duration 10 --no-response
```

Captured results are written to `logs/realtime_stream.csv`. This first live
adapter preserves the current 400-value model tensor contract, but the model
must be retrained on these live-derived features before its predictions are
treated as production-grade. The default capture scope is traffic visible to
the local Windows host. Monitoring other LAN devices requires a switch
mirror/SPAN port or another authorized network capture point.

### Real-Time Streaming Demo Options:
```powershell
python demo_realtime.py --speed 5 --burst    # 5 flows/sec with attack burst escalation
python demo_realtime.py --speed 10           # 10 flows/sec standard stream
python demo_realtime.py --speed 0 --count 50 # Benchmark maximum inference throughput
```

## Testing
`pytest tests/`

