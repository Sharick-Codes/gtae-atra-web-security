import argparse
import src.realtime_monitor as realtime_monitor
from src.utils import get_logger

logger = get_logger(__name__)

def main():
    """Real-time monitoring entry point."""
    parser = argparse.ArgumentParser(description="Real-time monitoring script.")
    parser.add_argument("--interface", type=str, help="Network interface to monitor")
    parser.add_argument("--list-interfaces", action="store_true", help="List capture interfaces and exit")
    parser.add_argument("--duration", type=int, default=10, help="Seconds per window")
    parser.add_argument("--no-response", action="store_true", help="Disable active responses")
    args = parser.parse_args()

    if args.list_interfaces:
        for interface in realtime_monitor.list_interfaces():
            print(interface)
        return

    engine = realtime_monitor.RTInferenceEngine(no_response=args.no_response)
    print("=== ATRA Real-Time Monitoring ===")
    print(f"Interface: {args.interface or 'Scapy default'}")
    print(f"Capturing in {args.duration}s windows. Press Ctrl+C to stop.\n")

    try:
        while not realtime_monitor.is_stop_requested():
            flows = realtime_monitor.capture_window(args.duration, args.interface)
            engine.process_window(flows)
    except KeyboardInterrupt:
        pass
    print("\nStopped by user. Summary generated.")

if __name__ == "__main__":
    main()
