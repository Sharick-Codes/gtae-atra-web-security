import { useState, useEffect, useRef, useCallback } from 'react';
import io from 'socket.io-client';

const MONITOR_URL = process.env.REACT_APP_MONITOR_URL || 'http://127.0.0.1:8765';

export function useSecuritySocket() {
  const [stats, setStats]   = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef(null);

  useEffect(() => {
    const socket = io(MONITOR_URL, {
      transports: ['websocket', 'polling'],
      reconnectionAttempts: 10,
      reconnectionDelay:    2000,
    });
    socketRef.current = socket;

    socket.on('connect',    () => setConnected(true));
    socket.on('disconnect', () => setConnected(false));

    socket.on('stats_update', (data) => {
      setStats(data);
    });

    socket.on('security_alert', (event) => {
      setAlerts(prev => [event, ...prev].slice(0, 100));
    });

    return () => { socket.disconnect(); };
  }, []);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${MONITOR_URL}/api/security/stats`);
      if (res.ok) { const data = await res.json(); setStats(data); }
    } catch { /* server offline */ }
  }, []);

  const fetchLogs = useCallback(async () => {
    try {
      const res = await fetch(`${MONITOR_URL}/api/logs`);
      if (res.ok) {
        const data = await res.json();
        setAlerts(data.events || []);
      }
    } catch { /* server offline */ }
  }, []);

  // Poll every 10s as fallback when socket isn't delivering
  useEffect(() => {
    fetchStats();
    fetchLogs();
    const timer = setInterval(() => { fetchStats(); }, 10000);
    return () => clearInterval(timer);
  }, [fetchStats, fetchLogs]);

  return { stats, alerts, connected };
}
