import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, AreaChart, Area,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { useSecuritySocket, MONITOR_URL } from './hooks/useSocket';
import './App.css';

// ============================================================
// HELPERS
// ============================================================

function riskBadge(level) {
  const cls = { Low: 'allow', Medium: 'monitor', High: 'alert', Critical: 'critical' }[level] || 'monitor';
  return <span className={`badge badge-${cls}`}>{level || 'Low'}</span>;
}

function actionBadge(action) {
  const cls = {
    ALLOW: 'allow', MONITOR: 'monitor', ALERT: 'alert', BLOCK: 'block',
  }[action] || 'monitor';
  return <span className={`badge badge-${cls}`}>{action || 'ALLOW'}</span>;
}

function formatTime(ts) {
  if (!ts) return '-';
  try {
    return new Date(ts).toLocaleTimeString();
  } catch { return ts; }
}

function AnomalyScore({ score }) {
  const pct = Math.min(score, 100);
  const color = pct < 30 ? '#34d399' : pct < 60 ? '#fbbf24' : pct < 85 ? '#fb923c' : '#f87171';
  return (
    <div className="score-bar">
      <div className="score-label mono">{(score || 0).toFixed(1)}</div>
      <div className="score-track">
        <div className="score-fill" style={{ width: `${Math.min(pct, 100)}%`, background: color }} />
      </div>
    </div>
  );
}

// ============================================================
// STAT CARD
// ============================================================

function StatCard({ label, value, icon, accent, sub }) {
  return (
    <div className="stat-card animate-fade-in-up" style={{ '--accent': accent }}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-body">
        <div className="stat-value" style={{ color: accent }}>{value ?? '—'}</div>
        <div className="stat-label">{label}</div>
        {sub && <div className="stat-sub">{sub}</div>}
      </div>
    </div>
  );
}

// ============================================================
// ALERT ROW
// ============================================================

function AlertRow({ event, index }) {
  return (
    <div
      className={`alert-row animate-slide-in ${event.is_anomaly ? 'alert-row--anomaly' : ''}`}
      style={{ animationDelay: `${index * 0.03}s` }}
    >
      <div className="alert-time mono">{formatTime(event.timestamp)}</div>
      <div className="alert-ip mono">{event.source_ip}</div>
      <div className="alert-site mono" style={{ color: '#38bdf8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {event.site_id || 'default'}
      </div>
      <div className="alert-endpoint mono">{event.endpoint}</div>
      <div className="alert-score"><AnomalyScore score={event.anomaly_score} /></div>
      <div className="alert-risk">{riskBadge(event.risk_level)}</div>
      <div className="alert-action">{actionBadge(event.action)}</div>
    </div>
  );
}

// ============================================================
// MAIN APP
// ============================================================

export default function App() {
  const { stats, alerts, connected } = useSecuritySocket();
  const [tab, setTab] = useState('overview');
  const [anomalyHistory, setAnomalyHistory] = useState([]);
  const [reqHistory, setReqHistory] = useState([]);
  const [copied, setCopied] = useState(false);
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('gtae_theme') || 'dark';
    } catch {
      return 'dark';
    }
  });

  useEffect(() => {
    try {
      document.documentElement.setAttribute('data-theme', theme);
      localStorage.setItem('gtae_theme', theme);
    } catch (e) {}
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  // Read initial site from URL: ?site=ai-research-paper-explainer
  const [selectedSite, setSelectedSite] = useState(() => {
    try {
      const p = new URLSearchParams(window.location.search).get('site');
      return p ? p.trim() : 'all';
    } catch {
      return 'all';
    }
  });

  // Calculate available sites dynamically
  const availableSites = React.useMemo(() => {
    const set = new Set(stats?.monitored_sites || []);
    alerts.forEach(a => { if (a.site_id) set.add(a.site_id); });
    if (set.size === 0) {
      set.add('ai-research-paper-explainer');
      set.add('fitfuel-store');
    }
    return Array.from(set).filter(Boolean);
  }, [stats, alerts]);

  // Handle site selection change & update URL
  const handleSiteChange = (site) => {
    setSelectedSite(site);
    try {
      const url = new URL(window.location);
      if (site === 'all') {
        url.searchParams.delete('site');
      } else {
        url.searchParams.set('site', site);
      }
      window.history.replaceState(null, '', url.toString());
    } catch (e) {
      console.warn('Could not update history state:', e);
    }
  };

  // Copy shareable link
  const copySiteLink = () => {
    try {
      const url = new URL(window.location.href);
      if (selectedSite !== 'all') {
        url.searchParams.set('site', selectedSite);
      }
      navigator.clipboard.writeText(url.toString());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // fallback
    }
  };

  // Filter events for the selected site
  const filteredAlerts = React.useMemo(() => {
    if (selectedSite === 'all') return alerts;
    return alerts.filter(e => (e.site_id || 'default') === selectedSite);
  }, [alerts, selectedSite]);

  // Filter blocked IPs for the selected site
  const filteredBlockedIPs = React.useMemo(() => {
    if (selectedSite === 'all') return stats?.blocked_ips || [];
    const ips = new Set();
    filteredAlerts.forEach(e => {
      if ((e.action === 'BLOCK' || e.action === 'SIMULATED_BLOCK') && e.source_ip) {
        ips.add(e.source_ip);
      }
    });
    return Array.from(ips);
  }, [filteredAlerts, selectedSite, stats]);

  // Unblock handlers
  const handleUnblock = async (ip) => {
    try {
      await fetch(`${MONITOR_URL}/api/blocklist/unblock?ip=${encodeURIComponent(ip)}`, { method: 'POST' });
    } catch (err) {
      console.error('Failed to unblock IP:', err);
    }
  };

  const handleClearAllBlocks = async () => {
    try {
      await fetch(`${MONITOR_URL}/api/blocklist/clear`, { method: 'POST' });
    } catch (err) {
      console.error('Failed to clear blocklist:', err);
    }
  };

  // Filtered metrics
  const total = selectedSite === 'all' ? (stats?.total_requests || 0) : filteredAlerts.length;
  const alerts_ = selectedSite === 'all'
    ? (stats?.total_alerts || 0)
    : filteredAlerts.filter(e => e.is_anomaly).length;
  const blocked = selectedSite === 'all'
    ? (stats?.total_blocked || 0)
    : filteredBlockedIPs.length;
  const normal = selectedSite === 'all'
    ? (stats?.total_normal || 0)
    : filteredAlerts.filter(e => !e.is_anomaly).length;
  const rps = stats?.requests_per_second || 0;
  
  const siteAnomalyPoints = selectedSite === 'all'
    ? (stats?.anomaly_score_history || [])
    : (stats?.anomaly_score_history || []).filter(pt => !pt.site_id || pt.site_id === selectedSite);
  const latestScore = siteAnomalyPoints.slice(-1)[0]?.score || (filteredAlerts[0]?.anomaly_score || 0);

  // Build chart data
  useEffect(() => {
    if (!stats) return;
    const ts = new Date().toLocaleTimeString();

    setAnomalyHistory(prev => {
      const point = { time: ts, score: latestScore || 0, level: latestScore > 70 ? 'Critical' : latestScore > 40 ? 'High' : 'Low' };
      return [...prev, point].slice(-60);
    });

    setReqHistory(prev => {
      const point = {
        time: ts,
        total: total || 0,
        alerts: alerts_ || 0,
        blocked: blocked || 0,
        rps: rps || 0,
      };
      return [...prev, point].slice(-60);
    });
  }, [stats, total, alerts_, blocked, rps, latestScore]);

  const isLight = theme === 'light';
  const gridStroke = isLight ? '#e2e8f0' : '#232542';
  const axisColor = isLight ? '#64748b' : '#8286a6';
  const tooltipStyle = isLight
    ? { background: '#ffffff', border: '1px solid #e2e8f0', color: '#0f172a', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }
    : { background: '#131428', border: '1px solid #232542', color: '#f1f5f9', borderRadius: '8px', boxShadow: '0 4px 20px rgba(0,0,0,0.5)' };

  return (
    <div className="app">
      {/* ---- NAVBAR ---- */}
      <nav className="navbar">
        <div className="navbar-brand">
          <span className="brand-icon">🛡</span>
          <span className="brand-name">GTAE-ATRA</span>
          <span className="brand-sub">Web Security Monitor</span>
        </div>

        {/* Multi-Tenant Site Selector */}
        <div className="site-selector-wrapper">
          <span className="site-selector-label">🌐 Site:</span>
          <select
            className="site-selector-select"
            value={selectedSite}
            onChange={(e) => handleSiteChange(e.target.value)}
          >
            <option value="all">All Protected Sites ({availableSites.length})</option>
            {availableSites.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          {selectedSite !== 'all' && (
            <button className="site-share-btn" onClick={copySiteLink} title="Copy shareable link for this site">
              {copied ? '✓ Copied!' : '🔗 Share Link'}
            </button>
          )}
        </div>

        <div className="navbar-status">
          <button
            className="theme-toggle-btn"
            onClick={toggleTheme}
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
            aria-label="Toggle Theme"
          >
            <span className="theme-toggle-icon">{theme === 'dark' ? '☀️' : '🌙'}</span>
            <span className="theme-toggle-text">{theme === 'dark' ? 'Light' : 'Dark'}</span>
          </button>
          <span className={`status-dot ${connected ? 'active' : 'offline'}`} />
          <span className="status-text">{connected ? 'Live' : 'Offline'}</span>
          <span className="status-time">{new Date().toLocaleTimeString()}</span>
        </div>
      </nav>

      {/* ---- TABS ---- */}
      <div className="tab-bar">
        {['overview', 'alerts', 'logs', 'blocked'].map(t => (
          <button
            key={t}
            className={`tab-btn ${tab === t ? 'tab-btn--active' : ''}`}
            onClick={() => setTab(t)}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      <div className="main-content">
        {/* Site Filter Notification Banner */}
        {selectedSite !== 'all' && (
          <div className="site-filter-banner">
            <div>
              Viewing private security logs for website: <b>{selectedSite}</b>
              <span style={{ color: '#94a3b8', marginLeft: '0.5rem', fontSize: '0.8rem' }}>
                (Only telemetry and threats targeting this site are shown)
              </span>
            </div>
            <button className="site-filter-reset-btn" onClick={() => handleSiteChange('all')}>
              Show All Sites
            </button>
          </div>
        )}

        {/* ================ OVERVIEW TAB ================ */}
        {tab === 'overview' && (
          <>
            {/* Stat Cards */}
            <div className="stats-grid">
              <StatCard label="Total Requests"    value={total.toLocaleString()} icon="📊" accent="#60a5fa" />
              <StatCard label="Req / Second"       value={rps.toFixed(1)}         icon="⚡" accent="#a78bfa" />
              <StatCard label={selectedSite === 'all' ? "Protected Sites" : "Monitored Target"}
                        value={selectedSite === 'all' ? (stats?.site_count || availableSites.length) : selectedSite}
                        icon="🌐" accent="#38bdf8"
                        sub={selectedSite === 'all' ? availableSites.slice(0, 2).join(', ') : 'Dedicated Mode'} />
              <StatCard label="Security Alerts"   value={alerts_.toLocaleString()} icon="🚨" accent="#fb923c" />
              <StatCard label="Blocked IPs"       value={blocked.toLocaleString()} icon="🚫" accent="#f87171" />
              <StatCard label="Normal Requests"   value={normal.toLocaleString()}  icon="✅" accent="#34d399" />
              <StatCard label="Current Score"     value={latestScore.toFixed(1)}   icon="🎯" accent="#fbbf24"
                        sub={latestScore < 30 ? 'Normal' : latestScore < 60 ? 'Suspicious' : 'High Risk'} />
            </div>

            {/* Charts row */}
            <div className="charts-grid">
              {/* Anomaly score chart */}
              <div className="card chart-card">
                <h3 className="chart-title">Anomaly Score Over Time</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={anomalyHistory}>
                    <defs>
                      <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="#a78bfa" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#a78bfa" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                    <XAxis dataKey="time" tick={{ fill: axisColor, fontSize:10 }} />
                    <YAxis tick={{ fill: axisColor, fontSize:10 }} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Area type="monotone" dataKey="score" stroke="#a78bfa" fill="url(#scoreGrad)" strokeWidth={2} dot={false} name="Score" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Request rate chart */}
              <div className="card chart-card">
                <h3 className="chart-title">Request Rate (req/s)</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={reqHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                    <XAxis dataKey="time" tick={{ fill: axisColor, fontSize:10 }} />
                    <YAxis tick={{ fill: axisColor, fontSize:10 }} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Line type="monotone" dataKey="rps"  stroke="#60a5fa" strokeWidth={2} dot={false} name="Req/s" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Alert distribution */}
            <div className="card" style={{ marginTop:'1.5rem' }}>
              <h3 className="chart-title" style={{ marginBottom:'1rem' }}>Alert Distribution</h3>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={[
                  { name:'Normal',   count: normal },
                  { name:'Monitor',  count: Math.max(0, alerts_ - blocked) },
                  { name:'Blocked',  count: blocked },
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                  <XAxis dataKey="name" tick={{ fill: axisColor, fontSize:11 }} />
                  <YAxis tick={{ fill: axisColor, fontSize:11 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="count" fill="#60a5fa" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Recent events (last 10) */}
            <div className="card" style={{ marginTop:'1.5rem' }}>
              <h3 className="chart-title" style={{ marginBottom:'1rem' }}>
                Recent Security Events {selectedSite !== 'all' && `(${selectedSite})`}
              </h3>
              <div className="alert-table">
                <div className="alert-header">
                  <div>Time</div><div>Source IP</div><div>Site</div><div>Endpoint</div>
                  <div>Score</div><div>Risk</div><div>Action</div>
                </div>
                {filteredAlerts.slice(0, 10).map((e, i) => <AlertRow key={i} event={e} index={i} />)}
                {filteredAlerts.length === 0 && (
                  <div style={{ textAlign:'center', padding:'2rem', color:'#7878a0' }}>
                    No events recorded for this website yet. Monitoring active...
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* ================ ALERTS TAB ================ */}
        {tab === 'alerts' && (
          <div className="card">
            <h3 className="chart-title" style={{ marginBottom:'1rem' }}>
              Security Alerts {selectedSite !== 'all' && `(${selectedSite})`}
              <span className="badge badge-high" style={{ marginLeft:'0.75rem' }}>
                {filteredAlerts.filter(e=>e.is_anomaly).length} alerts
              </span>
            </h3>
            {filteredAlerts.filter(e => e.is_anomaly).map((e, i) => (
              <div key={i} className="alert-detail animate-slide-in" style={{ animationDelay: `${i*0.03}s` }}>
                <div className="alert-detail-header">
                  <div>
                    <span className="mono" style={{ color:'#60a5fa' }}>{e.source_ip}</span>
                    <span style={{ margin:'0 0.5rem', color:'#7878a0' }}>→</span>
                    <span className="mono" style={{ color:'#a78bfa' }}>{e.endpoint}</span>
                    <span className="badge badge-allow" style={{ marginLeft:'0.75rem' }}>{e.site_id || 'default'}</span>
                  </div>
                  <div style={{ display:'flex', gap:'0.5rem', alignItems:'center' }}>
                    {riskBadge(e.risk_level)}
                    {actionBadge(e.action)}
                    <span className="mono" style={{ color:'#7878a0', fontSize:'0.8rem' }}>{formatTime(e.timestamp)}</span>
                  </div>
                </div>
                <div className="alert-detail-body">
                  <div className="alert-reason">
                    <span style={{ color:'#7878a0' }}>Reason: </span>
                    <span>{e.reason}</span>
                  </div>
                  <div className="alert-metrics">
                    <span>Score: <b style={{ color:'#fbbf24' }}>{(e.anomaly_score||0).toFixed(1)}</b></span>
                    <span>Risk: <b style={{ color:'#fb923c' }}>{(e.risk_score||0).toFixed(1)}/100</b></span>
                    <span>Type: <b style={{ color:'#a78bfa' }}>{e.attack_type || 'Unknown'}</b></span>
                    <span>Reqs: <b style={{ color:'#60a5fa' }}>{e.request_count}</b></span>
                    <span>ML ms: <b style={{ color:'#34d399' }}>{e.timing?.total_ms?.toFixed(1) || '-'}</b></span>
                  </div>
                </div>
              </div>
            ))}
            {filteredAlerts.filter(e=>e.is_anomaly).length === 0 && (
              <div style={{ textAlign:'center', padding:'3rem', color:'#7878a0' }}>
                ✅ No security alerts for {selectedSite === 'all' ? 'any site' : selectedSite}
              </div>
            )}
          </div>
        )}

        {/* ================ LOGS TAB ================ */}
        {tab === 'logs' && (
          <div className="card">
            <h3 className="chart-title" style={{ marginBottom:'1rem' }}>
              All Security Events {selectedSite !== 'all' && `(${selectedSite})`}
            </h3>
            <div className="alert-table">
              <div className="alert-header">
                <div>Time</div><div>Source IP</div><div>Site</div><div>Endpoint</div>
                <div>Score</div><div>Risk</div><div>Action</div>
              </div>
              {filteredAlerts.map((e, i) => <AlertRow key={i} event={e} index={i} />)}
              {filteredAlerts.length === 0 && (
                <div style={{ textAlign:'center', padding:'2rem', color:'#7878a0' }}>
                  No events logged for {selectedSite === 'all' ? 'any site' : selectedSite} yet.
                </div>
              )}
            </div>
          </div>
        )}

        {/* ================ BLOCKED TAB ================ */}
        {tab === 'blocked' && (
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
              <h3 className="chart-title" style={{ margin: 0 }}>
                Blocked IPs {selectedSite !== 'all' && `(${selectedSite})`}
                <span className="badge badge-block" style={{ marginLeft:'0.75rem' }}>
                  {filteredBlockedIPs.length}
                </span>
              </h3>
              {filteredBlockedIPs.length > 0 && (
                <button
                  onClick={handleClearAllBlocks}
                  style={{
                    background: 'rgba(239, 68, 68, 0.15)',
                    color: '#f87171',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    padding: '0.45rem 0.9rem',
                    borderRadius: '6px',
                    fontSize: '0.85rem',
                    cursor: 'pointer',
                    fontWeight: 600,
                    transition: 'all 0.2s ease'
                  }}
                  title="Unblock all blocked IPs across the system"
                >
                  🔓 Unblock All IPs
                </button>
              )}
            </div>

            {filteredBlockedIPs.map((ip, i) => (
              <div key={i} className="blocked-row animate-slide-in" style={{ animationDelay: `${i*0.05}s`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="mono" style={{ color:'#f87171' }}>🚫 {ip}</span>
                <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                  <span className="badge badge-block">BLOCKED</span>
                  <button
                    onClick={() => handleUnblock(ip)}
                    style={{
                      background: 'rgba(52, 211, 153, 0.15)',
                      color: '#34d399',
                      border: '1px solid rgba(52, 211, 153, 0.3)',
                      padding: '0.25rem 0.65rem',
                      borderRadius: '4px',
                      fontSize: '0.75rem',
                      cursor: 'pointer',
                      fontWeight: 600
                    }}
                    title={`Unblock IP ${ip}`}
                  >
                    🔓 Unblock
                  </button>
                </div>
              </div>
            ))}
            {filteredBlockedIPs.length === 0 && (
              <div style={{ textAlign:'center', padding:'3rem', color:'#7878a0' }}>
                ✅ No IPs currently blocked on {selectedSite === 'all' ? 'any site' : selectedSite}
              </div>
            )}
            <p style={{ marginTop:'1.5rem', color:'#7878a0', fontSize:'0.85rem' }}>
              🛡️ Active IP blocking is enabled. Blocked IPs receive HTTP 403 Forbidden
              and cannot access any routes on monitored websites.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
