import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, AreaChart, Area,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { useSecuritySocket } from './hooks/useSocket';
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

  // Build chart data from incoming stats
  useEffect(() => {
    if (!stats) return;
    const ts = new Date().toLocaleTimeString();

    setAnomalyHistory(prev => {
      const last = stats.anomaly_score_history || [];
      const newest = last.slice(-1)[0];
      const point = { time: ts, score: newest?.score || 0, level: newest?.level || 'Low' };
      return [...prev, point].slice(-60);
    });

    setReqHistory(prev => {
      const point = {
        time:    ts,
        total:   stats.total_requests || 0,
        alerts:  stats.total_alerts   || 0,
        blocked: stats.total_blocked  || 0,
        rps:     stats.requests_per_second || 0,
      };
      return [...prev, point].slice(-60);
    });
  }, [stats]);

  const total   = stats?.total_requests   || 0;
  const alerts_ = stats?.total_alerts     || 0;
  const blocked = stats?.total_blocked    || 0;
  const normal  = stats?.total_normal     || 0;
  const rps     = stats?.requests_per_second || 0;
  const latestScore = stats?.anomaly_score_history?.slice(-1)[0]?.score || 0;

  return (
    <div className="app">
      {/* ---- NAVBAR ---- */}
      <nav className="navbar">
        <div className="navbar-brand">
          <span className="brand-icon">🛡</span>
          <span className="brand-name">GTAE-ATRA</span>
          <span className="brand-sub">Web Security Monitor</span>
        </div>
        <div className="navbar-status">
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

        {/* ================ OVERVIEW TAB ================ */}
        {tab === 'overview' && (
          <>
            {/* Stat Cards */}
            <div className="stats-grid">
              <StatCard label="Total Requests"    value={total.toLocaleString()} icon="📊" accent="#60a5fa" />
              <StatCard label="Req / Second"       value={rps.toFixed(1)}         icon="⚡" accent="#a78bfa" />
              <StatCard label="Protected Sites"   value={stats?.site_count || (stats?.monitored_sites?.length || 1)} icon="🌐" accent="#38bdf8"
                        sub={(stats?.monitored_sites || ['Active']).slice(0, 2).join(', ')} />
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
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2a45" />
                    <XAxis dataKey="time" tick={{ fill:'#7878a0', fontSize:10 }} />
                    <YAxis tick={{ fill:'#7878a0', fontSize:10 }} />
                    <Tooltip contentStyle={{ background:'#161628', border:'1px solid #2a2a45', color:'#e8e8f0' }} />
                    <Area type="monotone" dataKey="score" stroke="#a78bfa" fill="url(#scoreGrad)" strokeWidth={2} dot={false} name="Score" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Request rate chart */}
              <div className="card chart-card">
                <h3 className="chart-title">Request Rate (req/s)</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={reqHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2a45" />
                    <XAxis dataKey="time" tick={{ fill:'#7878a0', fontSize:10 }} />
                    <YAxis tick={{ fill:'#7878a0', fontSize:10 }} />
                    <Tooltip contentStyle={{ background:'#161628', border:'1px solid #2a2a45', color:'#e8e8f0' }} />
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
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2a45" />
                  <XAxis dataKey="name" tick={{ fill:'#7878a0', fontSize:11 }} />
                  <YAxis tick={{ fill:'#7878a0', fontSize:11 }} />
                  <Tooltip contentStyle={{ background:'#161628', border:'1px solid #2a2a45', color:'#e8e8f0' }} />
                  <Bar dataKey="count" fill="#60a5fa" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Recent events (last 10) */}
            <div className="card" style={{ marginTop:'1.5rem' }}>
              <h3 className="chart-title" style={{ marginBottom:'1rem' }}>Recent Security Events</h3>
              <div className="alert-table">
                <div className="alert-header">
                  <div>Time</div><div>Source IP</div><div>Site</div><div>Endpoint</div>
                  <div>Score</div><div>Risk</div><div>Action</div>
                </div>
                {alerts.slice(0, 10).map((e, i) => <AlertRow key={i} event={e} index={i} />)}
                {alerts.length === 0 && (
                  <div style={{ textAlign:'center', padding:'2rem', color:'#7878a0' }}>
                    No events yet. Traffic monitoring active...
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
              Security Alerts
              <span className="badge badge-high" style={{ marginLeft:'0.75rem' }}>{alerts.filter(e=>e.is_anomaly).length} alerts</span>
            </h3>
            {alerts.filter(e => e.is_anomaly).map((e, i) => (
              <div key={i} className="alert-detail animate-slide-in" style={{ animationDelay: `${i*0.03}s` }}>
                <div className="alert-detail-header">
                  <div>
                    <span className="mono" style={{ color:'#60a5fa' }}>{e.source_ip}</span>
                    <span style={{ margin:'0 0.5rem', color:'#7878a0' }}>→</span>
                    <span className="mono" style={{ color:'#a78bfa' }}>{e.endpoint}</span>
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
            {alerts.filter(e=>e.is_anomaly).length === 0 && (
              <div style={{ textAlign:'center', padding:'3rem', color:'#7878a0' }}>
                ✅ No security alerts yet
              </div>
            )}
          </div>
        )}

        {/* ================ LOGS TAB ================ */}
        {tab === 'logs' && (
          <div className="card">
            <h3 className="chart-title" style={{ marginBottom:'1rem' }}>All Security Events</h3>
            <div className="alert-table">
              <div className="alert-header">
                <div>Time</div><div>IP</div><div>Endpoint</div>
                <div>Score</div><div>Risk</div><div>Action</div>
              </div>
              {alerts.map((e, i) => <AlertRow key={i} event={e} index={i} />)}
              {alerts.length === 0 && (
                <div style={{ textAlign:'center', padding:'2rem', color:'#7878a0' }}>
                  No events logged yet.
                </div>
              )}
            </div>
          </div>
        )}

        {/* ================ BLOCKED TAB ================ */}
        {tab === 'blocked' && (
          <div className="card">
            <h3 className="chart-title" style={{ marginBottom:'1rem' }}>
              Blocked IPs
              <span className="badge badge-block" style={{ marginLeft:'0.75rem' }}>{stats?.blocked_ips?.length || 0}</span>
            </h3>
            {(stats?.blocked_ips || []).map((ip, i) => (
              <div key={i} className="blocked-row animate-slide-in" style={{ animationDelay: `${i*0.05}s` }}>
                <span className="mono" style={{ color:'#f87171' }}>🚫 {ip}</span>
                <span className="badge badge-block">BLOCKED</span>
              </div>
            ))}
            {(!stats?.blocked_ips || stats.blocked_ips.length === 0) && (
              <div style={{ textAlign:'center', padding:'3rem', color:'#7878a0' }}>
                ✅ No IPs currently blocked
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
