/**
 * webapp/middleware/monitor.js
 * PRODUCTION Security Middleware for GTAE-ATRA web security monitoring.
 *
 * Two critical functions:
 *   1. ENFORCEMENT — Checks every incoming IP against the blocklist and
 *      returns HTTP 403 Forbidden with a high-tech Cyber Security alert screen
 *      if the IP is blocked (request never reaches application routes).
 *   2. TELEMETRY — Captures safe metadata for every allowed request and
 *      sends batches to the Python ML monitor server for inference.
 *
 * Environment Variables:
 *   MONITOR_URL      - Python monitor server URL (default: http://127.0.0.1:8765)
 *   MONITOR_API_KEY  - API key for authenticating with the monitor server
 *   MONITOR_SITE_ID  - Unique site identifier for multi-site monitoring
 *   MONITOR_FLUSH_MS - Telemetry batch flush interval (default: 5000)
 *   MONITOR_BATCH_SIZE - Max batch size before immediate flush (default: 50)
 *   MONITOR_DEBUG    - Set to '1' for verbose logging
 */

const axios = require('axios');

// ============================================================
// CONFIGURATION
// ============================================================

const MONITOR_BASE_URL = process.env.MONITOR_URL || 'http://127.0.0.1:8765';
const TELEMETRY_URL = `${MONITOR_BASE_URL}/telemetry`;
const BLOCKLIST_CHECK_URL = `${MONITOR_BASE_URL}/api/blocklist/check`;
const API_KEY = process.env.MONITOR_API_KEY || '';
const SITE_ID = process.env.MONITOR_SITE_ID || 'default';
const FLUSH_INTERVAL_MS = parseInt(process.env.MONITOR_FLUSH_MS || '5000', 10);
const MAX_BATCH_SIZE = parseInt(process.env.MONITOR_BATCH_SIZE || '50', 10);
const DEBUG = process.env.MONITOR_DEBUG === '1';

// ============================================================
// LOCAL BLOCKLIST CACHE
// ============================================================

const blockedIPCache = new Map(); // ip -> { blocked: boolean, expiresAt: timestamp }
const POSITIVE_CACHE_TTL_MS = 60_000; // Cache blocked state for 60s
const NEGATIVE_CACHE_TTL_MS = 1_000;  // Cache allowed state for only 1s so attacks are blocked promptly

/**
 * Check if an IP is blocked (local cache first, then server).
 * @param {string} ip
 * @returns {Promise<boolean>}
 */
async function isIPBlocked(ip) {
  // Check local cache first
  const cached = blockedIPCache.get(ip);
  if (cached) {
    if (Date.now() < cached.expiresAt) {
      return cached.blocked;
    }
    // Cache expired — remove and re-check
    blockedIPCache.delete(ip);
  }

  // Query the monitor server
  try {
    const res = await axios.get(BLOCKLIST_CHECK_URL, {
      params: { ip },
      timeout: 1200, // Fast response — fail open if server unreachable
    });
    const blocked = res.data?.blocked === true;

    // Cache positive blocks longer than negative allowances
    blockedIPCache.set(ip, {
      blocked,
      expiresAt: Date.now() + (blocked ? POSITIVE_CACHE_TTL_MS : NEGATIVE_CACHE_TTL_MS),
    });

    return blocked;
  } catch (err) {
    if (DEBUG) console.error(`[monitor] Blocklist check failed for ${ip}: ${err.message}`);
    return false;
  }
}

/**
 * Mark an IP as blocked in the local cache immediately.
 * @param {string} ip
 * @param {number} ttlMs
 */
function cacheBlock(ip, ttlMs = POSITIVE_CACHE_TTL_MS) {
  blockedIPCache.set(ip, {
    blocked: true,
    expiresAt: Date.now() + ttlMs,
  });
}

/**
 * Invalidate negative cache for an IP upon suspicious activity.
 * @param {string} ip
 */
function invalidateCache(ip) {
  const cached = blockedIPCache.get(ip);
  if (cached && !cached.blocked) {
    blockedIPCache.delete(ip);
  }
}

// ============================================================
// PER-IP REQUEST BUFFER
// ============================================================

const ipBuffers = new Map();
let flushTimer = null;

/**
 * Get the client IP, prioritizing X-Forwarded-For for proxies and attack simulations.
 * @param {import('express').Request} req
 */
function getClientIP(req) {
  const forwarded = req.headers['x-forwarded-for'];
  if (forwarded) {
    return forwarded.split(',')[0].trim();
  }
  return req.ip || req.socket?.remoteAddress || 'unknown';
}

/**
 * Send a specific IP's batch immediately to the monitor server.
 * @param {string} ip
 */
async function flushIPBuffer(ip) {
  const requests = ipBuffers.get(ip);
  if (!requests || requests.length === 0) return;
  ipBuffers.delete(ip);

  try {
    const res = await axios.post(TELEMETRY_URL, {
      source_ip: ip,
      requests:  requests,
      site_id:   SITE_ID,
    }, {
      timeout: 3000,
      headers: { 'X-API-Key': API_KEY },
      validateStatus: () => true,
    });

    if (res.data?.blocked === true) {
      cacheBlock(ip);
      if (DEBUG) console.log(`[monitor] IP ${ip} immediately BLOCKED by GTAE-ATRA engine`);
    }
  } catch (err) {
    if (DEBUG) console.error(`[monitor] Immediate flush failed for ${ip}: ${err.message}`);
  }
}

/**
 * Flush all buffered requests to the Python monitor server.
 */
async function flushBuffers() {
  if (ipBuffers.size === 0) return;

  const snapshot = new Map(ipBuffers);
  ipBuffers.clear();

  for (const [ip, requests] of snapshot.entries()) {
    if (requests.length === 0) continue;
    try {
      const res = await axios.post(TELEMETRY_URL, {
        source_ip: ip,
        requests:  requests,
        site_id:   SITE_ID,
      }, {
        timeout:        3000,
        headers:        { 'X-API-Key': API_KEY },
        validateStatus: () => true,
      });

      if (res.data?.blocked === true) {
        cacheBlock(ip);
        if (DEBUG) console.log(`[monitor] IP ${ip} BLOCKED by GTAE-ATRA engine`);
      }
    } catch (err) {
      if (DEBUG) console.error(`[monitor] Flush failed for ${ip}: ${err.message}`);
    }
  }
}

/**
 * Start periodic flush background loop.
 */
function startPeriodicFlush() {
  if (flushTimer) return;
  flushTimer = setInterval(flushBuffers, FLUSH_INTERVAL_MS);
  flushTimer.unref();
}

/**
 * Detect high-severity exploit signatures in an active request.
 */
function isHighRiskPattern(req, res) {
  const url = (req.originalUrl || req.url || '').toLowerCase();
  const query = decodeURIComponent(url.split('?')[1] || '');
  const path = url.split('?')[0] || '';

  // Sensitive endpoint scan / path traversal probe
  if (
    path.includes('/.env') || path.includes('/.git') || path.includes('/wp-admin') ||
    path.includes('/phpmyadmin') || path.includes('/etc/passwd') || path.includes('..') ||
    path.includes('%2e%2e') || path.includes('/admin/backup')
  ) {
    return true;
  }

  // SQL Injection or XSS patterns
  if (
    /union\s+select|select\s+.*from|drop\s+table|exec\(|<script|javascript:|onerror=|1=1|--/i.test(query)
  ) {
    return true;
  }

  // Repeated 401 Unauthorized (brute force) or 403 Forbidden
  if ((path.includes('/login') && res.statusCode === 401) || res.statusCode === 403) {
    return true;
  }

  return false;
}

// ============================================================
// PROFESSIONAL BLOCKED SCREEN GENERATOR (HTML)
// ============================================================

function renderBlockedPage(ip, path, incidentId) {
  return `<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>403 Forbidden - Access Blocked by GTAE-ATRA Security Engine</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-gradient: radial-gradient(circle at 50% 20%, #16102a 0%, #080912 75%);
      --card-bg: rgba(18, 20, 36, 0.85);
      --card-border: rgba(244, 63, 94, 0.35);
      --text-main: #f1f5f9;
      --text-sub: #94a3b8;
      --badge-bg: rgba(244, 63, 94, 0.12);
      --badge-border: rgba(244, 63, 94, 0.4);
      --badge-color: #fb7185;
      --accent-glow: 0 0 35px rgba(244, 63, 94, 0.25);
      --box-shadow: 0 20px 45px -10px rgba(0, 0, 0, 0.7);
      --info-bg: rgba(255, 255, 255, 0.03);
      --info-border: rgba(255, 255, 255, 0.08);
      --btn-bg: #22253a;
      --btn-hover: #2e3352;
    }

    [data-theme="light"] {
      --bg-gradient: radial-gradient(circle at 50% 20%, #fff1f2 0%, #f8fafc 80%);
      --card-bg: rgba(255, 255, 255, 0.95);
      --card-border: rgba(244, 63, 94, 0.4);
      --text-main: #0f172a;
      --text-sub: #64748b;
      --badge-bg: rgba(244, 63, 94, 0.08);
      --badge-border: rgba(244, 63, 94, 0.3);
      --badge-color: #e11d48;
      --accent-glow: 0 10px 30px rgba(244, 63, 94, 0.15);
      --box-shadow: 0 20px 40px -15px rgba(225, 29, 72, 0.15);
      --info-bg: #f8fafc;
      --info-border: #e2e8f0;
      --btn-bg: #f1f5f9;
      --btn-hover: #e2e8f0;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Inter', system-ui, sans-serif;
      background: var(--bg-gradient);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
      transition: background 0.3s ease, color 0.3s ease;
      overflow-x: hidden;
    }

    .theme-toggle {
      position: absolute;
      top: 1.5rem;
      right: 1.5rem;
      background: var(--card-bg);
      border: 1px solid var(--info-border);
      color: var(--text-main);
      padding: 0.5rem 1rem;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      backdrop-filter: blur(10px);
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      transition: all 0.2s;
    }
    .theme-toggle:hover {
      transform: translateY(-2px);
      border-color: #f43f5e;
    }

    .shield-container {
      position: relative;
      margin-bottom: 1.5rem;
    }
    .pulse-ring {
      position: absolute;
      inset: -12px;
      border-radius: 50%;
      border: 2px solid rgba(244, 63, 94, 0.4);
      animation: pulse-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    }
    @keyframes pulse-ring {
      0% { transform: scale(0.85); opacity: 1; }
      100% { transform: scale(1.4); opacity: 0; }
    }

    .shield-icon {
      width: 84px;
      height: 84px;
      background: linear-gradient(135deg, #f43f5e, #be123c);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 40px rgba(244, 63, 94, 0.45);
      position: relative;
      z-index: 2;
    }
    .shield-icon svg {
      width: 44px;
      height: 44px;
      fill: #ffffff;
    }

    .block-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      padding: 2.5rem 2.25rem;
      max-width: 620px;
      width: 100%;
      backdrop-filter: blur(16px);
      box-shadow: var(--box-shadow);
      text-align: center;
      position: relative;
      animation: float-in 0.5s ease-out forwards;
    }
    @keyframes float-in {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .badge-blocked {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: var(--badge-bg);
      border: 1px solid var(--badge-border);
      color: var(--badge-color);
      padding: 0.35rem 0.9rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 1.25rem;
    }
    .badge-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #f43f5e;
      animation: blink 1.2s infinite;
    }
    @keyframes blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.3; }
    }

    h1 {
      font-size: 1.85rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin-bottom: 0.6rem;
      line-height: 1.2;
    }
    .subtitle {
      color: var(--text-sub);
      font-size: 0.95rem;
      line-height: 1.6;
      margin-bottom: 1.8rem;
    }

    .info-grid {
      background: var(--info-bg);
      border: 1px solid var(--info-border);
      border-radius: 12px;
      padding: 1.2rem;
      margin-bottom: 1.8rem;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      text-align: left;
    }
    .info-item {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    .info-label {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-sub);
      font-weight: 600;
    }
    .info-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--text-main);
      word-break: break-all;
    }
    .val-threat { color: #f43f5e; }
    .val-active { color: #38bdf8; }

    .cooldown-box {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.9rem 1.25rem;
      background: rgba(244, 63, 94, 0.08);
      border: 1px solid rgba(244, 63, 94, 0.2);
      border-radius: 10px;
      margin-bottom: 1.5rem;
      font-size: 0.85rem;
    }
    .cooldown-timer {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 800;
      color: #f43f5e;
      font-size: 1.1rem;
    }

    .actions {
      display: flex;
      gap: 0.75rem;
      justify-content: center;
      flex-wrap: wrap;
    }
    .btn {
      padding: 0.75rem 1.4rem;
      border-radius: 10px;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--info-border);
      background: var(--btn-bg);
      color: var(--text-main);
      transition: all 0.2s;
      text-decoration: none;
    }
    .btn:hover {
      background: var(--btn-hover);
      transform: translateY(-1px);
    }
    .btn-primary {
      background: linear-gradient(135deg, #f43f5e, #be123c);
      color: #ffffff;
      border: none;
      box-shadow: 0 4px 15px rgba(244, 63, 94, 0.3);
    }
    .btn-primary:hover {
      box-shadow: 0 6px 20px rgba(244, 63, 94, 0.45);
    }

    footer {
      margin-top: 2rem;
      font-size: 0.78rem;
      color: var(--text-sub);
      text-align: center;
    }
    footer span {
      color: #38bdf8;
      font-weight: 600;
    }
  </style>
</head>
<body>

  <button class="theme-toggle" onclick="toggleTheme()" id="themeBtn" aria-label="Toggle Theme">
    <span id="themeIcon">☀️</span>
    <span id="themeLabel">Light Mode</span>
  </button>

  <div class="shield-container">
    <div class="pulse-ring"></div>
    <div class="shield-icon">
      <svg viewBox="0 0 24 24">
        <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/>
      </svg>
    </div>
  </div>

  <div class="block-card">
    <div class="badge-blocked">
      <span class="badge-dot"></span>
      GTAE-ATRA Active Defense &bull; Critical Threat Mitigation
    </div>

    <h1>Access Prohibited (HTTP 403)</h1>
    <p class="subtitle">
      Your request triggered automated intrusion countermeasures. Malicious traffic patterns or abnormal request anomalies were detected from your network address.
    </p>

    <div class="info-grid">
      <div class="info-item">
        <span class="info-label">Client IP Address</span>
        <span class="info-val">${ip}</span>
      </div>
      <div class="info-item">
        <span class="info-label">Incident ID</span>
        <span class="info-val val-threat">${incidentId}</span>
      </div>
      <div class="info-item">
        <span class="info-label">Target Route</span>
        <span class="info-val">${path || '/'}</span>
      </div>
      <div class="info-item">
        <span class="info-label">Defense Engine</span>
        <span class="info-val val-active">GTAE + ATRA NIDS</span>
      </div>
    </div>

    <div class="cooldown-box">
      <span>Dynamic Block Half-Life Cooldown:</span>
      <span class="cooldown-timer" id="countdown">00:59</span>
    </div>

    <div class="actions">
      <button class="btn btn-primary" onclick="window.location.reload()">
        &#x21bb; Retry Authorization
      </button>
      <a href="mailto:security@organization.internal?subject=Unblock%20Request%20${incidentId}" class="btn">
        Submit Unblock Appeal
      </a>
    </div>
  </div>

  <footer>
    Secured by <span>GTAE-ATRA Real-Time Network & Web Intrusion Detection System</span>
  </footer>

  <script>
    function toggleTheme() {
      const html = document.documentElement;
      const current = html.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);
      localStorage.setItem('gtae_theme', next);
      updateThemeUI(next);
    }

    function updateThemeUI(t) {
      const icon = document.getElementById('themeIcon');
      const label = document.getElementById('themeLabel');
      if (t === 'light') {
        icon.textContent = '🌙';
        label.textContent = 'Dark Mode';
      } else {
        icon.textContent = '☀️';
        label.textContent = 'Light Mode';
      }
    }

    const saved = localStorage.getItem('gtae_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    updateThemeUI(saved);

    // Live countdown
    let secondsLeft = 60;
    const countEl = document.getElementById('countdown');
    const timer = setInterval(() => {
      secondsLeft--;
      if (secondsLeft <= 0) {
        clearInterval(timer);
        countEl.textContent = 'EXPIRED';
        countEl.style.color = '#38bdf8';
      } else {
        const m = Math.floor(secondsLeft / 60).toString().padStart(2, '0');
        const s = (secondsLeft % 60).toString().padStart(2, '0');
        countEl.textContent = m + ':' + s;
      }
    }, 1000);
  </script>
</body>
</html>`;
}

// ============================================================
// EXPRESS MIDDLEWARE FACTORY
// ============================================================

/**
 * Express middleware factory for GTAE-ATRA Web Security.
 *
 * Usage:
 *   const { monitor } = require('./middleware/monitor');
 *   app.use(monitor());
 */
function monitor() {
  startPeriodicFlush();

  return async function monitorMiddleware(req, res, next) {
    const ip = getClientIP(req);

    // ── ENFORCEMENT: Check block status BEFORE processing routes ──
    const blocked = await isIPBlocked(ip);
    if (blocked) {
      if (DEBUG) console.log(`[monitor] BLOCKED request from ${ip} to ${req.url}`);
      res.setHeader('X-GTAE-ATRA-Blocked', '1');
      res.setHeader('X-GTAE-ATRA-Engine', 'GraphTransformer-ATRA');

      const acceptsHtml = req.headers.accept && req.headers.accept.includes('text/html');
      if (acceptsHtml && !req.url.startsWith('/api/')) {
        const incidentId = 'GTAE-' + Math.random().toString(36).substring(2, 8).toUpperCase();
        return res.status(403).send(renderBlockedPage(ip, req.url, incidentId));
      }

      return res.status(403).json({
        error:     'Forbidden',
        message:   'Your IP has been blocked by the GTAE-ATRA security system.',
        code:      'GTAE_ATRA_BLOCKED',
        ip:        ip,
        timestamp: new Date().toISOString(),
      });
    }

    // ── TELEMETRY: Record safe request metrics on response completion ──
    const startMs = Date.now();

    const onFinish = () => {
      res.removeListener('finish', onFinish);

      const latencyMs = Date.now() - startMs;
      const urlParts  = req.url.split('?');
      const path      = urlParts[0] || '/';
      const query     = urlParts[1] || '';

      const record = {
        timestamp_ms:     startMs,
        method:           req.method,
        path:             path,
        query:            query,
        status:           res.statusCode,
        status_code:      res.statusCode,
        response_time_ms: latencyMs,
        request_size:     parseInt(req.headers['content-length'] || '0', 10),
        response_size:    parseInt(res.getHeader('content-length') || '0', 10),
        user_agent:       (req.headers['user-agent'] || '').substring(0, 200),
      };

      if (!ipBuffers.has(ip)) {
        ipBuffers.set(ip, []);
      }
      const buf = ipBuffers.get(ip);
      buf.push(record);

      // Check if this request represents an active exploit attempt
      const suspicious = isHighRiskPattern(req, res);
      if (suspicious) {
        invalidateCache(ip);
        // Immediately flush this IP to trigger rapid real-time ML inference
        flushIPBuffer(ip);
      } else if (buf.length >= MAX_BATCH_SIZE) {
        flushIPBuffer(ip);
      }
    };

    res.on('finish', onFinish);
    next();
  };
}

module.exports = {
  monitor,
  flushBuffers,
  flushIPBuffer,
  isIPBlocked,
  cacheBlock,
  renderBlockedPage,
};
