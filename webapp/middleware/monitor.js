/**
 * webapp/middleware/monitor.js
 * PRODUCTION Security Middleware for GTAE-ATRA web security monitoring.
 *
 * Two critical functions:
 *   1. ENFORCEMENT — Checks every incoming IP against the blocklist and
 *      returns HTTP 403 Forbidden if the IP is blocked (request never
 *      reaches your app routes).
 *   2. TELEMETRY — Captures safe metadata for every allowed request and
 *      sends batches to the Python ML monitor server for inference.
 *
 * Does NOT store passwords, tokens, session cookies, or any sensitive payload.
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

// In-memory cache of blocked IPs to avoid hitting the server on every request.
// Synced periodically and updated from telemetry responses.
const blockedIPCache = new Map(); // ip -> { blockedAt: timestamp, expiresAt: timestamp }
const CACHE_TTL_MS = 30_000; // Cache entries for 30 seconds before re-checking

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
      timeout: 1000, // 1 second max — don't slow down requests
    });
    const blocked = res.data?.blocked === true;

    // Cache the result
    blockedIPCache.set(ip, {
      blocked,
      expiresAt: Date.now() + CACHE_TTL_MS,
    });

    return blocked;
  } catch (err) {
    // If monitor is unreachable, fail-open (allow the request)
    if (DEBUG) console.error(`[monitor] Blocklist check failed for ${ip}: ${err.message}`);
    return false;
  }
}

/**
 * Mark an IP as blocked in the local cache (called when telemetry response
 * indicates a BLOCK action).
 * @param {string} ip
 */
function cacheBlock(ip) {
  blockedIPCache.set(ip, {
    blocked: true,
    expiresAt: Date.now() + 60_000, // Cache block for 1 minute
  });
}

// ============================================================
// PER-IP REQUEST BUFFER
// ============================================================

const ipBuffers = new Map();
let flushTimer = null;

/**
 * Get the real client IP, respecting X-Forwarded-For for reverse-proxy setups.
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
 * Flush buffered requests to the Python monitor server.
 * Sends per-IP batches with API key authentication.
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
        timeout:          3000,
        headers:          { 'X-API-Key': API_KEY },
        validateStatus:   () => true,   // don't throw on 4xx/5xx
      });

      // If the monitor says BLOCK this IP, cache it immediately
      if (res.data?.blocked === true) {
        cacheBlock(ip);
        if (DEBUG) console.log(`[monitor] IP ${ip} BLOCKED by GTAE-ATRA engine`);
      }
    } catch (err) {
      // Monitor server offline — silently ignore, don't break the webapp
      if (DEBUG) {
        console.error(`[monitor] Flush failed for ${ip}: ${err.message}`);
      }
    }
  }
}

/**
 * Start the periodic flush background task.
 */
function startPeriodicFlush() {
  if (flushTimer) return;
  flushTimer = setInterval(flushBuffers, FLUSH_INTERVAL_MS);
  flushTimer.unref(); // don't prevent process exit
}

// ============================================================
// EXPRESS MIDDLEWARE
// ============================================================

/**
 * Express middleware factory.
 *
 * Usage:
 *   const { monitor } = require('./middleware/monitor');
 *   app.use(monitor());
 *
 * This middleware does two things:
 *   1. BEFORE route handlers: checks if the IP is blocked → 403
 *   2. AFTER response: captures telemetry metadata → batches to monitor
 */
function monitor() {
  startPeriodicFlush();

  return async function monitorMiddleware(req, res, next) {
    const ip = getClientIP(req);

    // ── ENFORCEMENT: Block known bad IPs ──────────────────
    const blocked = await isIPBlocked(ip);
    if (blocked) {
      if (DEBUG) console.log(`[monitor] BLOCKED request from ${ip} to ${req.url}`);
      return res.status(403).json({
        error: 'Forbidden',
        message: 'Your IP has been blocked by the security system.',
        code: 'GTAE_ATRA_BLOCKED',
      });
    }

    // ── TELEMETRY: Capture request metadata ───────────────
    const startMs = Date.now();

    const onFinish = () => {
      res.removeListener('finish', onFinish);

      const latencyMs = Date.now() - startMs;
      const urlParts  = req.url.split('?');
      const path      = urlParts[0] || '/';
      const query     = urlParts[1] || '';

      // Safe metadata only — no body content, no auth tokens
      const record = {
        timestamp_ms:    startMs,
        method:          req.method,
        path:            path,
        query:           query,
        status:          res.statusCode,
        response_time_ms:latencyMs,
        request_size:    parseInt(req.headers['content-length'] || '0', 10),
        response_size:   parseInt(res.getHeader('content-length') || '0', 10),
        user_agent:      (req.headers['user-agent'] || '').substring(0, 200),
      };

      if (!ipBuffers.has(ip)) {
        ipBuffers.set(ip, []);
      }
      const buf = ipBuffers.get(ip);
      buf.push(record);

      // Flush immediately if buffer is full
      if (buf.length >= MAX_BATCH_SIZE) {
        const full = ipBuffers.get(ip);
        ipBuffers.delete(ip);
        axios.post(TELEMETRY_URL, {
          source_ip: ip,
          requests:  full,
          site_id:   SITE_ID,
        }, {
          timeout: 3000,
          headers: { 'X-API-Key': API_KEY },
          validateStatus: () => true,
        }).then(response => {
          if (response.data?.blocked === true) {
            cacheBlock(ip);
          }
        }).catch(() => {});
      }
    };

    res.on('finish', onFinish);
    next();
  };
}

module.exports = { monitor, flushBuffers, isIPBlocked, cacheBlock };
