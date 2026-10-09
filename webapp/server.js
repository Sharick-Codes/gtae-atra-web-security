/**
 * webapp/server.js
 * Demo Express web application for GTAE-ATRA web security monitoring.
 *
 * Endpoints:
 *   GET  /                   Home page
 *   GET  /home               Home (alias)
 *   GET  /products           Product listing
 *   GET  /about              About page
 *   GET  /contact            Contact page
 *   POST /login              Login (bcrypt hash, no plaintext passwords stored)
 *   GET  /profile            User profile (auth simulated)
 *   GET  /api/products       Products API (JSON)
 *   GET  /api/categories     Categories API (JSON)
 *   GET  /api/search         Search API
 *   GET  /admin              Admin (restricted)
 *
 * All requests are instrumented by the security monitor middleware,
 * which forwards metadata to the Python GTAE-ATRA inference server.
 */

require('dotenv').config();
const express      = require('express');
const cors         = require('cors');
const cookieParser = require('cookie-parser');
const morgan       = require('morgan');
const bcrypt       = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');
const path         = require('path');

const { monitor } = require('./middleware/monitor');

const app  = express();
const PORT = parseInt(process.env.WEBAPP_PORT || '3000', 10);
const HOST = process.env.WEBAPP_HOST || '0.0.0.0';

// ---- Middleware -------------------------------------------------------

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(cookieParser());
app.use(morgan('combined'));

// GTAE-ATRA security monitoring middleware (attach before routes)
app.use(monitor());

// Static assets
app.use(express.static(path.join(__dirname, 'public')));

// ---- Simulated user store (bcrypt hashed, no plaintext) ---------------

const USERS = {
  'alice': bcrypt.hashSync('Password123!', 10),
  'bob':   bcrypt.hashSync('SecurePass456!', 10),
};

// ---- HTML page builder -----------------------------------------------

function buildPage(title, body) {
  return `<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title} - ShopNow Demo (GTAE-ATRA Protected)</title>
  <style>
    :root {
      --bg: #0b0c16;
      --card-bg: #141628;
      --card-border: #232642;
      --text: #e2e8f0;
      --text-sub: #94a3b8;
      --nav-bg: rgba(14, 16, 32, 0.92);
      --input-bg: #1e2238;
      --input-border: #333858;
      --accent: #8b5cf6;
      --product-card: #191c32;
      --shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    [data-theme="light"] {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --card-border: #e2e8f0;
      --text: #0f172a;
      --text-sub: #64748b;
      --nav-bg: rgba(255, 255, 255, 0.92);
      --input-bg: #f1f5f9;
      --input-border: #cbd5e1;
      --accent: #6d28d9;
      --product-card: #ffffff;
      --shadow: 0 8px 25px rgba(15,23,42,0.06);
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      transition: background 0.3s ease, color 0.3s ease;
    }
    nav {
      background: var(--nav-bg);
      padding: 0.9rem 2rem;
      display: flex;
      align-items: center;
      gap: 1.5rem;
      border-bottom: 1px solid var(--card-border);
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(12px);
    }
    nav a {
      color: var(--text-sub);
      text-decoration: none;
      font-weight: 600;
      font-size: 0.92rem;
      transition: color 0.2s;
    }
    nav a:hover { color: var(--accent); }
    .logo {
      font-size: 1.25rem;
      font-weight: 800;
      color: var(--accent);
      margin-right: auto;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .theme-btn {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 0.4rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.82rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.2s;
    }
    .theme-btn:hover {
      border-color: var(--accent);
      transform: translateY(-1px);
    }
    main { max-width: 960px; margin: 3rem auto; padding: 0 1.5rem; }
    h1 {
      font-size: 2.1rem;
      margin-bottom: 1.5rem;
      background: linear-gradient(135deg, #7c87ff, #a855f7);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 800;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.75rem;
      margin: 1.25rem 0;
      box-shadow: var(--shadow);
      transition: all 0.25s ease;
    }
    .card:hover {
      border-color: var(--accent);
      transform: translateY(-2px);
    }
    .badge {
      display: inline-block;
      padding: 0.3rem 0.85rem;
      border-radius: 99px;
      font-size: 0.8rem;
      font-weight: 700;
      margin: 0.3rem 0.3rem 0.3rem 0;
    }
    .badge-green  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .badge-blue   { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }
    .badge-purple { background: rgba(168,85,247,0.15); color: #c084fc; border: 1px solid rgba(168,85,247,0.3); }
    form input {
      width: 100%;
      padding: 0.75rem 1rem;
      margin: 0.5rem 0;
      background: var(--input-bg);
      border: 1px solid var(--input-border);
      border-radius: 8px;
      color: var(--text);
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s;
    }
    form input:focus { border-color: var(--accent); }
    form button {
      padding: 0.8rem 2.2rem;
      background: linear-gradient(135deg, #7c87ff, #a855f7);
      border: none;
      border-radius: 8px;
      color: white;
      font-weight: 700;
      cursor: pointer;
      margin-top: 0.75rem;
      transition: all 0.2s;
    }
    form button:hover {
      box-shadow: 0 4px 15px rgba(139,92,246,0.35);
      transform: translateY(-1px);
    }
    .product-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 1.25rem;
      margin-top: 1rem;
    }
    .product-card {
      background: var(--product-card);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.25rem;
      text-align: center;
      transition: all 0.25s ease;
      box-shadow: var(--shadow);
    }
    .product-card:hover {
      border-color: var(--accent);
      transform: translateY(-3px);
    }
    .price { color: #10b981; font-size: 1.25rem; font-weight: 800; }
    footer {
      text-align: center;
      padding: 2.5rem;
      color: var(--text-sub);
      font-size: 0.85rem;
      border-top: 1px solid var(--card-border);
      margin-top: 3rem;
    }
  </style>
</head>
<body>
  <nav>
    <span class="logo">🛡️ ShopNow</span>
    <a href="/">Home</a>
    <a href="/products">Products</a>
    <a href="/about">About</a>
    <a href="/contact">Contact</a>
    <a href="/login">Login</a>
    <a href="/profile">Profile</a>
    <button class="theme-btn" onclick="toggleTheme()" id="themeBtn">
      <span id="themeIcon">☀️</span>
      <span id="themeText">Light</span>
    </button>
  </nav>
  <main>${body}</main>
  <footer>GTAE-ATRA-NIDS Demo Application &mdash; Security Monitoring Active</footer>

  <script>
    function toggleTheme() {
      const cur = document.documentElement.getAttribute('data-theme') || 'dark';
      const nxt = cur === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', nxt);
      localStorage.setItem('site_theme', nxt);
      updateThemeBtn(nxt);
    }
    function updateThemeBtn(t) {
      const icon = document.getElementById('themeIcon');
      const text = document.getElementById('themeText');
      if (t === 'light') {
        icon.textContent = '🌙';
        text.textContent = 'Dark';
      } else {
        icon.textContent = '☀️';
        text.textContent = 'Light';
      }
    }
    const saved = localStorage.getItem('site_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    updateThemeBtn(saved);
  </script>
</body>
</html>`;
}

// ---- Routes ----------------------------------------------------------

app.get(['/', '/home'], (req, res) => {
  res.send(buildPage('Home', `
    <h1>Welcome to ShopNow</h1>
    <div class="card">
      <p style="color:#aaa">This is a demo web application instrumented with GTAE-ATRA real-time security monitoring.</p>
      <br>
      <span class="badge badge-green">🟢 Security Monitor: Active</span>
      <span class="badge badge-blue">🔵 GTAE Engine: Running</span>
      <span class="badge badge-purple">🟣 ATRA: Ready</span>
    </div>
    <div class="card">
      <h2 style="margin-bottom:1rem; color:#7c87ff">Featured Products</h2>
      <div class="product-grid">
        ${['Widget Pro', 'Gadget X', 'Device Ultra'].map(p => `
          <div class="product-card">
            <div style="font-size:2rem">📦</div>
            <div style="margin:0.5rem 0">${p}</div>
            <div class="price">$${(Math.random()*100+20).toFixed(2)}</div>
          </div>`).join('')}
      </div>
    </div>
  `));
});

app.get('/products', (req, res) => {
  const products = [
    { id: 1, name: 'Widget Pro 2.0',    price: 49.99, category: 'Electronics' },
    { id: 2, name: 'Gadget Ultra X',    price: 129.99, category: 'Electronics' },
    { id: 3, name: 'SmartDevice Plus',  price: 89.99, category: 'Smart Home' },
    { id: 4, name: 'SecureVault 3000',  price: 199.99, category: 'Security' },
    { id: 5, name: 'CloudHub Connect',  price: 74.99, category: 'Networking' },
  ];
  res.send(buildPage('Products', `
    <h1>Our Products</h1>
    <div class="product-grid">
      ${products.map(p => `
        <div class="product-card">
          <div style="font-size:2rem; margin-bottom:0.5rem">📦</div>
          <strong>${p.name}</strong><br>
          <small style="color:#888">${p.category}</small><br>
          <div class="price" style="margin-top:0.5rem">$${p.price}</div>
        </div>`).join('')}
    </div>
  `));
});

app.get('/about', (req, res) => {
  res.send(buildPage('About', `
    <h1>About Us</h1>
    <div class="card">
      <p>ShopNow is a demo e-commerce application built to demonstrate real-time web security monitoring
         using the GTAE-ATRA-NIDS system.</p>
      <br>
      <p style="color:#aaa">Every request is monitored by the GTAE Graph Transformer Autoencoder,
         which detects anomalous behaviour and triggers the Adaptive Threat Response Algorithm.</p>
    </div>
  `));
});

app.get('/contact', (req, res) => {
  res.send(buildPage('Contact', `
    <h1>Contact Us</h1>
    <div class="card">
      <form method="POST" action="/contact">
        <label>Name</label><input type="text" name="name" placeholder="Your name">
        <label>Email</label><input type="email" name="email" placeholder="your@email.com">
        <label>Message</label><input type="text" name="message" placeholder="Your message">
        <button type="submit">Send Message</button>
      </form>
    </div>
  `));
});

app.post('/contact', (req, res) => {
  res.send(buildPage('Contact', `
    <h1>Thank You!</h1>
    <div class="card"><p>Your message has been received.</p></div>
  `));
});

app.get('/login', (req, res) => {
  const msg = req.query.error ? '<div style="color:#f87171;margin-bottom:1rem">Invalid credentials.</div>' : '';
  res.send(buildPage('Login', `
    <h1>Login</h1>
    <div class="card">
      ${msg}
      <form method="POST" action="/login">
        <label>Username</label>
        <input type="text" name="username" id="username" placeholder="Enter username" autocomplete="username">
        <label>Password</label>
        <input type="password" name="password" id="password" placeholder="Enter password" autocomplete="current-password">
        <button type="submit">Sign In</button>
      </form>
      <p style="margin-top:1rem; color:#888; font-size:0.85rem">Demo users: alice / Password123!</p>
    </div>
  `));
});

app.post('/login', async (req, res) => {
  const { username, password } = req.body;
  const hash = USERS[username];

  if (!hash || !username || !password) {
    return res.status(401).send(buildPage('Login', `
      <h1>Login</h1>
      <div class="card">
        <div style="color:#f87171;margin-bottom:1rem">Invalid credentials.</div>
        <a href="/login" style="color:#7c87ff">Try again</a>
      </div>
    `));
  }

  const match = await bcrypt.compare(password, hash);
  if (!match) {
    return res.status(401).send(buildPage('Login', `
      <h1>Login</h1>
      <div class="card">
        <div style="color:#f87171;margin-bottom:1rem">Invalid credentials.</div>
        <a href="/login" style="color:#7c87ff">Try again</a>
      </div>
    `));
  }

  res.cookie('session', uuidv4(), { httpOnly: true, sameSite: 'strict' });
  res.redirect('/profile');
});

app.get('/profile', (req, res) => {
  const session = req.cookies?.session;
  if (!session) {
    return res.redirect('/login');
  }
  res.send(buildPage('Profile', `
    <h1>My Profile</h1>
    <div class="card">
      <p>Welcome back, User!</p>
      <br>
      <span class="badge badge-green">✓ Authenticated</span>
      <span class="badge badge-blue">Session: ${session.substring(0,8)}...</span>
    </div>
    <div class="card">
      <h3 style="color:#7c87ff; margin-bottom:1rem">Recent Orders</h3>
      <p style="color:#aaa">No orders yet.</p>
    </div>
  `));
});

app.get('/admin', (req, res) => {
  // Admin is restricted -- returns 403 for unauthenticated access
  // Repeated hits on this endpoint trigger ATRA's SensitiveEndpoint detection
  const session = req.cookies?.session;
  if (!session) {
    return res.status(403).send(buildPage('Forbidden', `
      <h1>403 Forbidden</h1>
      <div class="card">
        <p style="color:#f87171">Access to this resource is restricted.</p>
        <p style="color:#aaa; margin-top:0.5rem">This access attempt has been logged by the security monitoring system.</p>
      </div>
    `));
  }
  res.send(buildPage('Admin', `
    <h1>Admin Panel</h1>
    <div class="card"><p>Administrative functions.</p></div>
  `));
});

// ---- JSON APIs -------------------------------------------------------

app.get('/api/products', (req, res) => {
  res.json({
    products: [
      { id: 1, name: 'Widget Pro 2.0',   price: 49.99,  stock: 120 },
      { id: 2, name: 'Gadget Ultra X',   price: 129.99, stock: 45  },
      { id: 3, name: 'SmartDevice Plus', price: 89.99,  stock: 78  },
    ],
    total: 3,
  });
});

app.get('/api/categories', (req, res) => {
  res.json({
    categories: ['Electronics', 'Smart Home', 'Security', 'Networking'],
  });
});

app.get('/api/search', (req, res) => {
  const q = req.query.q || '';
  res.json({
    query: q,
    results: [
      { id: 1, name: 'Widget Pro 2.0', relevance: 0.92 },
    ],
    count: 1,
  });
});

// ---- Error handlers --------------------------------------------------

app.use((req, res) => {
  res.status(404).send(buildPage('404 Not Found', `
    <h1>404 Not Found</h1>
    <div class="card">
      <p style="color:#aaa">The page <code style="color:#7c87ff">${req.path}</code> was not found.</p>
      <p style="margin-top:1rem"><a href="/" style="color:#7c87ff">Go Home</a></p>
    </div>
  `));
});

app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send(buildPage('Error', `
    <h1>Server Error</h1>
    <div class="card"><p style="color:#f87171">An internal error occurred.</p></div>
  `));
});

// ---- Start -----------------------------------------------------------

app.listen(PORT, HOST, () => {
  console.log('='.repeat(70));
  console.log('  GTAE-ATRA DEMO WEB APPLICATION');
  console.log('='.repeat(70));
  console.log(`  URL         : http://localhost:${PORT}`);
  console.log(`  Monitor     : ${process.env.MONITOR_URL || 'http://127.0.0.1:8765/telemetry'}`);
  console.log('  Endpoints   : / /home /products /about /login /profile /admin');
  console.log('  API         : /api/products /api/categories /api/search');
  console.log('='.repeat(70));
});

module.exports = app;
