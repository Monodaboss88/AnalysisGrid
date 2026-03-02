# Polygon Proxy Conversion Guide

## What This Fixes
Frontend pages that call `https://api.polygon.io` **directly from the browser** — exposing the API key client-side and breaking when the backend is slow/down because they first fetch the key from `/api/polygon-key`.

After conversion, all Polygon calls go **through the backend** (server-side proxy). The API key never leaves the server. Works exactly like MTF Auction does.

---

## Backend (Already Done)

Two proxy endpoints exist in `unified_server.py`:

### `/api/polygon/bars/{ticker}`
Proxies → `https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{m}/{ts}/{from}/{to}`

| Param | Type | Default | Description |
|---|---|---|---|
| `from_date` | str | required | Start date (YYYY-MM-DD) |
| `to_date` | str | required | End date (YYYY-MM-DD) |
| `timespan` | str | `"day"` | day, hour, minute, etc. |
| `multiplier` | int | `1` | Timespan multiplier |
| `adjusted` | str | `"true"` | Adjusted for splits |
| `sort` | str | `"asc"` | Sort order |
| `limit` | int | `5000` | Max results |

### `/api/polygon/news/{ticker}`
Proxies → `https://api.polygon.io/v2/reference/news`

| Param | Type | Default | Description |
|---|---|---|---|
| `from_date` | str | optional | published_utc.gte |
| `to_date` | str | optional | published_utc.lte |
| `limit` | int | `3` | Max results |
| `sort` | str | `"published_utc"` | Sort field |

Both return the **exact same JSON** that Polygon returns — so `data.results` still works the same in the frontend.

---

## Frontend Conversion Checklist (Per File)

### Step 1: Add config.js (if missing)

If the file does NOT already have `<script src="config.js"></script>`, add it before the main `<script>` tag:

```html
<script src="config.js"></script>
```

This provides `BACKEND` (resolves to `''` on localhost, `'https://analysisgrid-production.up.railway.app'` in prod).

If the file has a hardcoded `BACKEND` const, **remove it** (config.js handles this).

### Step 2: Remove `const API = 'https://api.polygon.io'`

Delete this line entirely. No longer needed.

### Step 3: Replace DOMContentLoaded

**BEFORE** (fetches API key from backend, stores in hidden input):
```javascript
document.addEventListener('DOMContentLoaded', async () => {
  // ... setup code ...
  try {
    const resp = await fetch(`${BACKEND}/api/polygon-key`);
    if (resp.ok) {
      const data = await resp.json();
      document.getElementById('apiKeyInput').value = data.key;
      document.getElementById('apiDot').style.background = '#22c55e';
      document.getElementById('apiLabel').textContent = 'Polygon Connected';
      // ...
    } else {
      // localStorage fallback ...
    }
  } catch {
    // localStorage fallback ...
  }
});
```

**AFTER** (just pings health to confirm backend is up):
```javascript
document.addEventListener('DOMContentLoaded', async () => {
  // ... setup code (setQuickRange, event listeners, etc.) ...
  try {
    const ctrl = new AbortController();
    const tid = setTimeout(() => ctrl.abort(), 10000);
    const resp = await fetch(`${BACKEND}/api/health`, { signal: ctrl.signal });
    clearTimeout(tid);
    if (resp.ok) {
      document.getElementById('apiDot').style.background = '#22c55e';
      document.getElementById('apiLabel').textContent = 'Polygon Connected';
      document.getElementById('apiStatus').style.borderColor = 'rgba(34,197,94,0.4)';
    } else {
      document.getElementById('apiDot').style.background = '#ef4444';
      document.getElementById('apiLabel').textContent = 'Backend Error';
    }
  } catch {
    document.getElementById('apiDot').style.background = '#ef4444';
    document.getElementById('apiLabel').textContent = 'Backend Offline';
  }
});
```

> If the page has NO status dot UI (files `_2`, `_3`), skip the DOM updates — just do the health check or remove the try/catch entirely.

### Step 4: Replace `apiFetch`, `getBars`, `getNews`

**BEFORE:**
```javascript
async function apiFetch(ep, key) {
  const sep = ep.includes('?') ? '&' : '?';
  const r = await fetch(`${API}${ep}${sep}apiKey=${key}`);
  if (!r.ok) throw new Error(`API error ${r.status} for ${ep}`);
  return r.json();
}

async function getBars(ticker, from, to, key) {
  const d = await apiFetch(`/v2/aggs/ticker/${ticker}/range/1/day/${from}/${to}?adjusted=true&sort=asc&limit=5000`, key);
  return d.results || [];
}

async function getNews(ticker, date, key) {
  try {
    const d = new Date(date);
    const a = new Date(d); a.setDate(a.getDate() - 1);
    const b = new Date(d); b.setDate(b.getDate() + 1);
    const r = await apiFetch(`/v2/reference/news?ticker=${ticker}&published_utc.gte=${iso(a)}&published_utc.lte=${iso(b)}&limit=3&sort=published_utc`, key);
    return r.results || [];
  } catch { return []; }
}
```

**AFTER:**
```javascript
async function getBars(ticker, from, to) {
  const r = await fetch(`${BACKEND}/api/polygon/bars/${ticker}?from_date=${from}&to_date=${to}&adjusted=true&sort=asc&limit=5000`);
  if (!r.ok) throw new Error(`Bars API error ${r.status} for ${ticker}`);
  const d = await r.json();
  return d.results || [];
}

async function getNews(ticker, date) {
  try {
    const d = new Date(date);
    const a = new Date(d); a.setDate(a.getDate() - 1);
    const b = new Date(d); b.setDate(b.getDate() + 1);
    const r = await fetch(`${BACKEND}/api/polygon/news/${ticker}?from_date=${iso(a)}&to_date=${iso(b)}&limit=3&sort=published_utc`);
    if (!r.ok) return [];
    const data = await r.json();
    return data.results || [];
  } catch { return []; }
}
```

> `apiFetch` is **deleted entirely** — no longer needed.

### Step 5: Update `runScan()`

**Remove** the `key` variable and key check:
```javascript
// DELETE these lines:
const key = document.getElementById('apiKeyInput').value.trim();
if (!key) return showError('Polygon API key not available — check backend connection.');
// Also DELETE:
localStorage.setItem('polygon_api_key', key);
```

**Update** all calls that pass `key`:
```javascript
// BEFORE:
const bars_data = await getBars(tk, from, to, key);
outs[i].news = await getNews(tk, outs[i].dateISO, key);

// AFTER:
const bars_data = await getBars(tk, from, to);
outs[i].news = await getNews(tk, outs[i].dateISO);
```

### Step 6: Clean up hidden input (optional)

If present, the hidden input is now unused:
```html
<!-- Can remove or leave: -->
<input type="hidden" id="apiKeyInput" value="">
```

---

## Files That Need Conversion

| File | Location | Served? | Status |
|---|---|---|---|
| `public/catalyst.html` | public/ | YES | **DONE** (v17b) |
| `stock-catalyst-scanner_1.html` | root | NO (404) | Old version of catalyst |
| `stock-catalyst-scanner_2.html` | root | NO (404) | Old version, no BACKEND |
| `stock-catalyst-scanner_3.html` | root | NO (404) | Old version, no BACKEND |

> **The 3 root files are NOT served** — they're old backups. Only `public/catalyst.html` is live and already converted. If you move any of these into `public/`, they'll need the conversion above.

---

## Quick Diff Summary (what changed in `public/catalyst.html`)

```
REMOVED:
- const API = 'https://api.polygon.io';
- async function apiFetch(ep, key) { ... }
- All /api/polygon-key fetch logic + localStorage fallback
- const key = document.getElementById('apiKeyInput').value.trim();
- if (!key) return showError('Polygon API key not available...');
- getBars(tk, from, to, key)  →  key param removed
- getNews(tk, ..., key)       →  key param removed

ADDED:
- Health check via ${BACKEND}/api/health on page load
- getBars() calls ${BACKEND}/api/polygon/bars/${ticker}?from_date=...
- getNews() calls ${BACKEND}/api/polygon/news/${ticker}?from_date=...

BACKEND SIDE (unified_server.py):
- /api/polygon/bars/{ticker}  — proxies Polygon aggregates
- /api/polygon/news/{ticker}  — proxies Polygon news
- middleware: Cache-Control: no-cache on all .html responses
- explicit /catalyst.html route with no-cache headers
```
