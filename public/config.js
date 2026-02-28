// Shared configuration for all frontend pages
// Include this script before any page-specific JS

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? ''
    : 'https://analysisgrid-production.up.railway.app';

// Alias — some pages use BACKEND instead of API_BASE
const BACKEND = API_BASE;

/**
 * fetch() with timeout + auto-retry on failure/503.
 * @param {string} url
 * @param {object} opts - { timeout: ms, retries: count, retryDelay: ms }
 */
async function fetchWithRetry(url, opts = {}) {
  const timeout = opts.timeout || 30000;   // 30s default
  const retries = opts.retries || 2;       // 2 retries = 3 total attempts
  const retryDelay = opts.retryDelay || 2000;
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const resp = await fetch(url, { signal: controller.signal });
      clearTimeout(timer);
      if (resp.status === 503 && attempt < retries) {
        // Server still loading — show retry status + retry after delay
        _showRetryStatus(attempt + 1, retries);
        await new Promise(r => setTimeout(r, retryDelay));
        continue;
      }
      return resp;
    } catch (e) {
      clearTimeout(timer);
      lastErr = e;
      if (e.name === 'AbortError') lastErr = new Error('Request timed out — server may be restarting. Try again.');
      if (attempt < retries) {
        _showRetryStatus(attempt + 1, retries);
        await new Promise(r => setTimeout(r, retryDelay));
        continue;
      }
    }
  }
  throw lastErr;
}

/** Show a subtle retry notification so users know it's not stuck */
function _showRetryStatus(attempt, maxRetries) {
  const prog = document.getElementById('progressContainer');
  if (prog) {
    const existing = prog.querySelector('.retry-note');
    if (existing) existing.remove();
    const note = document.createElement('div');
    note.className = 'retry-note';
    note.style.cssText = 'color:#f59e0b;font-size:0.75rem;margin-top:6px;';
    note.textContent = `Server warming up — retry ${attempt}/${maxRetries}...`;
    prog.appendChild(note);
  }
}
