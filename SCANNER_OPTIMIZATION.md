# Scanner Optimization & Architecture Reference

> Last updated: Feb 28, 2026 — v9-executor-fix  
> All 3 scanners tested concurrently: War Room 5.2s, Regime 5.8s, Blood 12.4s cold / sub-1s warm

---

## Architecture Overview

```
Browser (fetchWithRetry)
    │
    ▼
Railway (single container, 1 replica)
    │
    ├── app.py .............. Entry point (Procfile: uvicorn app:app)
    │     ├── Instant /api/health endpoint
    │     ├── on_startup: expands default asyncio executor → 20 threads
    │     └── Background thread → imports unified_server.py → copies routes
    │
    ├── unified_server.py ... Main FastAPI app (all endpoints)
    │     ├── Server-level caches (per scanner, with TTL)
    │     ├── asyncio.wait_for() timeout guards on every scanner endpoint
    │     └── Static files mount (public/)
    │
    ├── war_room.py ......... Polygon intraday bars → extension DNA analysis
    ├── buffett_scanner.py .. yfinance fundamentals → value + blood-in-streets
    ├── regime_scanner.py ... Polygon daily bars → bull/bear/choppy classification
    └── polygon_data.py ..... Shared Polygon API layer + rate limiter
```

---

## Critical Lessons Learned

### 1. `unified_server.py`'s `on_startup` NEVER FIRES
`app.py` creates its own FastAPI `app` and that's what uvicorn runs.  
`unified_server.py` has its own `app` object, but it's imported as a module — its startup events never execute.  
**All startup logic must go in `app.py`'s `on_startup`.**

### 2. Default asyncio executor was only ~5 threads
Every `await loop.run_in_executor(None, ...)` shares the default executor.  
With only 5 threads and 3+ scanners running concurrently, requests queue indefinitely → server freeze.  
**Fix:** `loop.set_default_executor(ThreadPoolExecutor(max_workers=20))` in `app.py` on_startup.

### 3. Rate limiter must release lock before sleeping
Original `polygon_data.py` rate limiter held the lock during `time.sleep()` — serialized ALL callers across ALL scanners.  
**Fix:** Check-sleep-retry loop outside the lock.

### 4. Railway snapshot timeouts on large repos
At 58MB / 600 commits, Railway's "Initialization > Snapshot code" step can time out (7+ min).  
**Fix:** Push empty commit to retry: `git commit --allow-empty -m "Trigger redeploy"; git push`  
**Long-term:** Slim repo with `git filter-repo` to strip old large blobs (public/index.html has 10+ 600KB versions).

---

## Per-Scanner Setup

### War Room (`/api/war-room`)
| Item | Detail |
|------|--------|
| **Data source** | Polygon.io (5-min intraday bars, 30-60 days) |
| **Scanner module** | `war_room.py` |
| **Thread pool** | Own pool: `ThreadPoolExecutor(max_workers=4)` (was 10) |
| **Rate limiter** | `_fetch_aggs()` calls shared `polygon_data._limiter.acquire()` |
| **Server cache** | `_warroom_cache` in unified_server.py, 2-min TTL |
| **Backend timeout** | `asyncio.wait_for(async_run_war_room(...), timeout=45)` |
| **Frontend timeout** | `fetchWithRetry()` — 45s per attempt, 2 retries |
| **Internal cache** | `_dna_cache` in war_room.py, 2-min TTL |
| **Cold perf** | ~1.5-5s (single ticker) |
| **Warm perf** | ~0.3s |

### Blood / Buffett Scanner (`/api/buffett-scan`)
| Item | Detail |
|------|--------|
| **Data source** | yfinance (Yahoo Finance HTTP) |
| **Scanner module** | `buffett_scanner.py` |
| **Thread pool** | Uses default executor (no own pool) |
| **Rate limiter** | N/A (yfinance, not Polygon) |
| **Server cache** | `_buffett_cache` in unified_server.py, 5-min TTL |
| **Backend timeout** | `asyncio.wait_for(..., timeout=30)` |
| **Frontend timeout** | `fetchWithRetry()` — 45s per attempt, 2 retries |
| **Internal cache** | `_info_cache` in buffett_scanner.py, 5-min TTL |
| **Cold perf** | ~3-12s (yfinance is slower than Polygon) |
| **Warm perf** | ~0.3-0.5s |

### Regime Scanner (`/api/regime-scan`)
| Item | Detail |
|------|--------|
| **Data source** | Polygon.io (daily bars) |
| **Scanner module** | `regime_scanner.py` |
| **Thread pool** | Internal `ThreadPoolExecutor(max_workers=6)` for prefetch (was 2) |
| **Rate limiter** | Uses shared `polygon_data._limiter` via `get_polygon_bars()` |
| **Server cache** | `_regime_cache` in unified_server.py, 5-min TTL |
| **Backend timeout** | `asyncio.wait_for(...)` — single 45s, multi 60s |
| **Frontend timeout** | `fetchWithRetry()` — 45s per attempt, 2 retries |
| **Internal cache** | `_bars_cache` in polygon_data.py, 2-min TTL |
| **Cold perf** | ~5-8s (5 tickers / 30 days) |
| **Warm perf** | ~0.5-0.7s |

---

## Thread Pool Budget

All pools are capped to prevent GIL thrash:

| Pool | File | Max Workers | Purpose |
|------|------|-------------|---------|
| Default executor | `app.py` on_startup | 40 | All `asyncio.to_thread()` / `run_in_executor(None, ...)` calls |
| War Room | `war_room.py` | 6 | Polygon intraday fetches (own async endpoint only) |
| Regime prefetch | `regime_scanner.py` | 4-6 | Parallel bar prefetch |
| Alpha scanner | `alpha_scanner.py` | 4 | Alpha signal analysis |
| Alpha legacy | `alpha_scanner_legacy.py` | 4 | Legacy alpha |
| ~~Combo scanner~~ | ~~`combo_scanner.py`~~ | **removed** | Uses default executor now |
| ~~Options flow~~ | ~~`options_flow_scanner.py`~~ | **removed** | Uses default executor now |
| ~~Sustainability~~ | ~~`sustainability_endpoints.py`~~ | **removed** | Uses default executor now |
| ~~Research builder~~ | ~~`picks_shovels_builder.py`~~ | **removed** | Uses default executor now |
| **Total persistent** | — | **~18** | Was 51+ before optimization |

---

## Shared Rate Limiter (Polygon)

**File:** `polygon_data.py` → `_limiter = _RateLimiter()` at 5 req/s

All Polygon callers must use this limiter to prevent 429 cascades:
- `polygon_data.get_polygon_bars()` — uses it automatically
- `war_room._fetch_aggs()` — imports and calls `_limiter.acquire()`
- `polygon_options.py` — uses `_acquire_rate_limit()` helper (3 call sites)

**If adding a new module that calls Polygon directly:**
```python
from polygon_data import _limiter
_limiter.acquire()  # call before every requests.get() to Polygon
```

---

## Frontend Retry Pattern

**File:** `public/config.js` → `fetchWithRetry(url, opts)`

```javascript
// Usage in any scanner HTML:
const resp = await fetchWithRetry(`${BACKEND}/api/your-endpoint?params`, {
    timeout: 45000,   // per-attempt timeout (ms)
    retries: 2,       // 2 retries = 3 total attempts
    retryDelay: 2000  // delay between retries (ms)
});
```

- Auto-retries on 503 (server loading) and timeout
- Shows yellow "Server warming up — retry 1/2..." in `#progressContainer`
- Throws after all attempts exhausted

**Pages using it:** `buffett.html`, `regime.html`, `warroom.html`

---

## Checklist: Optimizing a New Scanner

1. **Cap thread pool** — max 4 workers for any dedicated ThreadPoolExecutor
2. **Use shared rate limiter** — if calling Polygon, import `_limiter` from `polygon_data`
3. **Add server-level cache** in `unified_server.py`:
   ```python
   _my_cache: dict = {}
   _MY_TTL = 300  # seconds
   # In endpoint: check cache_key, return if fresh, else compute + store
   ```
4. **Wrap in `asyncio.wait_for()`** — 30-45s timeout on the endpoint
5. **Frontend: use `fetchWithRetry()`** — include `config.js`, use 45s timeout
6. **Backend timeout on external calls** — set `timeout=10-15` on `requests.get()` / `yf.download()`
7. **Test concurrently** — fire all scanners at once, verify no freezing
8. **Monitor thread count** — should stay under ~25 total persistent threads

---

## Deploy Notes

- **Procfile:** `web: python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}`
- **railway.toml:** `healthcheckPath = "/api/health"`, `healthcheckTimeout = 300`
- **Normal deploy:** 30-120s. During deploy, old container killed → ~30s gap → new container starts
- **If deploy hangs:** Check Railway dashboard for snapshot timeout → push empty commit to retry
- **Version tag:** Check `GET /api/health` → `version` field confirms which code is running
