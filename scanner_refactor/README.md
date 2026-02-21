# Scanner Refactor — Changelog

## Overview
Structural improvements to the scanner system: centralized config, unified watchlists, rate limiting, circuit breaker, and test coverage. All changes are backward compatible — existing code that doesn't pass config will work identically to before.

---

## New Files

### `universe.py`
Single source of truth for all stock symbol lists. Replaces 6+ copies of hardcoded watchlists scattered across scanner files.

**Exports:**
- `TECH`, `MEGA`, `ETFS`, `MEME`, `ALL_SYMBOLS` — core universes (72 symbols)
- `ALPHA_UNIVERSES` — dict used by `alpha_scanner.py` (tech, semis, momentum, etfs, mag7, all)
- `OPTIONS_PRESETS` — dict used by `options_flow_scanner.py` (mega_tech, sp500_top, meme_watch, etc.)
- `BUFFETT_PRESETS` — dict used by `buffett_scanner.py` (mega_cap, blue_chip, growth, etc.)
- `AUTO_SCANNER_DEFAULTS` — 24-symbol default list for `auto_scanner.py`
- `get_universe(name)` — lookup helper with fallback to ALL_SYMBOLS
- `list_universes()` — returns all universe names with symbol counts

**To add a ticker:** Edit `universe.py` once. All scanners pick it up automatically.

### `test_scoring.py`
25 unit tests covering:
- **SignalScorer** — strong bull/bear signals, neutral, yellow/mixed, divergence bonus, score bounds
- **Config wiring** — conservative/aggressive/default thresholds, flow config, propagation through MTFAuctionScanner
- **VolumeProfileEngine** — basic VP calc, different bin counts, small data edge case
- **RSIEngine** — range bounds, uptrend/downtrend behavior, zone classification
- **FlowControlEngine** — imbalance range, buy/sell percentage sum
- **Universe** — dedup check, get_universe lookup, fallback, preset emptiness

Run: `python test_scoring.py` or `python -m pytest test_scoring.py -v`

---

## Modified Files

### `mtf_auction_scanner_v2.py`
**Config wiring into all engines.**

- `MTFAuctionScanner.__init__(config=None)` — accepts optional `SwingTradeConfig`, propagates sub-configs to all engines
- `VolumeProfileEngine.__init__(config=None)` — reads `value_area_pct` and `num_bins` from `VolumeProfileConfig`
- `FlowControlEngine.__init__(config=None)` — reads `momentum_period` from `FlowConfig`
- `RSIEngine.__init__(config=None)` — reads `period` and `slope_period` from `RSIConfig`
- `SignalScorer.__init__(config=None, flow_config=None)` — reads `strong_threshold`, `moderate_threshold`, `min_score_gap` from `ScoringConfig`; reads flow imbalance thresholds from `FlowConfig`
- Flow scoring section now uses `self._flow_strong/moderate/mild` instead of hardcoded `0.3/0.15/0.05`

**Usage:**
```python
from scanner_config import CONSERVATIVE, BALANCED, ACTIVE

# Before (still works):
scanner = MTFAuctionScanner()

# Now with config:
scanner = MTFAuctionScanner(config=CONSERVATIVE)  # Higher thresholds, fewer signals
scanner = MTFAuctionScanner(config=ACTIVE)         # Lower thresholds, more signals
```

### `integrated_scanner_v2.py`
- `IntegratedScanner.__init__(config=None)` — propagates config to `MTFAuctionScanner`

### `polygon_data.py`
**Thread-safe rate limiter for Polygon API.**

- Added `_RateLimiter` class — token bucket pattern, default 5 req/s (Polygon free tier)
- Global `_limiter` instance shared across all scanners
- Wired into `_fetch_aggs()` and `get_price_quote()` — every Polygon call goes through the limiter
- `_limiter.set_rate(n)` — adjust for paid tier (e.g., `_limiter.set_rate(100)`)
- `_limiter.stats` — returns `{total_requests, total_waits, rate_limit}`

### `auto_scanner.py`
**Circuit breaker + centralized watchlist.**

- `_should_skip_post(result)` — prevents garbage Discord posts when data sources are down
  - Triggers on: 2+ scan errors, all empty + errors, 3 consecutive empty cycles
- `DEFAULT_SYMBOLS` now imports from `universe.py`

### `alpha_scanner.py`
- `UNIVERSES` dict replaced with `from universe import ALPHA_UNIVERSES as UNIVERSES`

### `full_scan_discord.py`
- `TECH`, `MEGA`, `ETFS`, `MEME`, `ALL_SYMBOLS` replaced with imports from `universe.py`

### `push_scan_discord.py`
- Same — hardcoded lists replaced with `from universe import ALL_SYMBOLS`

### `options_flow_scanner.py`
- `PRESETS` replaced with `from universe import OPTIONS_PRESETS as PRESETS`

### `buffett_scanner.py`
- `PRESETS` replaced with `from universe import BUFFETT_PRESETS as PRESETS`

### `emtryscan/vp_scanner_integration.py`
- Added docstring noting this VP calc is intentionally standalone (uses `PriceBar` input, not DataFrame) and that the canonical VP lives in `market_scanner_v2.TechnicalCalculator`

---

## What Didn't Change
- All V1-compatible call signatures still work (no config = same defaults as before)
- Scoring math and point values unchanged
- Data source priority unchanged (Polygon > Alpaca > Finnhub > yfinance)
- Discord embed formatting unchanged
- No files deleted
