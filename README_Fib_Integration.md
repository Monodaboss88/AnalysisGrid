# Fibonacci Retracement Detection + Trade Journal Integration

## What This Fixes

### Fib Detection (fib_retracement.py)
The V1 fib detection in `unified_server.py` had 4 problems:

| Issue | V1 Behavior | V2 Fix |
|-------|-------------|--------|
| **Swing detection** | Raw max/min of entire lookback period | Pivot-based swing points with strength validation (3+ bars each side) |
| **Range too large** | Falls back to 95th/5th percentile (synthetic levels) | Scores swing pairs and selects the best quality match with real price action |
| **Range too small** | Fabricates a 5% range from nothing | Returns `None` (no analysis) — no fake levels |
| **Legacy field bug** | `fib_236` always mapped to bearish fibs regardless of trend | Legacy fields now map to the trend-appropriate (active) set |

### Trade Journal Gap (trade_journal.py)
The fib data was calculated, piped into GPT as text, then **vanished**. None of it was stored with trades:

| Component | Before | After |
|-----------|--------|-------|
| JournalEntry dataclass | 0 fib fields | 12 fib fields + 4 context fields |
| Database schema | 0 fib columns | 16 new columns (auto-migrated) |
| LogTradeRequest API | 0 fib params | 16 new params |
| Trade stats | No fib analytics | Zone win rates, golden zone edge, confluence edge |
| API endpoints | None | `/fib-report`, `/fib-stats` |
| UI trade logging | Ignores fib data | All 3 paths pass fib context |

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `fib_retracement.py` | ~520 | Standalone fib detection with pivot-based swing detection |
| `trade_journal.py` | ~480 | Updated journal with fib fields + fib analytics |
| `trade_journal_endpoints.py` | ~310 | Updated API with fib params + fib report endpoints |
| `UI_FIB_PATCH.py` | ~220 | JavaScript patches for unified_ui_v2.html |

## Fib Detection: How It Works

### Swing Point Detection
Instead of raw max/min, the detector finds **validated pivot points**:

```
A swing HIGH must have its high > all bars within N bars on BOTH sides
A swing LOW must have its low < all bars within N bars on BOTH sides
```

Scans strength levels 3 through 7, deduplicates, then selects the **best pair** based on:
- Pivot strength (higher = more validated)
- Range quality (2-15% sweet spot for swing trading)
- Recency (prefer recent swings)
- Volume at swing points (high volume = more significant)

### Quality Scoring (0-100)
Every fib analysis includes a swing quality score:

| Score | Grade | Meaning |
|-------|-------|---------|
| 80+ | STRONG | Well-validated pivots, clean range, high volume |
| 55-79 | MODERATE | Acceptable pivots, usable levels |
| < 55 | WEAK | Fallback to raw max/min, use with caution |

### VP + Fib Confluence
Detects when Volume Profile levels land within 1.5% of a Fib level:

```
VAL ≈ Fib 61.8% at $647.22 (STRONG)    ← 0.3% apart
POC ≈ Fib 50% at $642.85 (NEAR)        ← 1.2% apart
```

Confluence quality: EXACT (<0.5%), STRONG (<1%), NEAR (<1.5%)

## Trade Journal: Fib Analytics

After logging trades with fib context, the journal answers these questions:

### 1. Does the Golden Zone work for me?
```
⭐ GOLDEN ZONE (50%-61.8%):
   Trades: 12 | Win Rate: 75% | Avg R: 2.3
   🟢 EDGE: +18% vs non-golden trades (57%)
```

### 2. Does VP+Fib confluence give an edge?
```
🎯 VP+FIB CONFLUENCE EDGE:
   With Confluence:    8 trades | 81% WR | 2.8R avg
   Without Confluence: 15 trades | 53% WR | 1.1R avg
   🟢 CONFLUENCE WORKS: +28% edge
```

### 3. Which fib zones produce the best results?
```
📊 WIN RATE BY FIB ZONE:
   🟢 GOLDEN_ZONE        : 75.0% WR (12 trades, +2.30R)
   🟢 PULLBACK_ENTRY      : 66.7% WR (6 trades, +1.80R)
   🟡 PULLBACK_SHALLOW    : 50.0% WR (4 trades, +0.50R)
   🔴 DEEP_RETRACE        : 33.3% WR (3 trades, -0.90R)
```

### 4. Does swing quality matter?
```
📏 SWING QUALITY IMPACT:
   STRONG    : 72% WR (18 trades, +2.10R)
   MODERATE  : 55% WR (11 trades, +0.80R)
   WEAK      : 40% WR (5 trades, -0.30R)
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/journal/log` | POST | Log trade (now with fib context) |
| `/api/journal/stats` | GET | Stats + fib analytics |
| `/api/journal/fib-report` | GET | Human-readable fib performance report |
| `/api/journal/fib-stats` | GET | Raw fib analytics data for dashboards |

## Usage

### Standalone Fib Analysis
```python
from fib_retracement import analyze_fibs, format_fib_analysis

# Analyze with daily OHLCV data
result = analyze_fibs(df_daily, symbol="META", vah=667.72, poc=660.40, val=647.22)

# Human-readable output
print(format_fib_analysis(result))

# Dict for API response
api_data = result.to_dict()

# Dict for trade journal storage
journal_fields = result.to_journal_fields()
```

### Integration with unified_server.py
Replace the inline fib calculation (lines 2514-2700) with:
```python
from fib_retracement import analyze_fibs

# In the complete analysis endpoint:
fib_result = analyze_fibs(df_fib, symbol, vah=vah, poc=poc, val=val)
if fib_result:
    response["fib_levels"] = fib_result.to_dict()
    response["fib_position"] = fib_result.price_position
    response["fib_zone"] = fib_result.price_zone
    response["fib_quality"] = fib_result.swing_quality
    response["fib_confluence"] = fib_result.confluences
```

### Fib Performance Report
```python
journal = TradeJournal()
print(journal.get_fib_report(user_id="rob", days=90))
```

## Database Migration
The updated `trade_journal.py` auto-migrates existing databases. When it connects, it checks for each new column and adds it if missing. No data loss, no manual migration needed.
