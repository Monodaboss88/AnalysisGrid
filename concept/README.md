# SEF Trading System

## Multi-Timeframe Volume Profile + VWAP + Extension Duration Predictor

A professional-grade trading system that combines:
- **Volume Profile** (VAH/VAL/POC) across multiple timeframes
- **VWAP** with standard deviation bands
- **Extension Duration Predictor** — tracks TIME × PRICE EXTENSION

**Data Provider:** Polygon.io  
**Execution:** Tradier

---

## The Extension Duration Concept

The key innovation: **It's not just WHERE price is, but HOW LONG it's been there.**

| Candles | Hours | Trigger | Snap-Back Prob | Action |
|---------|-------|---------|----------------|--------|
| 1 | 2h | 👀 WATCHING | 45% | Wait |
| 2 | 4h | ⚠️ ALERT | 55% | Prepare |
| 3 | 6h | 🔥 HIGH_PROB | 65% | Look for entry |
| 4+ | 8h+ | 💥 EXTREME | 75%+ | High conviction |

**Think of it like a rubber band:**
- Price extended 1 ATR for 6 hours = tension building
- Price extended 2 ATR for 2 hours = tension building
- Both multiply together = **Extension Energy**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MAIN ORCHESTRATOR                         │
│                          (main.py)                               │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│  POLYGON.IO   │      │   ANALYTICS   │      │   TRADIER     │
│  Data Client  │      │    ENGINE     │      │   Execution   │
└───────────────┘      └───────────────┘      └───────────────┘
        │                       │                       │
        │         ┌─────────────┼─────────────┐        │
        │         ▼             ▼             ▼        │
        │  ┌───────────┐ ┌───────────┐ ┌───────────┐  │
        │  │  VOLUME   │ │   VWAP    │ │ EXTENSION │  │
        │  │  PROFILE  │ │  ENGINE   │ │ PREDICTOR │  │
        │  └───────────┘ └───────────┘ └───────────┘  │
        │         │             │             │        │
        │         └─────────────┼─────────────┘        │
        │                       ▼                      │
        │              ┌─────────────┐                 │
        │              │   SIGNAL    │                 │
        │              │  GENERATOR  │                 │
        │              └─────────────┘                 │
        │                       │                      │
        └───────────────────────┴──────────────────────┘
```

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All configuration parameters |
| `polygon_client.py` | Polygon.io data fetching + WebSocket |
| `volume_profile.py` | Multi-TF VP engine (VAH/VAL/POC) |
| `vwap_engine.py` | Anchored VWAPs with bands |
| `extension_predictor.py` | **Time × Extension tracking** |
| `extension_dashboard.py` | Visual dashboard for extension status |
| `signal_generator.py` | Hybrid signal logic |
| `execution.py` | Tradier order execution |
| `main.py` | Main orchestrator |
| `utils.py` | Helper functions |

---

## Extension Predictor Logic

### What It Tracks

For each reference level (VWAP, POC, VAH, VAL):
- **Consecutive 2H candles** in extension
- **Direction** (above or below)
- **Extension distance** in ATR units
- **Candle characteristics** (rejection, continuation)

### Trigger Levels

```python
1 candle  (2h)  = WATCHING     # Just observing
2 candles (4h)  = ALERT        # Something brewing
3 candles (6h)  = HIGH_PROB    # High probability setup
4+ candles (8h+) = EXTREME     # Snap-back imminent
```

### Probability Adjustments

| Factor | Adjustment |
|--------|------------|
| Rejection candle (long wick toward value) | +10% |
| Declining volume during extension | +5% |
| Extreme extension (> 2 ATR) | +5% |
| Continuation candle | -10% |

### Dashboard Output

```
┌────────────────────────────────────────────────────────────┐
│ EXTENSION STATUS: SPY                   2024-01-15 15:30  │
├────────────────────────────────────────────────────────────┤
│ Level          │ Dir   │ Candles │ Hours │ Status         │
├────────────────────────────────────────────────────────────┤
│ poc_daily      │ ▲ UP  │    4    │  8.0  │ 💥 EXTREME     │
│ vwap_daily     │ ▲ UP  │    3    │  6.0  │ 🔥 HIGH_PROB   │
├────────────────────────────────────────────────────────────┤
│ ACTION: SHORT setup @ poc_daily [75% snap-back]           │
└────────────────────────────────────────────────────────────┘
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
export POLYGON_API_KEY="your_polygon_key"
export TRADIER_API_KEY="your_tradier_key"
export TRADIER_ACCOUNT_ID="your_account_id"
```

### 3. Configure Symbols

Edit `config.py`:

```python
symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT"])
```

### 4. Run

```bash
python main.py
```

---

## Trade Setup (When Extension Triggers)

### Mean Reversion SHORT (Price Above VWAP for 3+ candles)

```
Entry:   On rejection candle (long upper wick)
Target:  VWAP or POC (snap-back target)
Stop:    Above the extension high + 0.5 ATR
```

### Mean Reversion LONG (Price Below VWAP for 3+ candles)

```
Entry:   On rejection candle (long lower wick)
Target:  VWAP or POC (snap-back target)
Stop:    Below the extension low - 0.5 ATR
```

---

## Configuration Options

### Extension Predictor Config

```python
@dataclass
class ExtensionConfig:
    watching_threshold: int = 1    # Start watching
    alert_threshold: int = 2       # 4 hours
    high_prob_threshold: int = 3   # 6 hours
    extreme_threshold: int = 4     # 8+ hours
    
    min_extension_atr: float = 0.5  # Min distance to count as extended
    
    rejection_candle_bonus: float = 0.10
    declining_volume_bonus: float = 0.05
    extreme_extension_bonus: float = 0.05
```

### Signal Integration

```python
@dataclass
class SignalConfig:
    use_extension_duration: bool = True  # Enable extension predictor
    min_extension_candles: int = 2       # Require 2+ candles for signal
    extension_quality_threshold: float = 50
```

---

## Polygon.io Features Used

- **Aggregates** — Historical OHLCV bars (1m, 5m, 1H, 2H, Daily)
- **Previous Close** — Prior day's data
- **WebSocket** — Real-time minute bars (AM.* channel)
- **Bar-level VWAP** — Polygon provides VWAP per bar

---

## Quick Reference

```
┌──────────────────────────────────────────────────────────────┐
│           EXTENSION DURATION QUICK REFERENCE                  │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  CANDLES   HOURS   TRIGGER      SNAP-BACK     ACTION         │
│  ───────   ─────   ──────────   ──────────    ──────         │
│     1       2h     WATCHING       45%         Wait           │
│     2       4h     ALERT          55%         Prepare        │
│     3       6h     HIGH_PROB      65%         Look for entry │
│     4+      8h+    EXTREME        75%+        High conviction│
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  CONFIRMATION SIGNALS:                                        │
│  ✓ Rejection candle (long wick toward value)                 │
│  ✓ Declining volume in extension                             │
│  ✓ Multiple levels aligned (VWAP + POC + VAH all extended)   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Risk Disclaimer

This system is for educational purposes. Trading involves substantial risk. Always test in sandbox mode first. You are responsible for your own trading decisions.
