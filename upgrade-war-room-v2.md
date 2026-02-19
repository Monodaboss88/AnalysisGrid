# Task: Upgrade Daily War Room Script — V2 Enhancements

Upgrade the existing `war_room.py` (or whatever the file is named) pre-market intelligence script with the following improvements. Do NOT rewrite the entire file — patch each section surgically.

---

## 1. Fix HOD Timing — Use Last Touch, Not First

**Current (broken):**
```python
hod_row = rth.loc[rth['h'] == d_high].iloc[0]
```

**Replace with:**
```python
hod_row = rth.loc[rth['h'] == d_high].iloc[-1]  # Last touch = exhaustion timing
```

Also add LOD timing the same way:
```python
lod_row = rth.loc[rth['l'] == d_low].iloc[-1]
```

Add both to `daily_stats`:
```python
'hod_hour': hod_row['t'].hour + hod_row['t'].minute / 60,  # Decimal hour for precision
'lod_hour': lod_row['t'].hour + lod_row['t'].minute / 60,
```

---

## 2. Add Close Position Metric

After the existing core metrics block, add:

```python
d_close = rth.iloc[-1]['c']
d_range = d_high - d_low
close_position = ((d_close - d_low) / d_range) * 100 if d_range > 0 else 50
# 0 = closed at low (bearish), 100 = closed at high (bullish)
```

Add to `daily_stats`:
```python
'close_pos': close_position,
```

> ~~`close_vs_open` was removed — computed but never referenced in aggregation, output, or signals.~~

Add to aggregated results:
```python
'avg_close_pos': stats_df['close_pos'].mean(),
# BUG FIX: Only count reversals on days that actually extended up first,
# not days that gapped down and just kept selling (those are trend days, not reversals)
'reversal_pct': (stats_df.loc[stats_df['up_ext'] > stats_df['up_ext'].median(), 'close_pos'] < 30).mean() * 100,
```

---

## 3. Use Polygon Native VWAP When Available

Replace the manual VWAP calculation with a try/except that uses Polygon's `vw` field first:

```python
# VWAP — use Polygon native if available, else calculate
if 'vw' in rth.columns and rth['vw'].notna().any():
    d_vwap = (rth['vw'] * rth['v']).sum() / rth['v'].sum()  # Volume-weighted average of per-bar VWAPs
else:
    rth_calc = rth.copy()
    rth_calc['tp'] = (rth_calc['c'] + rth_calc['h'] + rth_calc['l']) / 3
    d_vwap = (rth_calc['tp'] * rth_calc['v']).sum() / rth_calc['v'].sum()
```

---

## 3B. Add VWAP-at-HOD (Fade Qualifier)

*Stolen from the 2026 script — captures the VWAP value at the exact moment of HOD. If the high was made well above VWAP, the extension is above fair value and more fadeable.*

After the VWAP calculation and after the HOD/LOD timing from Section 1, add:

```python
# VWAP at moment of HOD — fade qualifier
# If HOD was made above VWAP, extension is above fair value
# BUG FIX: Both branches now use CUMULATIVE VWAP (open → HOD) for consistency
try:
    hod_mask = rth['h'] == d_high
    hod_idx = rth.loc[hod_mask].index[-1]
    rth_to_hod = rth.loc[:hod_idx]
    
    if 'vw' in rth.columns and rth_to_hod['vw'].notna().any():
        # Cumulative VWAP from open to HOD using Polygon per-bar VWAPs
        vwap_at_hod = (rth_to_hod['vw'] * rth_to_hod['v']).sum() / rth_to_hod['v'].sum()
    else:
        # Fallback: use running VWAP up to that point
        tp = (rth_to_hod['c'] + rth_to_hod['h'] + rth_to_hod['l']) / 3
        vwap_at_hod = (tp * rth_to_hod['v']).sum() / rth_to_hod['v'].sum()
    
    hod_vs_vwap = ((d_high - vwap_at_hod) / vwap_at_hod) * 100  # % above VWAP at HOD
except Exception:
    vwap_at_hod = d_vwap
    hod_vs_vwap = 0
```

Add to `daily_stats`:
```python
'hod_vs_vwap': hod_vs_vwap,
```

Add to aggregated results:
```python
'avg_hod_vs_vwap': stats_df['hod_vs_vwap'].mean(),  # Avg % HOD extends above VWAP
'fade_zone_pct': (stats_df['hod_vs_vwap'] > 0.5).mean() * 100,  # % of days HOD was >0.5% above VWAP
```

This replaces the old `vwap_fade_zone` metric with something more precise — instead of measuring average distance from VWAP to high across the whole day, it captures the VWAP *at the moment the high was printed*. Much more actionable for fade entries.

---

## 4. Add Volume Profile at Extremes

After VWAP calculation, add volume concentration analysis:

```python
# Volume profile — where did volume concentrate?
# BUG FIX: Use typical price (h+l+c)/3 to assign each bar to ONE zone.
# Old code used h >= threshold and l <= threshold — wide-range bars got
# double-counted in both top and bottom, making mid_vol_pct go negative.
d_range = d_high - d_low
if d_range > 0 and rth['v'].sum() > 0:
    total_vol = rth['v'].sum()
    rth_tp = (rth['h'] + rth['l'] + rth['c']) / 3  # Typical price per bar
    
    # Volume in top 20% of range (near highs)
    top_threshold = d_high - (d_range * 0.2)
    top_vol = rth[rth_tp >= top_threshold]['v'].sum()
    top_vol_pct = (top_vol / total_vol) * 100
    
    # Volume in bottom 20% of range (near lows)
    bot_threshold = d_low + (d_range * 0.2)
    bot_vol = rth[rth_tp <= bot_threshold]['v'].sum()
    bot_vol_pct = (bot_vol / total_vol) * 100
    
    # Volume in middle 60% (value area proxy)
    mid_vol_pct = 100 - top_vol_pct - bot_vol_pct
    
    # Thin extension flag: high was made on low volume
    thin_top = top_vol_pct < 15  # Less than 15% of volume at highs = thin liquidity sweep
else:
    top_vol_pct = bot_vol_pct = mid_vol_pct = 0
    thin_top = False
```

Add to `daily_stats`:
```python
'top_vol_pct': top_vol_pct,
'bot_vol_pct': bot_vol_pct,
'thin_top': thin_top,
```

Add to aggregated results:
```python
'avg_top_vol': stats_df['top_vol_pct'].mean(),
'thin_top_pct': stats_df['thin_top'].mean() * 100,  # % of days with thin liquidity sweeps at highs
```

---

## 5. Add Regime Detection — Rolling Window Comparison

After building `stats_df`, add regime analysis:

```python
# Regime detection — is recent behavior different from baseline?
# BUG FIX: Guard changed from >= 20 to >= 30. With only 20 rows,
# head(20) and tail(10) overlap on 10 rows, contaminating the comparison.
# BUG FIX: Renamed 'vol_regime' → 'ext_regime' — it measures extension
# volatility (std of up_ext), NOT trading volume. Old name was misleading.
if len(stats_df) >= 30:
    recent = stats_df.tail(10)
    baseline = stats_df.head(20)
    
    regime = {
        'expanding': recent['up_ext'].mean() > baseline['up_ext'].mean() * 1.3,  # 30%+ increase
        'contracting': recent['up_ext'].mean() < baseline['up_ext'].mean() * 0.7,  # 30%+ decrease
        'up_shift': recent['up_ext'].mean() - baseline['up_ext'].mean(),
        'down_shift': recent['down_ext'].mean() - baseline['down_ext'].mean(),
        'ext_regime': 'HOT' if recent['up_ext'].std() > baseline['up_ext'].std() * 1.5 else 
                      'COLD' if recent['up_ext'].std() < baseline['up_ext'].std() * 0.6 else 'NORMAL'
    }
else:
    regime = {
        'expanding': False,
        'contracting': False,
        'up_shift': 0,
        'down_shift': 0,
        'ext_regime': 'NORMAL'
    }
```

Add to return dict:
```python
'regime': regime,
```

---

## 6. Add Adaptive Exhaustion Threshold

Replace the static exhaustion calculation in `run_morning_war_room()`:

**Current:**
```python
exhaustion = dna['avg_up'] + (1.5 * dna['std_up'])
```

**Replace with:**
```python
# Adaptive exhaustion — tighter in cold regimes, wider in hot
if dna['regime']['ext_regime'] == 'HOT':
    exhaust_mult = 2.0
elif dna['regime']['ext_regime'] == 'COLD':
    exhaust_mult = 1.0
else:
    exhaust_mult = 1.5

exhaustion = dna['avg_up'] + (exhaust_mult * dna['std_up'])
```

---

## 7. Upgrade the Output Table

Replace the existing print block with an expanded format:

```python
def run_morning_war_room():
    print(f"\n{'='*120}")
    print(f" DAILY WAR ROOM: {datetime.now().strftime('%Y-%m-%d')} | PRE-MARKET INTELLIGENCE")
    print(f"{'='*120}")
    
    # Header
    print(f"{'TICKER':<8} | {'AVG UP%':<8} | {'EXHAUST%':<9} | {'AVG DN%':<8} | "
          f"{'PEAK HR':<9} | {'HOD>VWAP':<9} | {'CLOSE POS':<9} | "
          f"{'THIN TOP':<9} | {'REGIME':<8} | SIGNAL")
    print("-" * 130)
    
    # ── First pass: collect all DNA results ──
    all_results = {}
    for ticker in WATCHLIST:
        all_results[ticker] = get_master_analysis(ticker)
    
    # ── Correlation pass: compute sync scores (Section 9) ──
    hod_hours = [dna['peak_hour'] for dna in all_results.values() if dna]
    spy_dna = get_master_analysis('SPY')  # One extra API call for benchmark
    
    for ticker, dna in all_results.items():
        if dna:
            # Adaptive exhaustion
            if dna['regime']['ext_regime'] == 'HOT':
                exhaust_mult = 2.0
            elif dna['regime']['ext_regime'] == 'COLD':
                exhaust_mult = 1.0
            else:
                exhaust_mult = 1.5
            exhaustion = dna['avg_up'] + (exhaust_mult * dna['std_up'])
            
            # ── Correlation signals (Section 9) ──
            # Benchmark-relative extension: strip out market-wide move
            isolated_ext = dna['avg_up'] - (spy_dna['avg_up'] if spy_dna else 0)
            
            # Sync score: % of watchlist that peaked in same hour window
            same_window = sum(1 for h in hod_hours if abs(h - dna['peak_hour']) <= 0.5)
            sync_score = (same_window / len(hod_hours) * 100) if hod_hours else 0
            
            is_isolated = isolated_ext > 0.3 and sync_score < 50
            
            # Generate signal
            signals = []
            if dna['thin_top_pct'] > 60:
                signals.append('THIN HIGHS')
            if dna['avg_close_pos'] < 35:
                signals.append('FADING')
            if dna['reversal_pct'] > 40:
                signals.append('REVERSAL PRONE')
            if dna['fade_zone_pct'] > 70:
                signals.append('HOD>VWAP')
            if dna['regime']['ext_regime'] == 'HOT':
                signals.append('⚠️ HOT')
            if dna['regime']['contracting']:
                signals.append('COMPRESSING')
            # Correlation flag
            if is_isolated:
                signals.append('🟢 ISOLATED')
            elif sync_score > 60:
                signals.append('🔴 SECTOR')
            signal_str = ' | '.join(signals) if signals else '—'
            
            # BUG FIX: peak_hour is decimal (10.5 = 10:30), format properly
            ph = dna['peak_hour']
            peak_str = f"{int(ph):>2}:{int((ph % 1) * 60):02d} ET"
            
            print(f"{ticker:<8} | {dna['avg_up']:>6.2f}% | {exhaustion:>7.2f}% | {dna['avg_down']:>6.2f}% | "
                  f"{peak_str:<9} | {dna['avg_hod_vs_vwap']:>7.2f}% | "
                  f"{dna['avg_close_pos']:>7.1f}%  | "
                  f"{dna['thin_top_pct']:>6.1f}%  | "
                  f"{dna['regime']['ext_regime']:<8} | {signal_str}")
        else:
            print(f"{ticker:<8} | ERROR FETCHING DATA")
    
    print(f"{'='*120}")
    print("STRATEGY RULES:")
    print("  • EXHAUST% = Adaptive (tighter in COLD, wider in HOT regimes)")
    print("  • THIN HIGHS = >60% of days had <15% volume at highs → liquidity sweep, fadeable")
    print("  • FADING = Avg close position <35% of range → stocks closing weak despite extensions")
    print("  • REVERSAL PRONE = >40% of up-extension days close in bottom 30% of range → high reversion")
    print("  • HOD>VWAP = >70% of days HOD printed >0.5% above VWAP → extensions overshoot fair value")
    print("  • 🟢 ISOLATED = Extension is ticker-specific (>0.3% above SPY, <50% watchlist sync)")
    print("  • 🔴 SECTOR = >60% of watchlist peaked in same window → correlated move, lower fade conviction")
    print("  • Highest-Prob Fade: EXHAUST% + PEAK HR + THIN HIGHS + HOD>VWAP + FADING + 🟢 ISOLATED")
    print(f"{'='*120}")
```

---

## 8. Update Aggregated Results Return

The full return dict in `get_master_analysis()` should now be:

```python
return {
    # Extension DNA
    'avg_up': stats_df['up_ext'].mean(),
    'std_up': stats_df['up_ext'].std(),
    'avg_down': stats_df['down_ext'].mean(),
    'std_down': stats_df['down_ext'].std(),
    
    # Timing — BUG FIX: round to 30-min buckets before mode()
    # Raw decimal hours (10.517, 10.533, etc.) are nearly unique → mode() is useless on continuous data
    'peak_hour': ((stats_df['hod_hour'] * 2).round() / 2).mode()[0],
    'lod_hour': ((stats_df['lod_hour'] * 2).round() / 2).mode()[0],
    
    # VWAP-at-HOD (replaces old vwap_fade_zone)
    'avg_hod_vs_vwap': stats_df['hod_vs_vwap'].mean(),
    'fade_zone_pct': (stats_df['hod_vs_vwap'] > 0.5).mean() * 100,
    
    # Close behavior — BUG FIX: reversal_pct only counts days that extended up first
    'avg_close_pos': stats_df['close_pos'].mean(),
    'reversal_pct': (stats_df.loc[stats_df['up_ext'] > stats_df['up_ext'].median(), 'close_pos'] < 30).mean() * 100,
    
    # Volume profile
    'avg_top_vol': stats_df['top_vol_pct'].mean(),
    'thin_top_pct': stats_df['thin_top'].mean() * 100,
    
    # Regime
    'regime': regime,
}
```

---

## Summary of Changes

| # | Enhancement | What It Fixes |
|---|------------|---------------|
| 1 | HOD last touch + LOD timing | First touch ≠ exhaustion; last touch = rejection point |
| 2 | Close position metric | Distinguishes trend days from reversal days |
| 3 | Native Polygon VWAP | More accurate on paid tiers, fallback for free |
| 3B | VWAP-at-HOD fade qualifier | Captures if HOD was above fair value — direct fade signal |
| 4 | Volume at extremes | Identifies thin liquidity sweeps vs. real breakouts |
| 5 | Regime detection | Prevents stale thresholds after earnings/catalysts |
| 6 | Adaptive exhaustion | Tighter stops in quiet markets, wider in volatile |
| 7 | Expanded output | Adds signals column with actionable flags |
| 8 | Full return dict | All new metrics available for downstream use |
| 9 | Multi-ticker correlation | Distinguishes isolated extensions from sector-wide risk-on moves |

---

## 9. Multi-Ticker Correlation Check

*Prevents fading stocks in a correlated sector bid. If the whole watchlist extended together, it's a market move — not an isolated overextension.*

### Layer 1 — Benchmark-Relative Extension

Fetch SPY once per run (not per ticker). Compare each ticker's avg extension to SPY's:

```python
# In run_morning_war_room(), BEFORE the ticker loop:
spy_dna = get_master_analysis('SPY')  # Single extra API call

# Inside the per-ticker loop:
isolated_ext = dna['avg_up'] - (spy_dna['avg_up'] if spy_dna else 0)
# If AAPL avg_up is 1.5% but SPY avg_up is 1.2%, isolated_ext = 0.3% — barely anything
```

### Layer 2 — Watchlist Peak Sync Score

After processing all tickers, score how many peaked in the same 30-min window:

```python
# After collecting all DNA results:
hod_hours = [dna['peak_hour'] for dna in all_results.values() if dna]

# Per ticker:
same_window = sum(1 for h in hod_hours if abs(h - dna['peak_hour']) <= 0.5)
sync_score = (same_window / len(hod_hours)) * 100  # % of watchlist peaking together
```

### Signal Integration

```python
is_isolated = isolated_ext > 0.3 and sync_score < 50

if is_isolated:
    signals.append('🟢 ISOLATED')    # Ticker-specific extension → high fade conviction
elif sync_score > 60:
    signals.append('🔴 SECTOR')       # Correlated move → lower fade conviction
```

### Architecture Note

This requires a **two-pass structure** in `run_morning_war_room()`:
1. **First pass:** Run `get_master_analysis()` for all tickers + SPY, store results
2. **Second pass:** Compute sync scores and print output with correlation flags

The Section 7 output table has been updated to use this two-pass pattern.

---

## Important Notes

- Do NOT change the Polygon API call structure — `client.get_aggs()` stays the same
- The `daily_stats` list gets new fields added — make sure ALL fields are present in EVERY iteration (use defaults if calculation fails)
- Wrap volume profile calcs in try/except — some thin pre-market bars may have zero volume
- The `stats_df.tail(30)` stays the same — regime detection uses slices within that 30-day window
- `peak_hour` is decimal (e.g., 10.5 = 10:30 AM) — rounded to 30-min buckets for mode(), formatted as `f"{int(ph):>2}:{int((ph % 1) * 60):02d} ET"` in output
- All `vol_regime` references have been renamed to `ext_regime` — it measures extension volatility, not trading volume
- Section 9 adds ONE extra API call (SPY) — no per-ticker overhead
