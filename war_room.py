"""
War Room — Pre-Market Extension DNA Scanner
=============================================
Analyzes 30-60 days of intraday data per ticker via Polygon.io to compute
extension DNA, exhaustion thresholds, volume profiles, regime detection,
and multi-ticker correlation.

Sections implemented from upgrade-war-room-v2.md (audited + bug-fixed):
  1. HOD/LOD timing (last touch)
  2. Close position metric (reversal filter)
  3. Native Polygon VWAP
  3B. VWAP-at-HOD (fade qualifier, cumulative both branches)
  4. Volume profile at extremes (typical-price zone assignment)
  5. Regime detection (ext_regime, >= 30 guard)
  6. Adaptive exhaustion
  7/8. Full return dict
  9. Multi-ticker correlation (SPY benchmark + sync score)

Requires: POLYGON_API_KEY env var, pandas, requests
"""

import os
import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE = "https://api.polygon.io"
_pool = ThreadPoolExecutor(max_workers=6)


# ────────────────────────────────
#  Polygon Aggregates Fetcher
# ────────────────────────────────

def _get_key() -> str:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise RuntimeError("POLYGON_API_KEY not set")
    return key


def _fetch_aggs(ticker: str, from_date: str, to_date: str,
                timespan: str = "minute", multiplier: int = 5) -> pd.DataFrame:
    """Fetch aggregated bars from Polygon REST API."""
    key = _get_key()
    url = (f"{BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}"
           f"/{from_date}/{to_date}?adjusted=true&sort=asc&limit=50000&apiKey={key}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    # Polygon fields: o, h, l, c, v, vw, t, n
    df['t'] = pd.to_datetime(df['t'], unit='ms', utc=True)
    df['t'] = df['t'].dt.tz_convert('America/New_York')
    return df


def _rth_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Regular Trading Hours: 9:30–16:00 ET."""
    if df.empty:
        return df
    mask = (df['t'].dt.hour * 60 + df['t'].dt.minute >= 570) & \
           (df['t'].dt.hour * 60 + df['t'].dt.minute < 960)
    return df[mask].copy()


# ────────────────────────────────
#  Per-Day Analysis
# ────────────────────────────────

def _analyze_day(day_df: pd.DataFrame, d_open: float) -> Optional[Dict]:
    """Analyze a single trading day's bars. Returns daily_stats dict or None."""
    rth = _rth_filter(day_df)
    if len(rth) < 10:  # Need at least ~50 min of data
        return None

    try:
        d_high = rth['h'].max()
        d_low = rth['l'].min()
        d_close = rth.iloc[-1]['c']
        d_range = d_high - d_low
        if d_range <= 0 or d_open <= 0:
            return None

        # ── Section 1: HOD/LOD timing — last touch ──
        hod_row = rth.loc[rth['h'] == d_high].iloc[-1]
        lod_row = rth.loc[rth['l'] == d_low].iloc[-1]
        hod_hour = hod_row['t'].hour + hod_row['t'].minute / 60
        lod_hour = lod_row['t'].hour + lod_row['t'].minute / 60

        # Extension from open
        up_ext = ((d_high - d_open) / d_open) * 100
        down_ext = ((d_open - d_low) / d_open) * 100

        # ── Section 2: Close position ──
        close_position = ((d_close - d_low) / d_range) * 100

        # ── Section 3: VWAP ──
        if 'vw' in rth.columns and rth['vw'].notna().any() and rth['v'].sum() > 0:
            d_vwap = (rth['vw'] * rth['v']).sum() / rth['v'].sum()
        else:
            tp = (rth['c'] + rth['h'] + rth['l']) / 3
            d_vwap = (tp * rth['v']).sum() / rth['v'].sum() if rth['v'].sum() > 0 else d_close

        # ── Section 3B: VWAP-at-HOD (cumulative both branches) ──
        try:
            hod_mask = rth['h'] == d_high
            hod_idx = rth.loc[hod_mask].index[-1]
            rth_to_hod = rth.loc[:hod_idx]
            if rth_to_hod['v'].sum() > 0:
                if 'vw' in rth_to_hod.columns and rth_to_hod['vw'].notna().any():
                    vwap_at_hod = (rth_to_hod['vw'] * rth_to_hod['v']).sum() / rth_to_hod['v'].sum()
                else:
                    tp_h = (rth_to_hod['c'] + rth_to_hod['h'] + rth_to_hod['l']) / 3
                    vwap_at_hod = (tp_h * rth_to_hod['v']).sum() / rth_to_hod['v'].sum()
                hod_vs_vwap = ((d_high - vwap_at_hod) / vwap_at_hod) * 100
            else:
                vwap_at_hod = d_vwap
                hod_vs_vwap = 0
        except Exception:
            vwap_at_hod = d_vwap
            hod_vs_vwap = 0

        # ── Section 4: Volume profile at extremes (typical-price assignment) ──
        try:
            total_vol = rth['v'].sum()
            if d_range > 0 and total_vol > 0:
                rth_tp = (rth['h'] + rth['l'] + rth['c']) / 3
                top_threshold = d_high - (d_range * 0.2)
                bot_threshold = d_low + (d_range * 0.2)
                top_vol_pct = (rth[rth_tp >= top_threshold]['v'].sum() / total_vol) * 100
                bot_vol_pct = (rth[rth_tp <= bot_threshold]['v'].sum() / total_vol) * 100
                thin_top = top_vol_pct < 15
            else:
                top_vol_pct = bot_vol_pct = 0
                thin_top = False
        except Exception:
            top_vol_pct = bot_vol_pct = 0
            thin_top = False

        return {
            'up_ext': up_ext,
            'down_ext': down_ext,
            'hod_hour': hod_hour,
            'lod_hour': lod_hour,
            'close_pos': close_position,
            'hod_vs_vwap': hod_vs_vwap,
            'top_vol_pct': top_vol_pct,
            'bot_vol_pct': bot_vol_pct,
            'thin_top': thin_top,
            'vwap': d_vwap,
            'high': d_high,
            'low': d_low,
            'close': d_close,
            'open': d_open,
            'range': d_range,
        }
    except Exception as e:
        logger.warning(f"Day analysis failed: {e}")
        return None


# ────────────────────────────────
#  Master Analysis (single ticker)
# ────────────────────────────────

def get_master_analysis(ticker: str, lookback_days: int = 60) -> Optional[Dict]:
    """
    Fetch Polygon bars and compute extension DNA, volume profiles,
    regime detection, and all V2 metrics for a single ticker.
    Returns aggregated dict or None.
    """
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        from_str = start.strftime("%Y-%m-%d")
        to_str = end.strftime("%Y-%m-%d")

        df = _fetch_aggs(ticker, from_str, to_str, timespan="minute", multiplier=5)
        if df.empty:
            return None

        # Group by trading date
        df['date'] = df['t'].dt.date
        daily_groups = df.groupby('date')

        stats_list = []
        for dt, group in daily_groups:
            # Use first RTH bar's open as the day open
            rth = _rth_filter(group)
            if rth.empty:
                continue
            d_open = rth.iloc[0]['o']
            result = _analyze_day(group, d_open)
            if result:
                stats_list.append(result)

        if len(stats_list) < 10:
            return None

        stats_df = pd.DataFrame(stats_list).tail(45)  # Use last 45 trading days max

        # ── Section 5: Regime detection ──
        if len(stats_df) >= 30:
            recent = stats_df.tail(10)
            baseline = stats_df.head(20)
            regime = {
                'expanding': bool(recent['up_ext'].mean() > baseline['up_ext'].mean() * 1.3),
                'contracting': bool(recent['up_ext'].mean() < baseline['up_ext'].mean() * 0.7),
                'up_shift': round(recent['up_ext'].mean() - baseline['up_ext'].mean(), 4),
                'down_shift': round(recent['down_ext'].mean() - baseline['down_ext'].mean(), 4),
                'ext_regime': ('HOT' if recent['up_ext'].std() > baseline['up_ext'].std() * 1.5 else
                               'COLD' if recent['up_ext'].std() < baseline['up_ext'].std() * 0.6 else
                               'NORMAL')
            }
        else:
            regime = {
                'expanding': False,
                'contracting': False,
                'up_shift': 0,
                'down_shift': 0,
                'ext_regime': 'NORMAL'
            }

        # ── Section 8: Aggregated return dict ──
        # Timing: round to 30-min buckets before mode()
        hod_buckets = ((stats_df['hod_hour'] * 2).round() / 2)
        lod_buckets = ((stats_df['lod_hour'] * 2).round() / 2)

        # reversal_pct: only count days that extended up first
        up_days = stats_df[stats_df['up_ext'] > stats_df['up_ext'].median()]
        if len(up_days) > 0:
            reversal_pct = (up_days['close_pos'] < 30).mean() * 100
        else:
            reversal_pct = 0

        return {
            'ticker': ticker,
            # Extension DNA
            'avg_up': round(stats_df['up_ext'].mean(), 4),
            'std_up': round(stats_df['up_ext'].std(), 4),
            'avg_down': round(stats_df['down_ext'].mean(), 4),
            'std_down': round(stats_df['down_ext'].std(), 4),
            'median_up': round(stats_df['up_ext'].median(), 4),
            'median_down': round(stats_df['down_ext'].median(), 4),
            # Timing
            'peak_hour': float(hod_buckets.mode()[0]) if not hod_buckets.mode().empty else 10.5,
            'lod_hour': float(lod_buckets.mode()[0]) if not lod_buckets.mode().empty else 14.0,
            # VWAP-at-HOD
            'avg_hod_vs_vwap': round(stats_df['hod_vs_vwap'].mean(), 4),
            'fade_zone_pct': round((stats_df['hod_vs_vwap'] > 0.5).mean() * 100, 1),
            # Close behavior
            'avg_close_pos': round(stats_df['close_pos'].mean(), 1),
            'reversal_pct': round(reversal_pct, 1),
            # Volume profile
            'avg_top_vol': round(stats_df['top_vol_pct'].mean(), 1),
            'thin_top_pct': round(stats_df['thin_top'].mean() * 100, 1),
            # Regime
            'regime': regime,
            # Meta
            'days_analyzed': len(stats_df),
        }

    except Exception as e:
        logger.error(f"War Room analysis failed for {ticker}: {e}")
        return None


# ────────────────────────────────
#  Full War Room Scan (multi-ticker + correlation)
# ────────────────────────────────

def _compute_signals(dna: Dict, spy_dna: Optional[Dict], hod_hours: List[float]) -> Dict:
    """Compute exhaustion threshold, correlation flags, and signal list."""
    regime = dna.get('regime', {})
    ext_regime = regime.get('ext_regime', 'NORMAL')

    # Adaptive exhaustion (Section 6)
    exhaust_mult = 2.0 if ext_regime == 'HOT' else (1.0 if ext_regime == 'COLD' else 1.5)
    exhaustion = dna['avg_up'] + (exhaust_mult * dna['std_up'])

    # Correlation (Section 9)
    isolated_ext = dna['avg_up'] - (spy_dna['avg_up'] if spy_dna else 0)
    same_window = sum(1 for h in hod_hours if abs(h - dna['peak_hour']) <= 0.5) if hod_hours else 0
    sync_score = (same_window / len(hod_hours) * 100) if hod_hours else 0
    is_isolated = isolated_ext > 0.3 and sync_score < 50

    # Signals
    signals = []
    if dna['thin_top_pct'] > 60:
        signals.append('THIN HIGHS')
    if dna['avg_close_pos'] < 35:
        signals.append('FADING')
    if dna['reversal_pct'] > 40:
        signals.append('REVERSAL PRONE')
    if dna['fade_zone_pct'] > 70:
        signals.append('HOD>VWAP')
    if ext_regime == 'HOT':
        signals.append('HOT')
    if regime.get('contracting'):
        signals.append('COMPRESSING')
    if is_isolated:
        signals.append('ISOLATED')
    elif sync_score > 60:
        signals.append('SECTOR')

    # Conviction score: how many fade signals fire (0-100)
    fade_flags = ['THIN HIGHS', 'FADING', 'REVERSAL PRONE', 'HOD>VWAP', 'ISOLATED']
    fade_count = sum(1 for s in signals if s in fade_flags)
    fade_conviction = min(100, int(fade_count / len(fade_flags) * 100))

    return {
        'exhaustion': round(exhaustion, 4),
        'exhaust_mult': exhaust_mult,
        'isolated_ext': round(isolated_ext, 4),
        'sync_score': round(sync_score, 1),
        'is_isolated': is_isolated,
        'signals': signals,
        'fade_conviction': fade_conviction,
    }


def run_war_room(tickers: List[str]) -> Dict:
    """
    Run full War Room scan for a list of tickers.
    Returns JSON-serializable dict with results + meta.
    """
    errors = []
    all_dna = {}

    # First pass: gather DNA for all tickers + SPY benchmark
    spy_dna = None
    try:
        spy_dna = get_master_analysis('SPY')
    except Exception as e:
        logger.warning(f"SPY benchmark failed: {e}")

    for t in tickers:
        try:
            dna = get_master_analysis(t)
            if dna:
                all_dna[t] = dna
            else:
                errors.append({'ticker': t, 'error': 'Insufficient data'})
        except Exception as e:
            errors.append({'ticker': t, 'error': str(e)})

    if not all_dna:
        return {'results': [], 'meta': {}, 'errors': errors}

    # Collect peak hours for sync scoring
    hod_hours = [d['peak_hour'] for d in all_dna.values()]

    # Second pass: compute signals with correlation context
    results = []
    for t, dna in all_dna.items():
        sig = _compute_signals(dna, spy_dna, hod_hours)
        row = {**dna, **sig}
        results.append(row)

    # Sort by fade_conviction desc, then exhaustion desc
    results.sort(key=lambda r: (r['fade_conviction'], r['exhaustion']), reverse=True)

    # Meta
    avg_up = sum(r['avg_up'] for r in results) / len(results)
    meta = {
        'scanned': len(results),
        'failed': len(errors),
        'spy_avg_up': round(spy_dna['avg_up'], 2) if spy_dna else None,
        'avg_watchlist_up': round(avg_up, 2),
        'hot_count': sum(1 for r in results if r['regime']['ext_regime'] == 'HOT'),
        'cold_count': sum(1 for r in results if r['regime']['ext_regime'] == 'COLD'),
        'scan_time': datetime.now(timezone.utc).isoformat(),
    }

    return {'results': results, 'meta': meta, 'errors': errors}


# ── Async wrapper for FastAPI ──

async def async_run_war_room(tickers: List[str]) -> Dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_pool, run_war_room, tickers)


# ── Preset Watchlists ──

PRESETS = {
    'mega_tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'CRM', 'NFLX'],
    'semis': ['NVDA', 'AMD', 'AVGO', 'QCOM', 'INTC', 'MU', 'TSM', 'MRVL', 'LRCX', 'KLAC'],
    'finance': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'SCHW', 'AXP', 'USB'],
    'momentum': ['PLTR', 'SMCI', 'MSTR', 'COIN', 'RKLB', 'APP', 'HOOD', 'AFRM', 'IONQ', 'RDDT'],
    'mag7': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
    'etfs': ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'ARKK', 'SOXL', 'TQQQ'],
}
