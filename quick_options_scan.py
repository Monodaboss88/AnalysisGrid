"""
Quick Options Scanner
====================
Scans watchlist for best repositioning opportunities.
Finds compression reversal setups with good options parameters.

Run: python quick_options_scan.py
"""

from polygon_data import get_bars
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# Watchlist to scan - high liquidity options names
WATCHLIST = [
    # Mega caps with liquid options
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Other liquid names
    "AMD", "NFLX", "SPY", "QQQ", "IWM",
    "BA", "DIS", "JPM", "V", "MA", 
    "CRM", "ORCL", "INTC", "MU",
    "XOM", "CVX", "SLB",
    "COIN", "MARA", "RIOT"
]

def get_data(symbol: str, period: str = "30d", interval: str = "1h") -> pd.DataFrame:
    """Fetch OHLCV data"""
    try:
        df = get_bars(symbol, period=period, interval=interval)
        return df
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return pd.DataFrame()

def calculate_compression(df: pd.DataFrame) -> Dict:
    """Calculate compression metrics"""
    if len(df) < 20:
        return None
    
    # ATR calculation
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    atr_14 = tr.rolling(14).mean().iloc[-1]
    atr_5 = tr.rolling(5).mean().iloc[-1]
    atr_30 = tr.rolling(30).mean().iloc[-1] if len(df) >= 30 else tr.mean()
    
    compression_ratio = atr_5 / atr_30 if atr_30 > 0 else 1.0
    
    # Bollinger Band width
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    bb_width = ((std20 * 2) / sma20 * 100).iloc[-1]
    
    # 5-day range
    range_5d = (df['High'].rolling(5).max() - df['Low'].rolling(5).min()).iloc[-1]
    range_5d_pct = (range_5d / df['Close'].iloc[-1]) * 100
    
    return {
        'atr_14': atr_14,
        'atr_5': atr_5,
        'compression_ratio': compression_ratio,
        'bb_width': bb_width,
        'range_5d_pct': range_5d_pct,
        'squeezed': compression_ratio < 0.7
    }

def calculate_volume_profile(df: pd.DataFrame) -> Dict:
    """Calculate simple volume profile levels"""
    if len(df) < 20:
        return None
    
    # Use last 20 bars
    recent = df.tail(20)
    
    price_range = recent['High'].max() - recent['Low'].min()
    bin_size = price_range / 20
    
    # Build volume by price
    vbp = {}
    for _, row in recent.iterrows():
        mid = (row['High'] + row['Low']) / 2
        bin_key = round(mid / bin_size) * bin_size
        vbp[bin_key] = vbp.get(bin_key, 0) + row['Volume']
    
    # POC = price with most volume
    poc = max(vbp, key=vbp.get) if vbp else recent['Close'].mean()
    
    # Value area (70% of volume)
    sorted_vbp = sorted(vbp.items(), key=lambda x: x[1], reverse=True)
    total_vol = sum(vbp.values())
    cum_vol = 0
    va_prices = []
    for price, vol in sorted_vbp:
        va_prices.append(price)
        cum_vol += vol
        if cum_vol >= total_vol * 0.7:
            break
    
    vah = max(va_prices) if va_prices else recent['High'].max()
    val = min(va_prices) if va_prices else recent['Low'].min()
    
    return {
        'poc': poc,
        'vah': vah,
        'val': val
    }

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate RSI"""
    if len(df) < period + 1:
        return 50
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def scan_symbol(symbol: str) -> Dict:
    """Scan a single symbol for setup quality"""
    df = get_data(symbol)
    if df.empty or len(df) < 20:
        return None
    
    current_price = df['Close'].iloc[-1]
    
    # Calculate metrics
    compression = calculate_compression(df)
    vp = calculate_volume_profile(df)
    rsi = calculate_rsi(df)
    
    if not compression or not vp:
        return None
    
    # Distance from key levels
    dist_to_val = ((current_price - vp['val']) / vp['val']) * 100
    dist_to_poc = ((current_price - vp['poc']) / vp['poc']) * 100
    dist_to_vah = ((current_price - vp['vah']) / vp['vah']) * 100
    
    # Score the setup
    score = 50  # Base score
    direction = "NEUTRAL"
    setup_type = None
    
    # BULLISH SETUP: Price at/below VAL, RSI oversold, compressed
    if dist_to_val <= 0.5:  # At or below VAL
        score += 15
        if rsi < 40:
            score += 15
            if rsi < 35:
                score += 10
        if compression['squeezed']:
            score += 10
        if compression['compression_ratio'] < 0.5:
            score += 10
        direction = "BULLISH"
        setup_type = "Compression Reversal - CALLS"
    
    # BEARISH SETUP: Price at/above VAH, RSI overbought
    elif dist_to_vah >= -0.5:  # At or above VAH
        score += 15
        if rsi > 60:
            score += 15
            if rsi > 65:
                score += 10
        if compression['squeezed']:
            score += 10
        direction = "BEARISH"
        setup_type = "Mean Reversion - PUTS"
    
    # Near POC - wait for direction
    elif abs(dist_to_poc) < 1:
        setup_type = "At POC - Wait for Break"
        direction = "NEUTRAL"
    
    # Calculate entry levels
    if direction == "BULLISH":
        entry_zone = (vp['val'] * 0.995, vp['val'] * 1.005)
        stop = vp['val'] - (compression['atr_14'] * 1.5)
        target1 = vp['poc']
        target2 = vp['vah']
    elif direction == "BEARISH":
        entry_zone = (vp['vah'] * 0.995, vp['vah'] * 1.005)
        stop = vp['vah'] + (compression['atr_14'] * 1.5)
        target1 = vp['poc']
        target2 = vp['val']
    else:
        entry_zone = (current_price * 0.99, current_price * 1.01)
        stop = current_price - compression['atr_14'] if compression else current_price * 0.97
        target1 = vp['poc']
        target2 = vp['vah']
    
    # Price move to target
    if direction == "BULLISH":
        move_to_t1 = ((target1 - current_price) / current_price) * 100
        move_to_t2 = ((target2 - current_price) / current_price) * 100
    else:
        move_to_t1 = ((current_price - target1) / current_price) * 100
        move_to_t2 = ((current_price - target2) / current_price) * 100
    
    return {
        'symbol': symbol,
        'price': round(current_price, 2),
        'direction': direction,
        'setup_type': setup_type,
        'score': min(100, max(0, score)),
        'rsi': round(rsi, 1),
        'compression_ratio': round(compression['compression_ratio'], 2),
        'squeezed': compression['squeezed'],
        'dist_to_val': round(dist_to_val, 2),
        'dist_to_poc': round(dist_to_poc, 2),
        'dist_to_vah': round(dist_to_vah, 2),
        'val': round(vp['val'], 2),
        'poc': round(vp['poc'], 2),
        'vah': round(vp['vah'], 2),
        'entry_zone': (round(entry_zone[0], 2), round(entry_zone[1], 2)),
        'stop': round(stop, 2),
        'target1': round(target1, 2),
        'target2': round(target2, 2),
        'move_to_t1': round(move_to_t1, 1),
        'move_to_t2': round(move_to_t2, 1),
        'atr': round(compression['atr_14'], 2)
    }

def run_scan():
    """Run scan on full watchlist"""
    print("=" * 70)
    print("OPTIONS REPOSITIONING SCANNER")
    print(f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print()
    
    results = []
    
    print("Scanning watchlist...")
    for symbol in WATCHLIST:
        try:
            result = scan_symbol(symbol)
            if result and result['direction'] != "NEUTRAL":
                results.append(result)
                status = "✓" if result['score'] >= 65 else "○"
                print(f"  {status} {symbol}: {result['direction']} | Score: {result['score']} | RSI: {result['rsi']}")
            else:
                print(f"  - {symbol}: No setup")
        except Exception as e:
            print(f"  ✗ {symbol}: Error - {e}")
    
    # Sort by score
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # Print top opportunities
    print()
    print("=" * 70)
    print("TOP OPPORTUNITIES")
    print("=" * 70)
    
    bullish = [r for r in results if r['direction'] == 'BULLISH' and r['score'] >= 60]
    bearish = [r for r in results if r['direction'] == 'BEARISH' and r['score'] >= 60]
    
    print("\n🟢 BULLISH SETUPS (Buy Calls):")
    print("-" * 70)
    if bullish:
        for r in bullish[:5]:
            print(f"\n{r['symbol']} - Score: {r['score']}")
            print(f"  Price: ${r['price']} | RSI: {r['rsi']} | Compressed: {r['squeezed']}")
            print(f"  Position: {r['dist_to_val']:.1f}% from VAL (${r['val']})")
            print(f"  Levels: VAL ${r['val']} → POC ${r['poc']} → VAH ${r['vah']}")
            print(f"  Trade: Entry zone ${r['entry_zone'][0]}-${r['entry_zone'][1]}")
            print(f"         Stop ${r['stop']} | T1 ${r['target1']} (+{r['move_to_t1']}%) | T2 ${r['target2']} (+{r['move_to_t2']}%)")
            
            # Options recommendation
            delta = 0.65 if r['score'] >= 75 else 0.55
            dte = "3+ weeks" if r['score'] >= 70 else "4+ weeks"
            print(f"  📞 CALLS: {delta} delta, {dte} out")
    else:
        print("  No bullish setups meeting criteria")
    
    print("\n🔴 BEARISH SETUPS (Buy Puts):")
    print("-" * 70)
    if bearish:
        for r in bearish[:5]:
            print(f"\n{r['symbol']} - Score: {r['score']}")
            print(f"  Price: ${r['price']} | RSI: {r['rsi']} | Compressed: {r['squeezed']}")
            print(f"  Position: {r['dist_to_vah']:.1f}% from VAH (${r['vah']})")
            print(f"  Levels: VAH ${r['vah']} → POC ${r['poc']} → VAL ${r['val']}")
            print(f"  Trade: Entry zone ${r['entry_zone'][0]}-${r['entry_zone'][1]}")
            print(f"         Stop ${r['stop']} | T1 ${r['target1']} (+{r['move_to_t1']}%) | T2 ${r['target2']} (+{r['move_to_t2']}%)")
            
            delta = -0.65 if r['score'] >= 75 else -0.55
            dte = "3+ weeks" if r['score'] >= 70 else "4+ weeks"
            print(f"  📉 PUTS: {abs(delta)} delta, {dte} out")
    else:
        print("  No bearish setups meeting criteria")
    
    print()
    print("=" * 70)
    print("NOTES:")
    print("- Higher scores = better setups (>70 = A grade, >80 = A+)")
    print("- Look for RSI < 35 for calls, RSI > 65 for puts")
    print("- 'Squeezed' = low volatility, expect big move")
    print("- Always check options IV before entering")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    run_scan()
