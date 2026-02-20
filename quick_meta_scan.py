"""Quick META Scan using Polygon"""
from polygon_data import get_bars
import pandas as pd
import numpy as np
from datetime import datetime

symbol = 'META'
print(f'Scanning {symbol}...')
print('='*60)

# Get data
df_1h = get_bars(symbol, period='30d', interval='1h')
df_daily = get_bars(symbol, period='90d', interval='1d')

if df_1h.empty:
    print('No data!')
    exit()

# Rename columns
for df in [df_1h, df_daily]:
    df.columns = [c.lower() for c in df.columns]

# Current price
current = df_1h['close'].iloc[-1]
prev_close = df_daily['close'].iloc[-2] if len(df_daily) > 1 else current
change = ((current - prev_close) / prev_close) * 100

print(f'PRICE: ${current:.2f} ({change:+.2f}%)')
day_low = df_1h['low'].iloc[-8:].min()
day_high = df_1h['high'].iloc[-8:].max()
print(f'Day Range: ${day_low:.2f} - ${day_high:.2f}')
print()

# Volume Profile
def calc_vp(df, va_pct=0.70):
    price_min, price_max = df['low'].min(), df['high'].max()
    bins = np.linspace(price_min, price_max, 50)
    vp = np.zeros(len(bins)-1)
    for _, row in df.iterrows():
        for i in range(len(bins)-1):
            overlap_low = max(row['low'], bins[i])
            overlap_high = min(row['high'], bins[i+1])
            if overlap_high > overlap_low:
                bar_range = row['high'] - row['low']
                if bar_range > 0:
                    vp[i] += row['volume'] * (overlap_high - overlap_low) / bar_range
    poc_idx = np.argmax(vp)
    poc = (bins[poc_idx] + bins[poc_idx+1]) / 2
    
    # Value Area
    total_vol = vp.sum()
    target = total_vol * va_pct
    va_vol = vp[poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx
    while va_vol < target:
        lo_add = vp[lo_idx-1] if lo_idx > 0 else 0
        hi_add = vp[hi_idx+1] if hi_idx < len(vp)-1 else 0
        if lo_add >= hi_add and lo_idx > 0:
            lo_idx -= 1
            va_vol += lo_add
        elif hi_idx < len(vp)-1:
            hi_idx += 1
            va_vol += hi_add
        else:
            break
    return round(poc,2), round(bins[hi_idx+1],2), round(bins[lo_idx],2)

# VWAP
def calc_vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    return round((tp * df['volume']).sum() / df['volume'].sum(), 2)

# RSI
def calc_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 0.0001)
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 1)

# Relative Volume
def calc_rvol(df, lookback=20):
    if len(df) < lookback + 1:
        return 1.0
    avg_vol = df['volume'].iloc[-(lookback+1):-1].mean()
    curr_vol = df['volume'].iloc[-1]
    return round(curr_vol / avg_vol, 2) if avg_vol > 0 else 1.0

# ATR
def calc_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    return round(tr.rolling(period).mean().iloc[-1], 2)

# Calculate for multiple timeframes
print('VOLUME PROFILE LEVELS')
print('-'*40)

# 1HR analysis (last 30 bars)
df_30 = df_1h.tail(30)
poc_1h, vah_1h, val_1h = calc_vp(df_30)
vwap_1h = calc_vwap(df_30)
rsi_1h = calc_rsi(df_1h)
rvol_1h = calc_rvol(df_1h)
atr_1h = calc_atr(df_1h)

print(f'1HR Timeframe (30 bars):')
print(f'  VAH: ${vah_1h:.2f}')
print(f'  POC: ${poc_1h:.2f}')
print(f'  VAL: ${val_1h:.2f}')
print(f'  VWAP: ${vwap_1h:.2f}')
print(f'  RSI: {rsi_1h}')
print(f'  RVOL: {rvol_1h}x')
print(f'  ATR: ${atr_1h}')
print()

# Daily analysis
poc_d, vah_d, val_d = calc_vp(df_daily.tail(30))
vwap_d = calc_vwap(df_daily.tail(30))
rsi_d = calc_rsi(df_daily)
atr_d = calc_atr(df_daily)

print(f'DAILY Timeframe (30 bars):')
print(f'  VAH: ${vah_d:.2f}')
print(f'  POC: ${poc_d:.2f}')
print(f'  VAL: ${val_d:.2f}')
print(f'  VWAP: ${vwap_d:.2f}')
print(f'  RSI: {rsi_d}')
print(f'  ATR: ${atr_d}')
print()

# Signal analysis
print('='*60)
print('SIGNAL ANALYSIS')
print('='*60)

bull_score = 0
bear_score = 0
notes = []

# Price vs VP levels (1HR)
if current > vah_1h:
    bear_score += 15
    notes.append('ABOVE VAH (extended, fade risk)')
elif current < val_1h:
    bull_score += 15
    notes.append('BELOW VAL (oversold, bounce zone)')
elif current > poc_1h:
    bull_score += 10
    notes.append('Above POC (bullish)')
else:
    bear_score += 10
    notes.append('Below POC (bearish)')

# Price vs VWAP
if current > vwap_1h:
    bull_score += 10
    notes.append('Above VWAP')
else:
    bear_score += 10
    notes.append('Below VWAP')

# RSI
if rsi_1h > 70:
    bear_score += 15
    notes.append(f'RSI Overbought ({rsi_1h})')
elif rsi_1h < 30:
    bull_score += 15
    notes.append(f'RSI Oversold ({rsi_1h})')
elif rsi_1h > 50:
    bull_score += 5
    notes.append(f'RSI bullish ({rsi_1h})')
else:
    bear_score += 5
    notes.append(f'RSI bearish ({rsi_1h})')

# Daily trend context
if current > vah_d:
    bull_score += 20
    notes.append('DAILY: Above value area (STRONG TREND)')
elif current < val_d:
    bear_score += 20
    notes.append('DAILY: Below value area (WEAK)')
elif current > poc_d:
    bull_score += 10
    notes.append('DAILY: Above POC')
else:
    bear_score += 10
    notes.append('DAILY: Below POC')

# Volume
if rvol_1h > 1.5:
    notes.append(f'HIGH VOLUME ({rvol_1h}x avg)')
elif rvol_1h < 0.7:
    notes.append(f'LOW VOLUME ({rvol_1h}x avg)')

print('Notes:')
for n in notes:
    print(f'  - {n}')
print()

# Determine signal
gap = abs(bull_score - bear_score)
if bull_score >= 35 and gap >= 15:
    signal = 'BULLISH'
    emoji = '[LONG SETUP]'
elif bear_score >= 35 and gap >= 15:
    signal = 'BEARISH'
    emoji = '[SHORT SETUP]'
else:
    signal = 'NEUTRAL/MIXED'
    emoji = '[WAIT/YELLOW]'

print(f'Bull Score: {bull_score}')
print(f'Bear Score: {bear_score}')
print(f'Gap: {gap}')
print()
print(f'>>> SIGNAL: {emoji} {signal}')
print()

# Key levels for the week
print('='*60)
print('KEY LEVELS TO WATCH THIS WEEK')
print('='*60)
print(f'Resistance: ${vah_1h:.2f} (1HR VAH), ${vah_d:.2f} (Daily VAH)')
print(f'Support: ${val_1h:.2f} (1HR VAL), ${val_d:.2f} (Daily VAL)')
print(f'Pivot: ${poc_1h:.2f} (1HR POC), ${poc_d:.2f} (Daily POC)')
print()
print(f'If LONG:')
print(f'  Entry Zone: ${val_1h:.2f} - ${poc_1h:.2f}')
print(f'  Stop Loss: ${current - atr_1h * 1.5:.2f}')
print(f'  Target 1: ${current + atr_1h * 2:.2f}')
print(f'  Target 2: ${current + atr_1h * 4:.2f}')
print()
print(f'If SHORT:')
print(f'  Entry Zone: ${vah_1h:.2f} - ${poc_1h:.2f}')
print(f'  Stop Loss: ${current + atr_1h * 1.5:.2f}')
print(f'  Target 1: ${current - atr_1h * 2:.2f}')
print(f'  Target 2: ${current - atr_1h * 4:.2f}')
print()

# Weekly outlook
print('='*60)
print('WEEKLY OUTLOOK')
print('='*60)
# Check where price is relative to the bigger picture
weekly_bias = "BULLISH" if bull_score > bear_score else "BEARISH" if bear_score > bull_score else "NEUTRAL"
print(f'Bias: {weekly_bias}')

if current > vah_d:
    print('Context: Price extended above daily value area - watch for continuation or mean reversion')
    print('Scenario 1: Breakout holds -> target next resistance')
    print('Scenario 2: Failed breakout -> fade back to POC')
elif current < val_d:
    print('Context: Price below daily value area - oversold territory')
    print('Scenario 1: Bounce from support -> target POC')
    print('Scenario 2: Breakdown continues -> new lows')
else:
    print('Context: Price within value area - range-bound action likely')
    print('Scenario 1: Test VAH for breakout')
    print('Scenario 2: Test VAL for breakdown')
    print('Scenario 3: Chop around POC')

print()
print('EARNINGS: Jan 28 (Tuesday after close) - MAJOR CATALYST')
print('Expect elevated IV and potential gap risk')
