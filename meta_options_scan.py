"""META Options Chain Analysis — Polygon API"""
from polygon_data import get_price_quote
from polygon_options import fetch_options_snapshot_filtered, parse_contract, group_by_expiration
import pandas as pd
from datetime import datetime, timedelta

symbol = 'META'

# Get current price via Polygon
q = get_price_quote(symbol)
current_price = q['price'] if q else 658.75

print('='*70)
print(f'META OPTIONS CHAIN ANALYSIS')
print(f'Current Price: ${current_price:.2f}')
print(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print('='*70)

# Fetch options snapshot from Polygon
raw = fetch_options_snapshot_filtered(symbol, dte_min=0, dte_max=60, strike_range_pct=0.20)
parsed = [parse_contract(c) for c in raw.get("contracts", [])]
grouped = group_by_expiration(parsed)
expirations = sorted(grouped.keys())

print(f'\nAvailable Expirations: {len(expirations)} dates')

# Focus on near-term expirations
near_term = expirations[:5]
print(f'Analyzing: {near_term}')
print()

# Analyze each near-term expiration
for exp_date in near_term[:3]:
    print('='*70)
    print(f'EXPIRATION: {exp_date}')
    print('='*70)

    try:
        clist = grouped[exp_date]
        call_rows = [c for c in clist if c["contractType"] == "call"]
        put_rows  = [c for c in clist if c["contractType"] == "put"]

        calls = pd.DataFrame([{
            "strike": c["strike"],
            "lastPrice": c.get("lastPrice") or c.get("midpoint") or 0,
            "bid": c.get("bid") or 0,
            "ask": c.get("ask") or 0,
            "volume": c.get("dayVolume") or 0,
            "openInterest": c.get("openInterest") or 0,
            "impliedVolatility": c.get("iv") or 0,
        } for c in call_rows]) if call_rows else pd.DataFrame()

        puts = pd.DataFrame([{
            "strike": p["strike"],
            "lastPrice": p.get("lastPrice") or p.get("midpoint") or 0,
            "bid": p.get("bid") or 0,
            "ask": p.get("ask") or 0,
            "volume": p.get("dayVolume") or 0,
            "openInterest": p.get("openInterest") or 0,
            "impliedVolatility": p.get("iv") or 0,
        } for p in put_rows]) if put_rows else pd.DataFrame()

        # Filter to strikes near current price (+/- 15%)
        strike_low = current_price * 0.85
        strike_high = current_price * 1.15

        calls_filtered = calls[(calls['strike'] >= strike_low) & (calls['strike'] <= strike_high)].copy() if not calls.empty else pd.DataFrame()
        puts_filtered = puts[(puts['strike'] >= strike_low) & (puts['strike'] <= strike_high)].copy() if not puts.empty else pd.DataFrame()
        
        # Summary stats
        total_call_vol = calls['volume'].sum() if 'volume' in calls.columns else 0
        total_put_vol = puts['volume'].sum() if 'volume' in puts.columns else 0
        total_call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
        total_put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
        
        pc_vol_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        pc_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        print(f'\nOVERVIEW:')
        print(f'  Total Call Volume: {total_call_vol:,.0f}')
        print(f'  Total Put Volume:  {total_put_vol:,.0f}')
        print(f'  Put/Call Vol Ratio: {pc_vol_ratio:.2f}')
        print(f'  Total Call OI: {total_call_oi:,.0f}')
        print(f'  Total Put OI:  {total_put_oi:,.0f}')
        print(f'  Put/Call OI Ratio: {pc_oi_ratio:.2f}')
        
        # Interpret P/C ratio
        if pc_vol_ratio < 0.7:
            sentiment = "BULLISH (call heavy)"
        elif pc_vol_ratio > 1.0:
            sentiment = "BEARISH (put heavy)"
        else:
            sentiment = "NEUTRAL"
        print(f'  Sentiment: {sentiment}')
        
        # Find max pain (strike with most total OI)
        all_strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
        max_pain_data = []
        for strike in all_strikes:
            call_oi = calls[calls['strike'] == strike]['openInterest'].sum() if 'openInterest' in calls.columns else 0
            put_oi = puts[puts['strike'] == strike]['openInterest'].sum() if 'openInterest' in puts.columns else 0
            # Max pain = strike where ITM options cost most
            call_itm_value = max(0, current_price - strike) * call_oi
            put_itm_value = max(0, strike - current_price) * put_oi
            total_pain = call_itm_value + put_itm_value
            max_pain_data.append({'strike': strike, 'call_oi': call_oi, 'put_oi': put_oi, 'total_oi': call_oi + put_oi, 'pain': total_pain})
        
        mp_df = pd.DataFrame(max_pain_data)
        if not mp_df.empty:
            # Max pain is where total pain is MINIMIZED (MMs want price here)
            # Actually, we want the strike with highest total OI as a magnet
            max_oi_strike = mp_df.loc[mp_df['total_oi'].idxmax(), 'strike']
            print(f'  Max OI Strike (magnet): ${max_oi_strike:.2f}')
        
        # Highest volume calls (unusual activity)
        print(f'\nHIGHEST VOLUME CALLS:')
        if not calls_filtered.empty and 'volume' in calls_filtered.columns:
            top_calls = calls_filtered.nlargest(5, 'volume')[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
            for _, row in top_calls.iterrows():
                iv_pct = row['impliedVolatility'] * 100 if pd.notna(row['impliedVolatility']) else 0
                vol = row['volume'] if pd.notna(row['volume']) else 0
                oi = row['openInterest'] if pd.notna(row['openInterest']) else 0
                print(f"  ${row['strike']:.0f}C | Last: ${row['lastPrice']:.2f} | Vol: {vol:,.0f} | OI: {oi:,.0f} | IV: {iv_pct:.1f}%")
        
        # Highest volume puts
        print(f'\nHIGHEST VOLUME PUTS:')
        if not puts_filtered.empty and 'volume' in puts_filtered.columns:
            top_puts = puts_filtered.nlargest(5, 'volume')[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
            for _, row in top_puts.iterrows():
                iv_pct = row['impliedVolatility'] * 100 if pd.notna(row['impliedVolatility']) else 0
                vol = row['volume'] if pd.notna(row['volume']) else 0
                oi = row['openInterest'] if pd.notna(row['openInterest']) else 0
                print(f"  ${row['strike']:.0f}P | Last: ${row['lastPrice']:.2f} | Vol: {vol:,.0f} | OI: {oi:,.0f} | IV: {iv_pct:.1f}%")
        
        # Highest OI (where big positions are)
        print(f'\nHIGHEST OPEN INTEREST CALLS:')
        if not calls_filtered.empty and 'openInterest' in calls_filtered.columns:
            top_oi_calls = calls_filtered.nlargest(5, 'openInterest')[['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']]
            for _, row in top_oi_calls.iterrows():
                iv_pct = row['impliedVolatility'] * 100 if pd.notna(row['impliedVolatility']) else 0
                vol = row['volume'] if pd.notna(row['volume']) else 0
                oi = row['openInterest'] if pd.notna(row['openInterest']) else 0
                print(f"  ${row['strike']:.0f}C | Last: ${row['lastPrice']:.2f} | Vol: {vol:,.0f} | OI: {oi:,.0f} | IV: {iv_pct:.1f}%")
        
        print(f'\nHIGHEST OPEN INTEREST PUTS:')
        if not puts_filtered.empty and 'openInterest' in puts_filtered.columns:
            top_oi_puts = puts_filtered.nlargest(5, 'openInterest')[['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']]
            for _, row in top_oi_puts.iterrows():
                iv_pct = row['impliedVolatility'] * 100 if pd.notna(row['impliedVolatility']) else 0
                vol = row['volume'] if pd.notna(row['volume']) else 0
                oi = row['openInterest'] if pd.notna(row['openInterest']) else 0
                print(f"  ${row['strike']:.0f}P | Last: ${row['lastPrice']:.2f} | Vol: {vol:,.0f} | OI: {oi:,.0f} | IV: {iv_pct:.1f}%")
        
        # ATM IV (for expected move calc)
        atm_strike = min(calls['strike'], key=lambda x: abs(x - current_price))
        atm_call = calls[calls['strike'] == atm_strike]
        atm_put = puts[puts['strike'] == atm_strike]
        
        if not atm_call.empty and 'impliedVolatility' in atm_call.columns:
            atm_iv = atm_call['impliedVolatility'].iloc[0]
            if pd.notna(atm_iv):
                # Calculate days to expiry
                exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                days_to_exp = (exp_dt - datetime.now()).days + 1
                
                # Expected move = Price * IV * sqrt(DTE/365)
                import math
                expected_move = current_price * atm_iv * math.sqrt(days_to_exp / 365)
                expected_move_pct = (expected_move / current_price) * 100
                
                print(f'\nEXPECTED MOVE:')
                print(f'  ATM IV: {atm_iv*100:.1f}%')
                print(f'  Days to Expiry: {days_to_exp}')
                print(f'  Expected Move: +/- ${expected_move:.2f} ({expected_move_pct:.1f}%)')
                print(f'  Range: ${current_price - expected_move:.2f} - ${current_price + expected_move:.2f}')
        
        print()
        
    except Exception as e:
        print(f'Error fetching {exp_date}: {e}')
        continue

# Summary
print('='*70)
print('OPTIONS FLOW SUMMARY')
print('='*70)
print('''
KEY OBSERVATIONS:
- Check Put/Call ratios for sentiment
- High volume + low OI = new positions being opened
- High OI strikes act as "magnets" (max pain theory)
- ATM IV tells you market's expected move

EARNINGS PLAY CONSIDERATIONS (Jan 28):
- IV will be elevated pre-earnings
- Expect IV crush post-earnings (sellers win if move < expected)
- Straddles/strangles price in the expected move
- Consider selling premium if you think move will be smaller than priced

FLOW SIGNALS TO WATCH:
- Large call sweeps above ask = aggressive bullish bets
- Large put sweeps = hedging or bearish bets  
- Unusual OI buildup at specific strikes = institutional positioning
''')
