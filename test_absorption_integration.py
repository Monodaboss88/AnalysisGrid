"""
Test Absorption Detector Integration with Rule Engine
Tests that absorption zones affect trade plan confidence scores.
"""
from trade_rule_engine import generate_plan
from datetime import datetime
import sys
import io

# Set UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Base scanner data for AAPL - make it a strong setup
scanner_data_base = {
    'symbol': 'AAPL',
    'timeframe': '1HR',
    'signal': 'AUCTION_SIGNAL_LONG',  # Changed to a recognized signal
    'signal_strength': 'strong',
    'confidence': 75,  # Higher baseline confidence
    'open': 261.00,
    'close': 261.59,
    'high': 262.50,
    'low': 260.50,
    'vah': 263.00,
    'poc': 261.50,
    'val': 260.00,
    'vwap': 261.20,
    'rsi': 58,
    'rvol': 1.8,  # Higher RVOL
    'position': 'in_value',
    'volume_trend': 'increasing',
    'volume_divergence': False,
    'atr': 2.50,
    'timestamp': datetime.now().isoformat(),
    'bull_score': 75,  # Explicit scores
    'bear_score': 25
}

print("=" * 80)
print("ABSORPTION DETECTOR - RULE ENGINE INTEGRATION TEST")
print("=" * 80)

# Test 1: No absorption zones (baseline)
print("=" * 80)
print("TEST 1: BASELINE (no absorption)")
print("=" * 80)
scanner_data_baseline = scanner_data_base.copy()
plan1, expl1, plan_id1 = generate_plan(scanner_data_baseline, explain=True)
print(f"\nConfidence: {plan1.confidence}%")
print(f"Direction: {plan1.direction}")
print(f"Entry: ${plan1.entry_price:.2f}")
print(f"Target 1: ${plan1.target_1 if plan1.target_1 else 0:.2f}")
print(f"Stop Loss: ${plan1.stop_loss if plan1.stop_loss else 0:.2f}")
print("\nCaution Flags:")
for flag in plan1.caution_flags or []:
    print(f"  {flag}")
print("\nEntry Reasons:")
for reason in plan1.entry_reasons or []:
    print(f"  {reason}")

# Test 2: STRONG CEILING absorption above price (should reduce bullish conviction)
print("\n" + "=" * 80)
print("TEST 2: STRONG CEILING at $262.00 (above price, resistance)")
print("=" * 80)
scanner_data_ceiling = scanner_data_base.copy()
scanner_data_ceiling['absorption_zones'] = [
    {
        'center_price': 262.00,
        'upper_bound': 262.40,
        'lower_bound': 261.60,
        'absorption_type': 'CEILING',
        'strength': 'STRONG',
        'status': 'DEFENDED',
        'total_touches': 13,
        'rvol_ratio': 1.63,
        'score': 83
    }
]
plan2, expl2, plan_id2 = generate_plan(scanner_data_ceiling, explain=True)
print(f"\nConfidence: {plan2.confidence}% (was {plan1.confidence}%)")
print(f"Confidence Change: {plan2.confidence - plan1.confidence:+.0f} points")
print(f"Direction: {plan2.direction}")
print(f"Entry: ${plan2.entry_price:.2f}")
print(f"Target 1: ${plan2.target_1 if plan2.target_1 else 0:.2f}")
print(f"Stop Loss: ${plan2.stop_loss if plan2.stop_loss else 0:.2f}")
print("\nCaution Flags:")
for flag in plan2.caution_flags or []:
    print(f"  {flag}")

# Test 3: STRONG FLOOR absorption below price (should reduce bearish conviction for shorts)
print("\n" + "=" * 80)
print("TEST 3: STRONG FLOOR at $260.00 (below price, support)")
print("=" * 80)
scanner_data_floor = scanner_data_base.copy()
scanner_data_floor['signal'] = 'BREAKOUT_SHORT'
scanner_data_floor['confidence'] = 70
scanner_data_floor['absorption_zones'] = [
    {
        'center_price': 260.00,
        'upper_bound': 260.40,
        'lower_bound': 259.60,
        'absorption_type': 'FLOOR',
        'strength': 'STRONG',
        'status': 'DEFENDED',
        'total_touches': 10,
        'rvol_ratio': 1.45,
        'score': 75
    }
]
plan3, expl3, plan_id3 = generate_plan(scanner_data_floor, explain=True)
baseline_short = generate_plan({**scanner_data_base, 'signal': 'BREAKOUT_SHORT'}, explain=False)[0]
print(f"\nConfidence: {plan3.confidence}% (was {baseline_short.confidence}%)")
print(f"Confidence Change: {plan3.confidence - baseline_short.confidence:+.0f} points")
print(f"Direction: {plan3.direction}")
print(f"Entry: ${plan3.entry_price:.2f}")
print(f"Target 1: ${plan3.target_1 if plan3.target_1 else 0:.2f}")
print(f"Stop Loss: ${plan3.stop_loss if plan3.stop_loss else 0:.2f}")
print("\nCaution Flags:")
for flag in plan3.caution_flags or []:
    print(f"  {flag}")

# Test 4: CEILING WEAKENING (breakout potential)
print("\n" + "=" * 80)
print("TEST 4: CEILING WEAKENING at $261.60 (breakout potential)")
print("=" * 80)
scanner_data_weakening = scanner_data_base.copy()
scanner_data_weakening['absorption_zones'] = [
    {
        'center_price': 261.60,
        'upper_bound': 262.00,
        'lower_bound': 261.20,
        'absorption_type': 'CEILING',
        'strength': 'MODERATE',
        'status': 'WEAKENING',
        'total_touches': 8,
        'rvol_ratio': 1.20,
        'score': 60
    }
]
plan4, expl4, plan_id4 = generate_plan(scanner_data_weakening, explain=True)
print(f"\nConfidence: {plan4.confidence}% (was {plan1.confidence}%)")
print(f"Confidence Change: {plan4.confidence - plan1.confidence:+.0f} points")
print(f"Direction: {plan4.direction}")
print("\nEntry Reasons:")
for reason in plan4.entry_reasons or []:
    print(f"  {reason}")

# Test 5: INSTITUTIONAL strength absorption (maximum impact)
print("\n" + "=" * 80)
print("TEST 5: INSTITUTIONAL CEILING at $262.50 (maximum resistance)")
print("=" * 80)
scanner_data_institutional = scanner_data_base.copy()
scanner_data_institutional['absorption_zones'] = [
    {
        'center_price': 262.50,
        'upper_bound': 263.00,
        'lower_bound': 262.00,
        'absorption_type': 'CEILING',
        'strength': 'INSTITUTIONAL',
        'status': 'HOLDING',
        'total_touches': 20,
        'rvol_ratio': 2.50,
        'score': 95
    }
]
plan5, expl5, plan_id5 = generate_plan(scanner_data_institutional, explain=True)
print(f"\nConfidence: {plan5.confidence}% (was {plan1.confidence}%)")
print(f"Confidence Change: {plan5.confidence - plan1.confidence:+.0f} points")
print(f"Direction: {plan5.direction}")
print("\nCaution Flags:")
for flag in plan5.caution_flags or []:
    print(f"  {flag}")

# Test 6: PINNING absorption (range-bound warning)
print("\n" + "=" * 80)
print("TEST 6: PINNING at $261.50 (range-bound, both sides absorbing)")
print("=" * 80)
scanner_data_pinning = scanner_data_base.copy()
scanner_data_pinning['absorption_zones'] = [
    {
        'center_price': 261.50,
        'upper_bound': 262.00,
        'lower_bound': 261.00,
        'absorption_type': 'PINNING',
        'strength': 'STRONG',
        'status': 'HOLDING',
        'total_touches': 15,
        'rvol_ratio': 1.80,
        'score': 82
    }
]
plan6, expl6, plan_id6 = generate_plan(scanner_data_pinning, explain=True)
print(f"\nConfidence: {plan6.confidence}% (was {plan1.confidence}%)")
print(f"Confidence Change: {plan6.confidence - plan1.confidence:+.0f} points")
print(f"Direction: {plan6.direction}")
print("\nCaution Flags:")
for flag in plan6.caution_flags or []:
    print(f"  {flag}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Baseline (no absorption):      {plan1.confidence}%")
print(f"STRONG CEILING (resistance):   {plan2.confidence}% ({plan2.confidence - plan1.confidence:+.0f})")
print(f"STRONG FLOOR (support):        {plan3.confidence}% ({plan3.confidence - baseline_short.confidence:+.0f} for short)")
print(f"CEILING WEAKENING (breakout):  {plan4.confidence}% ({plan4.confidence - plan1.confidence:+.0f})")
print(f"INSTITUTIONAL CEILING:         {plan5.confidence}% ({plan5.confidence - plan1.confidence:+.0f})")
print(f"PINNING (range-bound):         {plan6.confidence}% ({plan6.confidence - plan1.confidence:+.0f})")
print("\nExpected Impact:")
print("  STRONG walls (near price): -10 points")
print("  INSTITUTIONAL walls: -15 points")
print("  PINNING: -5 points (both directions)")
print("  WEAKENING walls: +5 points (breakout potential)")
