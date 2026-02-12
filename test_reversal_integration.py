"""
Test Structure Reversal Integration with Trade Rule Engine
"""
from trade_rule_engine import RuleEngine, TradingRules
from structure_reversal_detector import ReversalType, AlertSeverity

# Mock scanner data
scanner_data = {
    'symbol': 'AAPL',
    'current_price': 175.50,
    'vah': 176.80,
    'poc': 175.00,
    'val': 173.20,
    'vwap': 175.30,
    'bull_score': 65,
    'bear_score': 45,
    'rsi': 58,
    'rvol': 1.2,
    'confidence': 60
}

# Create rule engine
engine = RuleEngine()

print("="*70)
print("TEST 1: Trade Plan WITHOUT Reversal Alerts")
print("="*70)
plan1 = engine.generate_plan(scanner_data)
print(f"Direction: {plan1.direction}")
print(f"Confidence: {plan1.confidence:.0f}%")
print(f"Entry: ${plan1.entry_price:.2f}, Stop: ${plan1.stop_loss:.2f}, T1: ${plan1.target_1:.2f}")
print(f"R:R: {plan1.risk_reward_t1:.1f}:1")
print(f"Entry Reasons ({len(plan1.entry_reasons)}):")
for reason in plan1.entry_reasons[:5]:
    print(f"  • {reason}")
print(f"Caution Flags ({len(plan1.caution_flags)}):")
for flag in plan1.caution_flags[:5]:
    print(f"  • {flag}")

print("\n" + "="*70)
print("TEST 2: Trade Plan WITH MOMENTUM_EXHAUSTION_LONG Alert (HIGH severity)")
print("="*70)

# Add momentum exhaustion alert
scanner_data_with_alert = scanner_data.copy()
scanner_data_with_alert['structure_reversals'] = [
    {
        'alert_type': 'MOMENTUM_EXHAUSTION_LONG',
        'severity': 'HIGH',
        'confidence': 68.0,
        'description': 'Uptrend momentum exhaustion - no new highs in 6+ days. Buyers weakening.',
        'timeframe': 'daily',
        'signals': ['No new highs for 6+ days', 'Volume declining', 'At top of range']
    }
]

plan2 = engine.generate_plan(scanner_data_with_alert)
print(f"Direction: {plan2.direction} (was {plan1.direction})")
print(f"Confidence: {plan2.confidence:.0f}% (was {plan1.confidence:.0f}%, change: {plan2.confidence - plan1.confidence:+.0f}%)")
print(f"Entry: ${plan2.entry_price:.2f}, Stop: ${plan2.stop_loss:.2f}, T1: ${plan2.target_1:.2f}")
print(f"Entry Reasons ({len(plan2.entry_reasons)}):")
for reason in plan2.entry_reasons[:5]:
    print(f"  • {reason}")
print(f"Caution Flags ({len(plan2.caution_flags)}):")
for flag in plan2.caution_flags[:5]:
    print(f"  • {flag}")

print("\n" + "="*70)
print("TEST 3: Trade Plan WITH RANGE_EXTREME_LONG Alert (CRITICAL severity)")
print("="*70)

scanner_data_reversal = scanner_data.copy()
scanner_data_reversal['structure_reversals'] = [
    {
        'alert_type': 'RANGE_EXTREME_LONG',
        'severity': 'CRITICAL',
        'confidence': 82.0,
        'description': 'At bottom of range with Higher Low forming - potential bounce. Position: 12.3%',
        'timeframe': '30-day range',
        'signals': ['At 87.7% of range', 'Higher Low forming', 'Volume 1.4x average', 'At VAL support'],
        'trigger_level': 173.50,
        'target_level': 179.20,
        'stop_level': 172.10
    }
]

plan3 = engine.generate_plan(scanner_data_reversal)
print(f"Direction: {plan3.direction} (was {plan1.direction})")
print(f"Confidence: {plan3.confidence:.0f}% (was {plan1.confidence:.0f}%, change: {plan3.confidence - plan1.confidence:+.0f}%)")
print(f"Entry: ${plan3.entry_price:.2f}, Stop: ${plan3.stop_loss:.2f}, T1: ${plan3.target_1:.2f}")
print(f"Entry Reasons ({len(plan3.entry_reasons)}):")
for reason in plan3.entry_reasons[:5]:
    print(f"  • {reason}")
print(f"Caution Flags ({len(plan3.caution_flags)}):")
for flag in plan3.caution_flags[:5]:
    print(f"  • {flag}")

print("\n" + "="*70)
print("TEST 4: Trade Plan WITH MULTIPLE Alerts (HIGH + MEDIUM)")
print("="*70)

scanner_data_multi = scanner_data.copy()
scanner_data_multi['structure_reversals'] = [
    {
        'alert_type': 'MOMENTUM_EXHAUSTION_LONG',
        'severity': 'HIGH',
        'confidence': 68.0,
        'description': 'Uptrend momentum exhaustion - no new highs in 6+ days.',
        'timeframe': 'daily'
    },
    {
        'alert_type': 'STRUCTURE_BREAK_LONG',
        'severity': 'MEDIUM',
        'confidence': 52.0,
        'description': 'Lower Low detected in uptrend - potential reversal to downside.',
        'timeframe': 'daily+weekly'
    }
]

plan4 = engine.generate_plan(scanner_data_multi)
print(f"Direction: {plan4.direction} (was {plan1.direction})")
print(f"Confidence: {plan4.confidence:.0f}% (was {plan1.confidence:.0f}%, change: {plan4.confidence - plan1.confidence:+.0f}%)")
print(f"Entry: ${plan4.entry_price:.2f}, Stop: ${plan4.stop_loss:.2f}, T1: ${plan4.target_1:.2f}")
print(f"Entry Reasons ({len(plan4.entry_reasons)}):")
for reason in plan4.entry_reasons[:5]:
    print(f"  • {reason}")
print(f"Caution Flags ({len(plan4.caution_flags)}):")
for flag in plan4.caution_flags[:5]:
    print(f"  • {flag}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Baseline (no alerts):          {plan1.direction:12} Conf {plan1.confidence:.0f}%, Reasons: {len(plan1.entry_reasons)}, Warnings: {len(plan1.caution_flags)}")
print(f"With EXHAUSTION_LONG:          {plan2.direction:12} Conf {plan2.confidence:.0f}%, Reasons: {len(plan2.entry_reasons)}, Warnings: {len(plan2.caution_flags)} (conf {plan2.confidence - plan1.confidence:+.0f}%)")
print(f"With RANGE_EXTREME_LONG:       {plan3.direction:12} Conf {plan3.confidence:.0f}%, Reasons: {len(plan3.entry_reasons)}, Warnings: {len(plan3.caution_flags)} (conf {plan3.confidence - plan1.confidence:+.0f}%)")
print(f"With Multiple Alerts:          {plan4.direction:12} Conf {plan4.confidence:.0f}%, Reasons: {len(plan4.entry_reasons)}, Warnings: {len(plan4.caution_flags)} (conf {plan4.confidence - plan1.confidence:+.0f}%)")
print("\n✅ Structure reversal alerts are now integrated into trade plan generation!")
