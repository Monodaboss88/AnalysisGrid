"""Quick test for Structure Reversal Detector"""
from polygon_data import get_bars
from structure_reversal_detector import StructureReversalDetector, StructureContext
from rangewatcher.range_watcher import RangeWatcher
from finnhub_scanner_v2 import TechnicalCalculator

def test_structure_reversals(symbol: str = "AAPL"):
    """Test structure reversal detection"""
    print(f"\n{'='*60}")
    print(f"STRUCTURE REVERSAL ANALYSIS: {symbol}")
    print(f"{'='*60}\n")
    
    # Get data
    df_daily = get_bars(symbol, period="3mo", interval="1d")
    df_weekly = get_bars(symbol, period="1y", interval="1wk")
    
    df_daily.columns = [c.lower() for c in df_daily.columns]
    df_weekly.columns = [c.lower() for c in df_weekly.columns]
    
    print(f"Daily candles: {len(df_daily)}")
    print(f"Weekly candles: {len(df_weekly)}")
    
    # Run RangeWatcher analysis
    range_watcher = RangeWatcher()
    range_result = range_watcher.analyze(df_daily, symbol=symbol)
    
    print(f"\nRangeWatcher: Trend={range_result.trend_structure.value}, Strength={range_result.trend_strength:.1f}")
    
    # Get weekly structure
    from chart_input_analyzer import RangeContext
    range_context = TechnicalCalculator.calculate_range_structure(
        df_weekly=df_weekly,
        df_daily=df_daily,
        current_price=df_daily['close'].iloc[-1]
    )
    
    print(f"Weekly Structure: {range_context.trend}")
    print(f"  HH: {range_context.hh_count}, HL: {range_context.hl_count}")
    print(f"  LH: {range_context.lh_count}, LL: {range_context.ll_count}")
    print(f"  Compression: {range_context.compression_ratio:.2f}")
    
    # Build StructureContext
    period_3d = range_result.periods.get(3)
    period_6d = range_result.periods.get(6)
    period_30d = range_result.periods.get(30)
    
    structure_ctx = StructureContext(
        weekly_trend=range_context.trend,
        weekly_hh=range_context.hh_count,
        weekly_hl=range_context.hl_count,
        weekly_lh=range_context.lh_count,
        weekly_ll=range_context.ll_count,
        weekly_close_position=range_context.weekly_close_position,
        
        period_3d_hh=period_3d.higher_highs if period_3d else False,
        period_3d_hl=period_3d.higher_lows if period_3d else False,
        period_3d_lh=period_3d.lower_highs if period_3d else False,
        period_3d_ll=period_3d.lower_lows if period_3d else False,
        
        period_6d_hh=period_6d.higher_highs if period_6d else False,
        period_6d_hl=period_6d.higher_lows if period_6d else False,
        period_6d_lh=period_6d.lower_highs if period_6d else False,
        period_6d_ll=period_6d.lower_lows if period_6d else False,
        
        period_30d_hh=period_30d.higher_highs if period_30d else False,
        period_30d_hl=period_30d.higher_lows if period_30d else False,
        period_30d_lh=period_30d.lower_highs if period_30d else False,
        period_30d_ll=period_30d.lower_lows if period_30d else False,
        
        current_price=range_result.current_price,
        position_in_3d_range=period_3d.position_in_range if period_3d else 0.5,
        position_in_30d_range=period_30d.position_in_range if period_30d else 0.5,
        compression_ratio=range_context.compression_ratio,
        
        nearest_resistance=period_30d.nearest_resistance if period_30d else range_result.current_price * 1.05,
        nearest_support=period_30d.nearest_support if period_30d else range_result.current_price * 0.95,
    )
    
    print(f"\n3-Day Structure: HH={period_3d.higher_highs}, HL={period_3d.higher_lows}, LH={period_3d.lower_highs}, LL={period_3d.lower_lows}")
    print(f"6-Day Structure: HH={period_6d.higher_highs}, HL={period_6d.higher_lows}, LH={period_6d.lower_highs}, LL={period_6d.lower_lows}")
    print(f"30-Day Structure: HH={period_30d.higher_highs}, HL={period_30d.higher_lows}, LH={period_30d.lower_highs}, LL={period_30d.lower_lows}")
    print(f"Position in 30D Range: {structure_ctx.position_in_30d_range*100:.1f}%")
    
    # Run reversal detection
    detector = StructureReversalDetector(min_confidence=40.0)
    alerts = detector.analyze(
        df=df_daily,
        structure_context=structure_ctx,
        symbol=symbol,
        vp_data=None
    )
    
    print(f"\n{'='*60}")
    print(f"REVERSAL ALERTS: {len(alerts)} found")
    print(f"{'='*60}\n")
    
    if not alerts:
        print("No reversal alerts detected with current criteria.")
    else:
        for i, alert in enumerate(alerts, 1):
            print(f"\n[{i}] {alert.alert_type.value}")
            print(f"    Severity: {alert.severity.value}")
            print(f"    Confidence: {alert.confidence:.1f}%")
            print(f"    {alert.description}")
            print(f"    Timeframe: {alert.timeframe}")
            if alert.trigger_level:
                print(f"    Trigger: ${alert.trigger_level:.2f}")
            if alert.target_level:
                print(f"    Target: ${alert.target_level:.2f}")
            if alert.stop_level:
                print(f"    Stop: ${alert.stop_level:.2f}")
            print(f"    Signals:")
            for sig in alert.signals:
                print(f"      • {sig}")
            print(f"    Scores: Structure={alert.structure_score:.1f}, Volume={alert.volume_score:.1f}, "
                  f"VP={alert.vp_confluence:.1f}, Momentum={alert.momentum_score:.1f}, "
                  f"Range={alert.range_position:.1f}, Divergence={alert.divergence_score:.1f}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    # Test with multiple symbols
    symbols = ["AAPL", "TSLA", "SPY", "NVDA"]
    
    for sym in symbols:
        try:
            test_structure_reversals(sym)
        except Exception as e:
            print(f"\nError testing {sym}: {e}\n")
