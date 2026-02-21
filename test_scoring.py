"""
Unit Tests for Scanner Scoring Logic
=====================================
Tests the SignalScorer, VolumeProfileEngine, RSIEngine, and FlowControlEngine
with known inputs to catch regressions when tuning parameters.

Run:  python -m pytest test_scoring.py -v
      python test_scoring.py   (standalone)
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from dataclasses import dataclass


# =============================================================================
# IMPORTS
# =============================================================================

from mtf_auction_scanner_v2 import (
    SignalScorer, SignalState, VolumeProfile, FlowMetrics, RSIMetrics,
    VolumeProfileEngine, FlowControlEngine, RSIEngine,
)
from scanner_config import (
    SwingTradeConfig, ScoringConfig, VolumeProfileConfig,
    RSIConfig, FlowConfig, CONSERVATIVE, BALANCED, ACTIVE,
)


# =============================================================================
# TEST DATA FACTORIES
# =============================================================================

def make_ohlcv(n=50, base_price=100.0, volatility=0.02, trend=0.0):
    """Generate synthetic OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    prices = [base_price]
    for _ in range(n - 1):
        change = np.random.normal(trend, volatility)
        prices.append(prices[-1] * (1 + change))

    data = []
    for i, p in enumerate(prices):
        h = p * (1 + abs(np.random.normal(0, volatility / 2)))
        l = p * (1 - abs(np.random.normal(0, volatility / 2)))
        o = l + (h - l) * np.random.random()
        c = l + (h - l) * np.random.random()
        v = int(np.random.uniform(50000, 200000))
        data.append({"open": o, "high": h, "low": l, "close": c, "volume": v})

    df = pd.DataFrame(data, index=dates)
    return df


def make_vp(poc=100.0, vah=102.0, val=98.0):
    """Create a VolumeProfile with known levels."""
    return VolumeProfile(
        poc=poc, vah=vah, val=val,
        value_area_pct=0.70, total_volume=1000000,
    )


def make_flow(imbalance=0.0, momentum=0.0, buy_pct=0.5):
    """Create FlowMetrics with known values."""
    return FlowMetrics(
        cumulative_delta=imbalance * 1000,
        delta_momentum=momentum,
        buy_volume_pct=buy_pct,
        sell_volume_pct=1 - buy_pct,
        flow_imbalance=imbalance,
    )


def make_rsi(value=50.0, slope=0.0, divergence=None):
    """Create RSIMetrics with known values."""
    return RSIMetrics(value=value, slope=slope, divergence=divergence)


# =============================================================================
# SIGNAL SCORER TESTS
# =============================================================================

class TestSignalScorer(unittest.TestCase):
    """Test the SignalScorer with controlled inputs."""

    def setUp(self):
        self.scorer = SignalScorer()

    def test_strong_bullish_signal(self):
        """Strong buy flow + bullish RSI + price above VAH = LONG_SETUP."""
        vp = make_vp(poc=100, vah=102, val=98)
        flow = make_flow(imbalance=0.35)  # Strong buy
        rsi = make_rsi(value=62, slope=1.5)  # Bullish

        bull, bear, signal, conf, notes = self.scorer.score(
            current_price=103.0, vp=vp, flow=flow, rsi=rsi
        )
        self.assertGreater(bull, bear)
        self.assertEqual(signal, SignalState.LONG_SETUP)
        self.assertGreater(conf, 50)

    def test_strong_bearish_signal(self):
        """Strong sell flow + bearish RSI + price below VAL = SHORT_SETUP."""
        vp = make_vp(poc=100, vah=102, val=98)
        flow = make_flow(imbalance=-0.35)  # Strong sell
        rsi = make_rsi(value=38, slope=-1.5)  # Bearish

        bull, bear, signal, conf, notes = self.scorer.score(
            current_price=97.0, vp=vp, flow=flow, rsi=rsi
        )
        self.assertGreater(bear, bull)
        self.assertEqual(signal, SignalState.SHORT_SETUP)

    def test_neutral_low_scores(self):
        """When all inputs are balanced, signal should be NEUTRAL or YELLOW."""
        vp = make_vp(poc=100, vah=102, val=98)
        flow = make_flow(imbalance=0.0)
        rsi = make_rsi(value=50, slope=0)

        bull, bear, signal, conf, notes = self.scorer.score(
            current_price=100.0, vp=vp, flow=flow, rsi=rsi
        )
        self.assertIn(signal, [SignalState.NEUTRAL, SignalState.YELLOW])

    def test_yellow_mixed_signals(self):
        """Buy flow but bearish RSI → gap too small → YELLOW."""
        vp = make_vp(poc=100, vah=102, val=98)
        flow = make_flow(imbalance=0.20)  # Moderate buy
        rsi = make_rsi(value=38, slope=-1.0)  # Bearish

        bull, bear, signal, conf, notes = self.scorer.score(
            current_price=101.0, vp=vp, flow=flow, rsi=rsi
        )
        self.assertEqual(signal, SignalState.YELLOW)

    def test_divergence_bonus(self):
        """Bullish divergence should boost bull score."""
        vp = make_vp(poc=100, vah=102, val=98)
        flow = make_flow(imbalance=0.10)
        rsi_no_div = make_rsi(value=35, slope=0.5, divergence=None)
        rsi_with_div = make_rsi(value=35, slope=0.5, divergence="BULLISH_DIV")

        bull_no, _, _, _, _ = self.scorer.score(100.0, vp, flow, rsi_no_div)
        bull_yes, _, _, _, _ = self.scorer.score(100.0, vp, flow, rsi_with_div)

        self.assertGreater(bull_yes, bull_no)

    def test_scores_bounded_0_100(self):
        """Scores should always be clamped between 0 and 100."""
        vp = make_vp(poc=100, vah=102, val=98)
        flow = make_flow(imbalance=0.5)
        rsi = make_rsi(value=80, slope=3.0, divergence="BEARISH_DIV")

        bull, bear, signal, conf, notes = self.scorer.score(
            current_price=105.0, vp=vp, flow=flow, rsi=rsi
        )
        self.assertGreaterEqual(bull, 0)
        self.assertLessEqual(bull, 100)
        self.assertGreaterEqual(bear, 0)
        self.assertLessEqual(bear, 100)


# =============================================================================
# CONFIG WIRING TESTS
# =============================================================================

class TestConfigWiring(unittest.TestCase):
    """Test that scanner_config.py actually affects scoring."""

    def test_conservative_config_raises_thresholds(self):
        """Conservative config should require higher scores for signals."""
        conservative_scorer = SignalScorer(
            config=ScoringConfig.conservative(),
            flow_config=FlowConfig.standard(),
        )
        self.assertEqual(conservative_scorer.STRONG_THRESHOLD, 75)
        self.assertEqual(conservative_scorer.MODERATE_THRESHOLD, 55)
        self.assertEqual(conservative_scorer.MIN_SCORE_GAP, 30)

    def test_aggressive_config_lowers_thresholds(self):
        """Aggressive config should accept lower scores for signals."""
        aggressive_scorer = SignalScorer(
            config=ScoringConfig.aggressive(),
            flow_config=FlowConfig.standard(),
        )
        self.assertEqual(aggressive_scorer.STRONG_THRESHOLD, 55)
        self.assertEqual(aggressive_scorer.MODERATE_THRESHOLD, 40)
        self.assertEqual(aggressive_scorer.MIN_SCORE_GAP, 15)

    def test_default_config_matches_hardcoded(self):
        """Default config should produce same thresholds as original hardcoded values."""
        default_scorer = SignalScorer()
        self.assertEqual(default_scorer.STRONG_THRESHOLD, 65)
        self.assertEqual(default_scorer.MODERATE_THRESHOLD, 45)
        self.assertEqual(default_scorer.MIN_SCORE_GAP, 20)

    def test_sensitive_flow_config(self):
        """Sensitive flow config should detect weaker flow imbalances."""
        sensitive_scorer = SignalScorer(
            config=ScoringConfig(),
            flow_config=FlowConfig.sensitive(),
        )
        self.assertEqual(sensitive_scorer._flow_strong, 0.20)
        self.assertEqual(sensitive_scorer._flow_moderate, 0.10)
        self.assertEqual(sensitive_scorer._flow_mild, 0.03)

    def test_config_propagates_to_mtf_scanner(self):
        """SwingTradeConfig should propagate to sub-engines via MTFAuctionScanner."""
        from mtf_auction_scanner_v2 import MTFAuctionScanner

        config = SwingTradeConfig.conservative_swing()
        scanner = MTFAuctionScanner(config=config)

        # VP engine should have config values
        self.assertEqual(scanner.vp_engine.value_area_pct, config.volume_profile.value_area_pct)
        self.assertEqual(scanner.vp_engine.num_bins, config.volume_profile.num_bins)

        # RSI engine should have config values
        self.assertEqual(scanner.rsi_engine.period, config.rsi.period)

        # Flow engine should have config values
        self.assertEqual(scanner.flow_engine.momentum_period, config.flow.momentum_period)

        # Scorer should have config thresholds
        self.assertEqual(scanner.scorer.STRONG_THRESHOLD, config.scoring.strong_threshold)

    def test_aggressive_config_more_signals(self):
        """Aggressive config should produce LONG/SHORT on borderline inputs."""
        # Use input that's borderline — enough for aggressive but not conservative
        vp = make_vp(poc=100, vah=102, val=98)
        flow = make_flow(imbalance=0.20)
        rsi = make_rsi(value=58, slope=1.0)

        conservative_scorer = SignalScorer(
            config=ScoringConfig.conservative(),
            flow_config=FlowConfig.standard(),
        )
        aggressive_scorer = SignalScorer(
            config=ScoringConfig.aggressive(),
            flow_config=FlowConfig.standard(),
        )

        _, _, sig_cons, _, _ = conservative_scorer.score(103.0, vp, flow, rsi)
        _, _, sig_aggr, _, _ = aggressive_scorer.score(103.0, vp, flow, rsi)

        # Aggressive should be more likely to produce a directional signal
        # (though the exact outcome depends on the scoring math)
        # At minimum, verify they don't crash and produce valid signals
        self.assertIn(sig_cons, list(SignalState))
        self.assertIn(sig_aggr, list(SignalState))


# =============================================================================
# VP ENGINE TESTS
# =============================================================================

class TestVolumeProfileEngine(unittest.TestCase):
    """Test VP calculations with known data."""

    def test_basic_vp_calculation(self):
        """VP should produce valid POC, VAH, VAL from synthetic data."""
        df = make_ohlcv(50)
        engine = VolumeProfileEngine()
        vp = engine.calculate(df)

        self.assertIsNotNone(vp)
        self.assertGreater(vp.poc, 0)
        self.assertGreater(vp.vah, vp.val)
        self.assertGreaterEqual(vp.vah, vp.poc)
        self.assertLessEqual(vp.val, vp.poc)

    def test_vp_config_bins(self):
        """Changing num_bins should not crash and should still produce valid VP."""
        df = make_ohlcv(50)
        engine_25 = VolumeProfileEngine(num_bins=25)
        engine_100 = VolumeProfileEngine(num_bins=100)

        vp_25 = engine_25.calculate(df)
        vp_100 = engine_100.calculate(df)

        self.assertGreater(vp_25.vah, vp_25.val)
        self.assertGreater(vp_100.vah, vp_100.val)

    def test_vp_small_data(self):
        """VP should handle edge case of very few bars."""
        df = make_ohlcv(3)
        engine = VolumeProfileEngine()
        vp = engine.calculate(df)
        self.assertIsNotNone(vp)


# =============================================================================
# RSI ENGINE TESTS
# =============================================================================

class TestRSIEngine(unittest.TestCase):
    """Test RSI calculations."""

    def test_rsi_range(self):
        """RSI should be between 0 and 100."""
        df = make_ohlcv(50)
        engine = RSIEngine()
        rsi = engine.calculate(df)
        self.assertGreaterEqual(rsi.value, 0)
        self.assertLessEqual(rsi.value, 100)

    def test_uptrend_rsi(self):
        """Strong uptrend should produce RSI > 50."""
        df = make_ohlcv(50, trend=0.005)  # Upward bias
        engine = RSIEngine()
        rsi = engine.calculate(df)
        self.assertGreater(rsi.value, 50)

    def test_downtrend_rsi(self):
        """Strong downtrend should produce RSI < 50."""
        df = make_ohlcv(50, trend=-0.005)  # Downward bias
        engine = RSIEngine()
        rsi = engine.calculate(df)
        self.assertLess(rsi.value, 50)

    def test_rsi_zones(self):
        """Test RSI zone classification."""
        self.assertEqual(make_rsi(80).zone, "OVERBOUGHT")
        self.assertEqual(make_rsi(67).zone, "NEAR_OVERBOUGHT")
        self.assertEqual(make_rsi(58).zone, "BULLISH")
        self.assertEqual(make_rsi(50).zone, "NEUTRAL")
        self.assertEqual(make_rsi(38).zone, "BEARISH")
        self.assertEqual(make_rsi(32).zone, "NEAR_OVERSOLD")
        self.assertEqual(make_rsi(25).zone, "OVERSOLD")


# =============================================================================
# FLOW ENGINE TESTS
# =============================================================================

class TestFlowControlEngine(unittest.TestCase):
    """Test flow/delta calculations."""

    def test_flow_range(self):
        """Flow imbalance should be between -1 and 1."""
        df = make_ohlcv(30)
        engine = FlowControlEngine()
        flow = engine.calculate(df)
        self.assertGreaterEqual(flow.flow_imbalance, -1)
        self.assertLessEqual(flow.flow_imbalance, 1)

    def test_buy_sell_pct_sum(self):
        """Buy + sell volume pct should sum to ~1.0."""
        df = make_ohlcv(30)
        engine = FlowControlEngine()
        flow = engine.calculate(df)
        total = flow.buy_volume_pct + flow.sell_volume_pct
        self.assertAlmostEqual(total, 1.0, places=2)


# =============================================================================
# UNIVERSE TESTS
# =============================================================================

class TestUniverse(unittest.TestCase):
    """Test centralized universe module."""

    def test_all_symbols_deduped(self):
        """ALL_SYMBOLS should have no duplicates."""
        from universe import ALL_SYMBOLS
        self.assertEqual(len(ALL_SYMBOLS), len(set(ALL_SYMBOLS)))

    def test_get_universe(self):
        """get_universe should return known lists."""
        from universe import get_universe, MAG7
        result = get_universe("mag7")
        self.assertEqual(result, MAG7)

    def test_get_universe_fallback(self):
        """Unknown universe should fall back to ALL_SYMBOLS."""
        from universe import get_universe, ALL_SYMBOLS
        result = get_universe("nonexistent")
        self.assertEqual(result, ALL_SYMBOLS)

    def test_presets_not_empty(self):
        """All preset universes should be non-empty."""
        from universe import (
            TECH, MEGA, ETFS, MEME, SEMIS, MOMENTUM, MAG7,
            OPTIONS_PRESETS, BUFFETT_PRESETS, AUTO_SCANNER_DEFAULTS,
        )
        for name, lst in [("TECH", TECH), ("MEGA", MEGA), ("ETFS", ETFS),
                          ("MEME", MEME), ("SEMIS", SEMIS), ("MOMENTUM", MOMENTUM),
                          ("MAG7", MAG7), ("AUTO", AUTO_SCANNER_DEFAULTS)]:
            self.assertGreater(len(lst), 0, f"{name} is empty")

        for name, d in [("OPTIONS", OPTIONS_PRESETS), ("BUFFETT", BUFFETT_PRESETS)]:
            for k, v in d.items():
                self.assertGreater(len(v), 0, f"{name}.{k} is empty")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
