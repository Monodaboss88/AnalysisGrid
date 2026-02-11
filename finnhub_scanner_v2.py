"""
Finnhub Scanner V2 — Extended Market Scanner with Range Structure + Order Flow
================================================================================
Inherits from market_scanner_v2.MarketScanner (canonical data pipeline) and adds:
- calculate_range_structure(): Weekly HH/HL/LH/LL analysis with RangeContext
- get_order_flow_analysis(): Candle-based buying/selling pressure from Polygon
- Enhanced Polygon snapshot quotes (min bar → quote mid → trade → day → prev)
- VP bars windowing (vp_bars parameter for visible-range VP like Webull)
- Smart days_back auto-calculation per timeframe

ARCHITECTURE:
    market_scanner_v2.py  →  Canonical TechnicalCalculator, data pipeline, V2Context
    finnhub_scanner_v2.py →  Extends with range structure, order flow, production quotes

V1 this file was 1,760 lines — 95% duplicated with market_scanner.py.
V2 eliminates duplication via inheritance, keeping only unique logic.

RETAINED FROM V1:
- All data source priority (Polygon > Alpaca > Finnhub > yfinance)
- calculate_range_structure() with full RangeContext
- get_order_flow_analysis() with candle pressure + Polygon snapshot
- Enhanced Polygon snapshot quote pipeline
- vp_bars parameter for VP window sizing
- Auto days_back calculation per timeframe
- FinnhubScanner backward-compat alias

NEW IN V2:
- Inherits V2Context (weekly, squeeze, IV, enhanced VP) from market_scanner_v2
- Range structure integrated into V2Context.weekly
- Order flow context added to enriched analysis
- analyze_enriched() returns (AnalysisResult, V2Context, OrderFlow)
- scan_enriched() batch scan with full context
- RangeContext dataclass defined here (was in chart_input_analyzer)
- TechnicalCalculatorExtended adds range_structure to canonical calculator

Author: Rob's Trading Systems (Strategic Edge Flow)
Version: 2.0.0
"""

import os
import time
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Import canonical V2 base
from market_scanner_v2 import (
    MarketScanner as BaseMarketScanner,
    TechnicalCalculator as BaseTechnicalCalculator,
    V2Context, WeeklyContext, SqueezeContext, IVContext, EnhancedVP
)

# Import chart input system for V1 analysis
try:
    from chart_input_analyzer import ChartInputSystem, ChartInput, AnalysisResult, MTFResult
    chart_system_available = True
except ImportError:
    chart_system_available = False

# Optional: polygon for order flow
try:
    from polygon import RESTClient as PolygonClient
    polygon_available = True
except ImportError:
    polygon_available = False


# =============================================================================
# RANGE CONTEXT (was in chart_input_analyzer, canonicalized here)
# =============================================================================

@dataclass
class RangeContext:
    """
    Weekly range structure analysis.

    Built from weekly candles — each week = complete auction cycle.
    HH/HL/LH/LL on weekly bars = high-conviction structural signals.
    """
    trend: str = "NEUTRAL"
    range_state: str = "NORMAL"       # COMPRESSING, EXPANDING, NORMAL
    compression_ratio: float = 1.0    # Current week / 8-week avg
    ll_count: int = 0
    hh_count: int = 0
    lh_count: int = 0
    hl_count: int = 0
    total_periods: int = 0
    near_support: bool = False
    near_resistance: bool = False
    breakout_watch: float = 0.0
    breakdown_watch: float = 0.0
    weekly_close_position: float = 0.5
    weekly_close_signal: str = ""
    last_week_structure: str = ""

    def to_dict(self) -> Dict:
        return {
            'trend': self.trend,
            'range_state': self.range_state,
            'compression_ratio': self.compression_ratio,
            'll_count': self.ll_count,
            'hh_count': self.hh_count,
            'lh_count': self.lh_count,
            'hl_count': self.hl_count,
            'total_periods': self.total_periods,
            'near_support': self.near_support,
            'near_resistance': self.near_resistance,
            'breakout_watch': self.breakout_watch,
            'breakdown_watch': self.breakdown_watch,
            'weekly_close_position': self.weekly_close_position,
            'weekly_close_signal': self.weekly_close_signal,
            'last_week_structure': self.last_week_structure,
        }


@dataclass
class OrderFlowContext:
    """Candle-based order flow analysis result"""
    buy_pressure: float = 50.0
    sell_pressure: float = 50.0
    flow_bias: str = "NEUTRAL"        # BULLISH, BEARISH, NEUTRAL
    total_bars_analyzed: int = 0
    high_volume_bars: int = 0
    buy_candles: int = 0
    sell_candles: int = 0
    volume_ratio: float = 1.0         # Today vs yesterday
    volume_spike: bool = False
    momentum: str = "MIXED"           # STRONG BUY, STRONG SELL, MIXED
    spread_pct: float = 0.0
    timeframe: str = "1HR"
    period: str = "swing"

    def to_dict(self) -> Dict:
        return {
            'buy_pressure': self.buy_pressure,
            'sell_pressure': self.sell_pressure,
            'flow_bias': self.flow_bias,
            'total_bars_analyzed': self.total_bars_analyzed,
            'high_volume_bars': self.high_volume_bars,
            'buy_candles': self.buy_candles,
            'sell_candles': self.sell_candles,
            'volume_ratio': self.volume_ratio,
            'volume_spike': self.volume_spike,
            'momentum': self.momentum,
            'spread_pct': self.spread_pct,
            'timeframe': self.timeframe,
            'period': self.period,
        }


# =============================================================================
# EXTENDED TECHNICAL CALCULATOR
# =============================================================================

class TechnicalCalculator(BaseTechnicalCalculator):
    """
    Extended calculator — inherits all V2 canonical methods from
    market_scanner_v2.TechnicalCalculator, adds range structure.
    """

    @staticmethod
    def calculate_range_structure(df_weekly: pd.DataFrame,
                                   df_daily: pd.DataFrame = None,
                                   current_price: float = None) -> RangeContext:
        """
        Calculate range structure using WEEKLY candles for macro structure
        and optional DAILY candles for tactical proximity.

        Weekly candles represent complete auction cycles — HH/HL/LH/LL
        patterns on weekly bars are much more meaningful than daily.

        Args:
            df_weekly: DataFrame with weekly OHLCV (min 6 bars)
            df_daily: Optional daily OHLCV for proximity checks
            current_price: Current price for proximity calculations

        Returns:
            RangeContext with structural analysis
        """
        if df_weekly is None or len(df_weekly) < 6:
            return RangeContext()

        # Normalize columns
        cols = {c: c.lower() for c in df_weekly.columns}
        df_w = df_weekly.rename(columns=cols) if any(c[0].isupper() for c in df_weekly.columns) else df_weekly

        price = current_price or float(df_w['close'].iloc[-1])
        weeks = df_w.tail(8)

        # === WEEKLY STRUCTURE: HH/HL/LH/LL ===
        ll_count = hh_count = lh_count = hl_count = 0
        structures = {}

        for i in range(1, len(weeks)):
            curr, prev = weeks.iloc[i], weeks.iloc[i - 1]
            structure = ""

            if curr['high'] > prev['high'] * 1.001:
                structure += "HH"; hh_count += 1
            elif curr['high'] < prev['high'] * 0.999:
                structure += "LH"; lh_count += 1
            else:
                structure += "EQ"

            if curr['low'] > prev['low'] * 1.001:
                structure += "+HL"; hl_count += 1
            elif curr['low'] < prev['low'] * 0.999:
                structure += "+LL"; ll_count += 1
            else:
                structure += "+EQ"

            structures[f"W{i}"] = structure

        total_periods = len(structures)

        # === TREND CLASSIFICATION ===
        bearish_signals = ll_count + lh_count
        bullish_signals = hh_count + hl_count

        # Recent bias (last 3 weeks)
        recent_weeks = df_w.tail(4)
        recent_bearish = recent_bullish = 0
        for i in range(1, len(recent_weeks)):
            curr, prev = recent_weeks.iloc[i], recent_weeks.iloc[i - 1]
            if curr['high'] < prev['high'] * 0.999:
                recent_bearish += 1
            if curr['high'] > prev['high'] * 1.001:
                recent_bullish += 1
            if curr['low'] < prev['low'] * 0.999:
                recent_bearish += 1
            if curr['low'] > prev['low'] * 1.001:
                recent_bullish += 1

        if bearish_signals >= 8 and bullish_signals <= 2:
            trend = "STRONG_DOWNTREND"
        elif bearish_signals >= 5 and bearish_signals > bullish_signals * 2:
            trend = "DOWNTREND"
        elif bullish_signals >= 8 and bearish_signals <= 2:
            trend = "STRONG_UPTREND"
        elif bullish_signals >= 5 and bullish_signals > bearish_signals * 2:
            trend = "UPTREND"
        elif recent_bearish >= 4 and recent_bullish <= 1:
            trend = "DOWNTREND"
        elif recent_bullish >= 4 and recent_bearish <= 1:
            trend = "UPTREND"
        else:
            trend = "NEUTRAL"

        # === WEEKLY COMPRESSION ===
        week_ranges = [(weeks.iloc[i]['high'] - weeks.iloc[i]['low']) for i in range(len(weeks))]
        current_week_range = week_ranges[-1] if week_ranges else 0
        avg_8w_range = sum(week_ranges) / len(week_ranges) if week_ranges else 1

        narrowing = False
        if len(week_ranges) >= 4:
            half = len(week_ranges) // 2
            first_half = sum(week_ranges[:half]) / half
            second_half = sum(week_ranges[half:]) / (len(week_ranges) - half)
            narrowing = second_half < first_half * 0.80

        compression_ratio = current_week_range / avg_8w_range if avg_8w_range > 0 else 1.0
        if compression_ratio < 0.50 or narrowing:
            range_state = "COMPRESSING"
        elif compression_ratio > 1.50:
            range_state = "EXPANDING"
        else:
            range_state = "NORMAL"

        # === PROXIMITY ===
        high_8w = float(weeks['high'].max())
        low_8w = float(weeks['low'].min())
        price_range = high_8w - low_8w if high_8w > low_8w else 1

        near_support = (price - low_8w) / price_range < 0.10
        near_resistance = (high_8w - price) / price_range < 0.10

        if df_daily is not None and len(df_daily) >= 5:
            cols_d = {c: c.lower() for c in df_daily.columns}
            df_d = df_daily.rename(columns=cols_d) if any(c[0].isupper() for c in df_daily.columns) else df_daily
            daily_5d = df_d.tail(5)
            recent_low = float(daily_5d['low'].min())
            recent_high = float(daily_5d['high'].max())
            breakdown_watch = round(min(recent_low, low_8w), 2)
            breakout_watch = round(max(recent_high, high_8w), 2)
            daily_pos = (price - low_8w) / price_range if price_range > 0 else 0.5
            if daily_pos < 0.10:
                near_support = True
            elif daily_pos > 0.90:
                near_resistance = True
        else:
            cw = weeks.iloc[-1]
            breakdown_watch = round(min(float(cw['low']), low_8w), 2)
            breakout_watch = round(max(float(cw['high']), high_8w), 2)

        # === WEEKLY CLOSE POSITION ===
        last_week = weeks.iloc[-1]
        lw_range = last_week['high'] - last_week['low']
        wcp = (last_week['close'] - last_week['low']) / lw_range if lw_range > 0 else 0.5

        last_structure = list(structures.values())[-1] if structures else ""

        # Close signal
        if "LL" in last_structure and wcp > 0.70:
            weekly_close_signal = "BULLISH_REVERSAL"
        elif "HH" in last_structure and wcp < 0.30:
            weekly_close_signal = "BEARISH_REVERSAL"
        elif ("HH" in last_structure or "HL" in last_structure) and wcp > 0.70:
            weekly_close_signal = "STRONG_BULL_CLOSE"
        elif ("LL" in last_structure or "LH" in last_structure) and wcp < 0.30:
            weekly_close_signal = "STRONG_BEAR_CLOSE"
        elif wcp < 0.30:
            weekly_close_signal = "WEAK_CLOSE"
        elif wcp > 0.70:
            weekly_close_signal = "STRONG_CLOSE"
        else:
            weekly_close_signal = "NEUTRAL_CLOSE"

        return RangeContext(
            trend=trend,
            range_state=range_state,
            compression_ratio=round(compression_ratio, 3),
            ll_count=ll_count,
            hh_count=hh_count,
            lh_count=lh_count,
            hl_count=hl_count,
            total_periods=total_periods,
            near_support=near_support,
            near_resistance=near_resistance,
            breakout_watch=breakout_watch,
            breakdown_watch=breakdown_watch,
            weekly_close_position=round(float(wcp), 3),
            weekly_close_signal=weekly_close_signal,
            last_week_structure=last_structure,
        )


# =============================================================================
# EXTENDED V2 CONTEXT (adds range structure + order flow)
# =============================================================================

@dataclass
class V2ContextExtended(V2Context):
    """V2Context plus range structure and order flow"""
    range_structure: RangeContext = field(default_factory=RangeContext)
    order_flow: OrderFlowContext = field(default_factory=OrderFlowContext)

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d['range_structure'] = self.range_structure.to_dict()
        d['order_flow'] = self.order_flow.to_dict()
        return d


# =============================================================================
# MARKET SCANNER V2 (Extended)
# =============================================================================

class MarketScanner(BaseMarketScanner):
    """
    Extended Market Scanner — inherits all V2 capabilities from
    market_scanner_v2.MarketScanner, adds:

    - Enhanced Polygon snapshot quotes (min bar → quote mid → trade → day)
    - get_order_flow_analysis() for candle-based buying/selling pressure
    - Range structure integration into enriched analysis
    - vp_bars parameter for VP window sizing
    - Smart days_back auto-calculation per timeframe

    Usage:
        from finnhub_scanner_v2 import MarketScanner
        scanner = MarketScanner()

        # V1 compatible
        result = scanner.analyze("META", "1HR")

        # V2 enriched with range structure + order flow
        result, ctx = scanner.analyze_enriched("META")
        print(ctx.range_structure.trend, ctx.order_flow.flow_bias)
    """

    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.calc = TechnicalCalculator()  # Use extended calculator

    # =========================================================================
    # ENHANCED POLYGON SNAPSHOT QUOTES (from V1 finnhub_scanner)
    # =========================================================================

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Enhanced quote — uses Polygon snapshot API with multi-level fallback:
        min_bar → quote_mid → last_trade → day_close → prev_close →
        then falls through to Alpaca → Finnhub → yfinance.
        """
        # Try Polygon snapshot first (richer than base class simple trade)
        if self.polygon_client:
            try:
                snapshot = self.polygon_client.get_snapshot_ticker("stocks", symbol.upper())
                if snapshot:
                    current_price = None
                    source = None
                    open_price = high_price = low_price = prev_close = None

                    # Priority 1: Min bar (most real-time on Stocks Starter)
                    if hasattr(snapshot, 'min') and snapshot.min and hasattr(snapshot.min, 'close'):
                        current_price = float(snapshot.min.close)
                        source = 'polygon_min_bar'

                    # Priority 2: Quote midpoint
                    if current_price is None and hasattr(snapshot, 'last_quote') and snapshot.last_quote:
                        bid = getattr(snapshot.last_quote, 'bid', None) or getattr(snapshot.last_quote, 'p', None)
                        ask = getattr(snapshot.last_quote, 'ask', None) or getattr(snapshot.last_quote, 'P', None)
                        if bid and ask:
                            current_price = (float(bid) + float(ask)) / 2
                            source = 'polygon_quote_mid'

                    # Priority 3: Last trade
                    if current_price is None and hasattr(snapshot, 'last_trade') and snapshot.last_trade:
                        if hasattr(snapshot.last_trade, 'price') and snapshot.last_trade.price:
                            current_price = float(snapshot.last_trade.price)
                            source = 'polygon_snapshot_trade'

                    # Priority 4: Day close
                    if current_price is None and hasattr(snapshot, 'day') and snapshot.day:
                        if hasattr(snapshot.day, 'close') and snapshot.day.close:
                            current_price = float(snapshot.day.close)
                            source = 'polygon_snapshot_day'

                    # Priority 5: Prev day close
                    if current_price is None and hasattr(snapshot, 'prev_day') and snapshot.prev_day:
                        if hasattr(snapshot.prev_day, 'close') and snapshot.prev_day.close:
                            current_price = float(snapshot.prev_day.close)
                            source = 'polygon_snapshot_prev'

                    if current_price:
                        if hasattr(snapshot, 'day') and snapshot.day:
                            open_price = float(snapshot.day.open) if hasattr(snapshot.day, 'open') and snapshot.day.open else None
                            high_price = float(snapshot.day.high) if hasattr(snapshot.day, 'high') and snapshot.day.high else None
                            low_price = float(snapshot.day.low) if hasattr(snapshot.day, 'low') and snapshot.day.low else None
                        if hasattr(snapshot, 'prev_day') and snapshot.prev_day:
                            prev_close = float(snapshot.prev_day.close) if hasattr(snapshot.prev_day, 'close') and snapshot.prev_day.close else None

                        change = current_price - prev_close if prev_close else None
                        change_pct = (change / prev_close * 100) if prev_close and change else None

                        return {
                            'current': current_price,
                            'open': open_price, 'high': high_price,
                            'low': low_price, 'prev_close': prev_close,
                            'change': change, 'change_pct': change_pct,
                            'timestamp': datetime.now(), 'source': source,
                        }
            except Exception:
                pass

            # Fallback: latest 1-minute bar (works on free tier — 15-min delayed)
            try:
                today = date.today()
                aggs = self.polygon_client.get_aggs(
                    ticker=symbol, multiplier=1, timespan="minute",
                    from_=today - timedelta(days=1), to=today,
                    limit=5, sort="desc"
                )
                if aggs and len(aggs) > 0:
                    latest = aggs[0]
                    return {
                        'current': float(latest.close),
                        'open': float(latest.open) if hasattr(latest, 'open') else None,
                        'high': float(latest.high) if hasattr(latest, 'high') else None,
                        'low': float(latest.low) if hasattr(latest, 'low') else None,
                        'prev_close': None, 'change': None, 'change_pct': None,
                        'timestamp': datetime.fromtimestamp(latest.timestamp / 1000) if hasattr(latest, 'timestamp') else datetime.now(),
                        'source': 'polygon_latest_bar',
                    }
            except Exception:
                pass

            # Fallback: previous close
            try:
                prev = self.polygon_client.get_previous_close(symbol)
                if prev and prev.results and len(prev.results) > 0:
                    r = prev.results[0]
                    return {
                        'current': float(r.close),
                        'open': float(r.open) if hasattr(r, 'open') else None,
                        'high': float(r.high) if hasattr(r, 'high') else None,
                        'low': float(r.low) if hasattr(r, 'low') else None,
                        'prev_close': float(r.close),
                        'change': None, 'change_pct': None,
                        'timestamp': datetime.now(), 'source': 'polygon_prev_close',
                    }
            except Exception:
                pass

        # Fall through to base class (Alpaca → Finnhub → yfinance)
        return super().get_quote(symbol)

    # =========================================================================
    # ORDER FLOW ANALYSIS (Polygon candle pressure)
    # =========================================================================

    def get_order_flow_analysis(self, symbol: str,
                                 timeframe: str = "1HR",
                                 vp_period: str = "swing") -> OrderFlowContext:
        """
        Lite order flow analysis from candle data (Polygon Stocks Starter).
        Analyzes buying/selling pressure from price action and volume.

        Args:
            symbol: Stock ticker
            timeframe: Bar size — 5MIN, 15MIN, 30MIN, 1HR, 2HR, 4HR
            vp_period: Lookback — 'day' (1d), 'swing' (5d), 'position' (20d)

        Returns:
            OrderFlowContext with pressure, bias, momentum
        """
        ctx = OrderFlowContext(timeframe=timeframe.upper(), period=vp_period.lower())
        if self.polygon_client is None:
            return ctx

        try:
            # Map timeframe to Polygon params
            tf_map = {
                "5MIN": (5, "minute", 1), "15MIN": (15, "minute", 1),
                "30MIN": (15, "minute", 2), "1HR": (15, "minute", 4),
                "2HR": (15, "minute", 8), "4HR": (15, "minute", 16),
            }
            multiplier, timespan, bar_mult = tf_map.get(timeframe.upper(), (15, "minute", 4))

            period_config = {
                "day": {"days": 5, "bars": {"5MIN": 78, "15MIN": 26, "30MIN": 13, "1HR": 7, "2HR": 4, "4HR": 2}},
                "swing": {"days": 12, "bars": {"5MIN": 390, "15MIN": 130, "30MIN": 65, "1HR": 35, "2HR": 18, "4HR": 9}},
                "position": {"days": 35, "bars": {"5MIN": 500, "15MIN": 300, "30MIN": 150, "1HR": 140, "2HR": 70, "4HR": 35}},
            }
            config = period_config.get(vp_period.lower(), period_config["swing"])
            days_back = config["days"]
            bar_limit = config["bars"].get(timeframe.upper(), 35) * bar_mult

            today = date.today()
            aggs = self.polygon_client.get_aggs(
                ticker=symbol, multiplier=multiplier, timespan=timespan,
                from_=today - timedelta(days=days_back), to=today,
                limit=bar_limit, sort="desc"
            )

            if not aggs or len(aggs) < 3:
                return ctx

            # Analyze candle pressure
            buy_candles = sell_candles = 0
            buy_volume = sell_volume = total_volume = 0
            high_vol_bars = 0

            volumes = [getattr(b, 'volume', 0) or 0 for b in aggs]
            avg_vol = sum(volumes) / len(volumes) if volumes else 0

            for bar in aggs:
                vol = getattr(bar, 'volume', 0) or 0
                open_p = getattr(bar, 'open', 0) or 0
                close_p = getattr(bar, 'close', 0) or 0
                total_volume += vol

                if close_p > open_p:
                    buy_candles += 1
                    buy_volume += vol
                elif close_p < open_p:
                    sell_candles += 1
                    sell_volume += vol

                if vol > avg_vol * 1.5:
                    high_vol_bars += 1

            buy_pressure = (buy_volume / total_volume * 100) if total_volume > 0 else 50
            sell_pressure = (sell_volume / total_volume * 100) if total_volume > 0 else 50

            # Snapshot for spread and volume ratio
            spread_pct = 0.0
            vol_ratio = 1.0
            volume_spike = False

            try:
                snap = self.polygon_client.get_snapshot_ticker("stocks", symbol)
                if snap:
                    if hasattr(snap, 'last_quote') and snap.last_quote:
                        bid = getattr(snap.last_quote, 'bid', 0) or getattr(snap.last_quote, 'p', 0) or 0
                        ask = getattr(snap.last_quote, 'ask', 0) or getattr(snap.last_quote, 'P', 0) or 0
                        if bid > 0 and ask > 0:
                            spread_pct = ((ask - bid) / bid) * 100

                    today_d = getattr(snap, 'day', None)
                    prev_d = getattr(snap, 'prev_day', None)
                    if today_d and prev_d:
                        tv = today_d.v if hasattr(today_d, 'v') else 0
                        pv = prev_d.v if hasattr(prev_d, 'v') else 0
                        if pv > 0:
                            vol_ratio = tv / pv
                            volume_spike = vol_ratio > 1.5
            except Exception:
                pass

            # Flow bias
            if buy_pressure > 60:
                flow_bias = "BULLISH"
            elif sell_pressure > 60:
                flow_bias = "BEARISH"
            else:
                flow_bias = "NEUTRAL"

            # Recent momentum
            recent = aggs[:5] if len(aggs) >= 5 else aggs
            recent_buy = sum(1 for b in recent if (getattr(b, 'close', 0) or 0) > (getattr(b, 'open', 0) or 0))
            recent_sell = len(recent) - recent_buy
            momentum = "STRONG BUY" if recent_buy >= 4 else "STRONG SELL" if recent_sell >= 4 else "MIXED"

            ctx.buy_pressure = round(buy_pressure, 1)
            ctx.sell_pressure = round(sell_pressure, 1)
            ctx.flow_bias = flow_bias
            ctx.total_bars_analyzed = len(aggs)
            ctx.high_volume_bars = high_vol_bars
            ctx.buy_candles = buy_candles
            ctx.sell_candles = sell_candles
            ctx.volume_ratio = round(vol_ratio, 2)
            ctx.volume_spike = volume_spike
            ctx.momentum = momentum
            ctx.spread_pct = round(spread_pct, 3)

        except Exception as e:
            pass

        return ctx

    # =========================================================================
    # OVERRIDDEN: ANALYZE (with vp_bars + smart days_back)
    # =========================================================================

    def analyze(self, symbol: str, timeframe: str = "1HR",
                days_back: int = None, vp_bars: int = 30):
        """
        Single timeframe analysis with vp_bars windowing.

        Args:
            symbol: Stock symbol
            timeframe: 5MIN, 15MIN, 30MIN, 1HR, 2HR, 4HR, DAILY
            days_back: Days of history (auto if None)
            vp_bars: Number of bars for VP calculation (visible range)
        """
        if self.system is None:
            return None

        # Smart days_back
        if days_back is None:
            days_map = {
                "5MIN": 1, "15MIN": 2, "30MIN": 5,
                "1HR": 7, "2HR": 15, "4HR": 30, "DAILY": 60
            }
            days_back = days_map.get(timeframe.upper(), 7)

        # Native resolution for accurate VP
        resolution_map = {
            "5MIN": "5", "15MIN": "15", "30MIN": "30",
            "1HR": "60", "2HR": "60", "4HR": "60", "DAILY": "D"
        }
        resolution = resolution_map.get(timeframe.upper(), "60")

        df = self._get_candles(symbol, resolution, days_back)
        if df is None or len(df) < 10:
            return None

        if timeframe.upper() in ("2HR", "4HR"):
            df = self._resample_to_timeframe(df, timeframe)
        if len(df) < 10:
            return None

        # VP window (like Webull visible range)
        if len(df) > vp_bars:
            df = df.tail(vp_bars)

        quote = self.get_quote(symbol)
        current_price = quote['current'] if quote else df['close'].iloc[-1]

        poc, vah, val = self.calc.calculate_volume_profile(df)
        vwap = self.calc.calculate_vwap(df)
        rsi = self.calc.calculate_rsi(df)
        rvol = self.calc.calculate_relative_volume(df)
        vol_trend = self.calc.calculate_volume_trend(df)
        vol_div = self.calc.detect_volume_divergence(df)
        atr = self.calc.calculate_atr(df)

        has_rejection = False
        if current_price < val:
            has_rejection = self.calc.is_rejection_candle(df, "bullish")
        elif current_price > vah:
            has_rejection = self.calc.is_rejection_candle(df, "bearish")

        return self.system.analyze(
            symbol=symbol, price=current_price,
            vah=vah, poc=poc, val=val, vwap=vwap, rsi=rsi,
            timeframe=timeframe, rvol=rvol,
            volume_trend=vol_trend, volume_divergence=vol_div,
            atr=atr, has_rejection=has_rejection
        )

    # =========================================================================
    # V2: BUILD EXTENDED CONTEXT (range structure + order flow)
    # =========================================================================

    def build_v2_context_extended(self, symbol: str,
                                    timeframe: str = "1HR",
                                    days_back: int = 60,
                                    include_order_flow: bool = True) -> V2ContextExtended:
        """
        Build extended V2 context with range structure and order flow.

        Args:
            symbol: Stock symbol
            timeframe: Analysis timeframe
            days_back: History depth
            include_order_flow: Whether to fetch order flow (requires Polygon)

        Returns:
            V2ContextExtended with all enrichment + range + order flow
        """
        # Get base V2 context
        base_ctx = self.build_v2_context(symbol, timeframe=timeframe, days_back=days_back)

        # Create extended context with base fields
        ctx = V2ContextExtended(
            symbol=base_ctx.symbol,
            current_price=base_ctx.current_price,
            timestamp=base_ctx.timestamp,
            rsi=base_ctx.rsi,
            rsi_2hr=base_ctx.rsi_2hr,
            atr=base_ctx.atr,
            rvol=base_ctx.rvol,
            volume_trend=base_ctx.volume_trend,
            weekly=base_ctx.weekly,
            squeeze=base_ctx.squeeze,
            iv=base_ctx.iv,
            vp=base_ctx.vp,
        )

        # Add range structure from weekly data
        try:
            import yfinance as yf_mod
            ticker = yf_mod.Ticker(symbol)
            df_weekly = ticker.history(period="6mo", interval="1wk")
            df_daily = ticker.history(period="1mo", interval="1d")

            if not df_weekly.empty and len(df_weekly) >= 6:
                ctx.range_structure = self.calc.calculate_range_structure(
                    df_weekly, df_daily, ctx.current_price
                )
                # Sync range structure with weekly context for downstream compat
                ctx.weekly.trend = ctx.range_structure.trend
                ctx.weekly.last_week_structure = ctx.range_structure.last_week_structure
                ctx.weekly.weekly_close_position = ctx.range_structure.weekly_close_position
                ctx.weekly.weekly_close_signal = ctx.range_structure.weekly_close_signal

                # Update supports_long/short from range structure
                wcp = ctx.range_structure.weekly_close_position
                t = ctx.range_structure.trend
                wcs = ctx.range_structure.weekly_close_signal
                ctx.weekly.supports_long = (
                    t in ("UPTREND", "STRONG_UPTREND") or
                    wcs in ("BULLISH_REVERSAL", "STRONG_BULL_CLOSE") or
                    (t == "NEUTRAL" and wcp > 0.6)
                )
                ctx.weekly.supports_short = (
                    t in ("DOWNTREND", "STRONG_DOWNTREND") or
                    wcs in ("BEARISH_REVERSAL", "STRONG_BEAR_CLOSE") or
                    (t == "NEUTRAL" and wcp < 0.4)
                )
        except Exception:
            pass

        # Add order flow
        if include_order_flow and self.polygon_client:
            ctx.order_flow = self.get_order_flow_analysis(symbol, timeframe)

        return ctx

    # =========================================================================
    # V2: ENRICHED ANALYSIS (overridden to use extended context)
    # =========================================================================

    def analyze_enriched(self, symbol: str, timeframe: str = "1HR",
                          days_back: int = 60,
                          include_order_flow: bool = True) -> Tuple[Optional[object], V2ContextExtended]:
        """
        V2 enriched analysis — returns (AnalysisResult, V2ContextExtended).

        Usage:
            result, ctx = scanner.analyze_enriched("NVDA")
            print(f"Signal: {result.signal}")
            print(f"Weekly: {ctx.range_structure.trend}")
            print(f"Squeeze: {ctx.squeeze.is_squeezed}")
            print(f"Order flow: {ctx.order_flow.flow_bias}")
        """
        ctx = self.build_v2_context_extended(
            symbol, timeframe, days_back, include_order_flow
        )
        result = self.analyze(symbol, timeframe, days_back)
        return result, ctx

    def scan_enriched(self, symbols: List[str],
                       timeframe: str = "1HR",
                       days_back: int = 60,
                       include_order_flow: bool = False) -> List[Tuple[Optional[object], V2ContextExtended]]:
        """
        V2 batch scan — returns list of (AnalysisResult, V2ContextExtended)
        sorted by quality: squeeze → range compressing → weekly aligned → RSI extremes.

        Note: include_order_flow=False by default for batch scans (rate limits).
        """
        results = []
        for i, symbol in enumerate(symbols):
            try:
                result, ctx = self.analyze_enriched(
                    symbol, timeframe, days_back, include_order_flow
                )
                results.append((result, ctx))
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
            if i < len(symbols) - 1:
                time.sleep(1)

        def quality_key(item):
            _, ctx = item
            score = 0
            if ctx.squeeze.is_squeezed:
                score += 50
            if ctx.range_structure.range_state == "COMPRESSING":
                score += 40
            if ctx.weekly.supports_long or ctx.weekly.supports_short:
                score += 30
            if ctx.rsi < 35 or ctx.rsi > 65:
                score += 20
            if ctx.order_flow.flow_bias != "NEUTRAL":
                score += 10
            return -score

        results.sort(key=quality_key)
        return results

    # =========================================================================
    # DISPLAY METHODS
    # =========================================================================

    def print_extended_context(self, ctx: V2ContextExtended) -> str:
        """Print extended V2 context with range structure + order flow"""
        lines = [self.print_v2_context(ctx)]

        rs = ctx.range_structure
        lines.extend([
            "",
            f"📐 RANGE STRUCTURE:",
            f"   Trend: {rs.trend}  State: {rs.range_state}  Compression: {rs.compression_ratio:.2f}x",
            f"   HH:{rs.hh_count} HL:{rs.hl_count} | LH:{rs.lh_count} LL:{rs.ll_count}  ({rs.total_periods} weeks)",
            f"   Last Week: {rs.last_week_structure}  Close: {rs.weekly_close_position:.0%} ({rs.weekly_close_signal})",
            f"   {'🟢 Near Support' if rs.near_support else ''}{'🔴 Near Resistance' if rs.near_resistance else ''}",
            f"   Breakout Watch: ${rs.breakout_watch:.2f}  Breakdown Watch: ${rs.breakdown_watch:.2f}",
        ])

        of = ctx.order_flow
        if of.total_bars_analyzed > 0:
            lines.extend([
                "",
                f"📊 ORDER FLOW ({of.timeframe}/{of.period}):",
                f"   Bias: {of.flow_bias}  Buy: {of.buy_pressure:.0f}%  Sell: {of.sell_pressure:.0f}%",
                f"   Bars: {of.total_bars_analyzed}  HV Bars: {of.high_volume_bars}",
                f"   Vol Ratio: {of.volume_ratio:.2f}x {'⚡ SPIKE' if of.volume_spike else ''}",
                f"   Momentum: {of.momentum}  Spread: {of.spread_pct:.3f}%",
            ])

        return "\n".join(lines)


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

FinnhubScanner = MarketScanner


# =============================================================================
# QUICK FUNCTIONS
# =============================================================================

def quick_analyze(symbol: str, api_key: str = None):
    scanner = MarketScanner(api_key)
    return scanner.analyze(symbol)


def quick_mtf(symbol: str, api_key: str = None):
    scanner = MarketScanner(api_key)
    return scanner.analyze_mtf(symbol)


def quick_scan(symbols: List[str], api_key: str = None):
    scanner = MarketScanner(api_key)
    return scanner.scan_symbols(symbols)


def quick_v2_extended(symbol: str) -> V2ContextExtended:
    """Quick extended V2 context with range structure + order flow"""
    scanner = MarketScanner()
    return scanner.build_v2_context_extended(symbol)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  FINNHUB SCANNER V2 (Extended Market Scanner)")
    print("=" * 60)

    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    try:
        scanner = MarketScanner()

        print(f"\n📊 Building extended V2 context for {symbol}...")
        ctx = scanner.build_v2_context_extended(symbol)
        print(scanner.print_extended_context(ctx))

        if chart_system_available:
            print(f"\n📊 Analysis for {symbol}...")
            result = scanner.analyze(symbol, "1HR")
            if result:
                print(scanner.print_analysis(result))

    except Exception as e:
        print(f"Error: {e}")
        print("\nUsage: python finnhub_scanner_v2.py [SYMBOL]")
