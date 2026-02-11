"""
Integrated Scanner V2 — The Capstone
======================================
Combines MTF Auction, Overnight/Gap, and Dual Setup analysis with
full V2 enrichment (squeeze, weekly, IV, VP shape) into a single
pre-market and intraday decision engine.

V1 → V2 CHANGES:
- Dynamic MTF/overnight weighting based on overnight signal strength:
  * Spring-loaded squeeze break gaps → overnight weight UP (50-60%)
  * Strong directional gaps → overnight weight 40% (standard)
  * Neutral no-gap overnights → overnight weight DOWN (20-25%)
  * Hardcoded 60/40 eliminated
- V2Context integration: squeeze, weekly, IV, VP shape flow through
  to all sub-scanners and aggregate scoring
- DualSetupGeneratorV2 wired in for trade plan enrichment with
  weekly/squeeze/IV context driving options strategy selection
- IntegratedAnalysis gains: v2_context, squeeze_setup_bonus,
  weekly_alignment, iv_regime, dual_setup fields
- All V1 methods preserved + backward compatible

Author: Rob's Trading Systems (Strategic Edge Flow)
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# V2 imports (preferred)
try:
    from mtf_auction_scanner_v2 import (
        MTFAuctionScanner, ScanResult, SignalState, Timeframe,
        VolumeProfileEngine
    )
    _mtf_v2 = True
except ImportError:
    from mtf_auction_scanner import (
        MTFAuctionScanner, ScanResult, SignalState, Timeframe,
        VolumeProfileEngine
    )
    _mtf_v2 = False

try:
    from overnight_model_v2 import (
        OvernightModelV2, OvernightPredictionEngine, OvernightPrediction,
    )
    _overnight_v2 = True
except ImportError:
    from overnight_model import (
        OvernightPredictionEngine, OvernightPrediction, OvernightBias,
        GapType, GapFillProbability
    )
    _overnight_v2 = False

try:
    from dual_setup_generator_v2 import DualSetupGeneratorV2, DualSetupResult
    _dual_v2 = True
except ImportError:
    _dual_v2 = False

try:
    from market_scanner_v2 import V2Context, MarketScanner as BaseMarketScanner
    _v2ctx_available = True
except ImportError:
    _v2ctx_available = False
    class V2Context:
        pass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IntegratedAnalysis:
    """Complete analysis combining intraday, overnight, and V2 context"""
    symbol: str
    analysis_time: datetime

    # Components
    mtf_scan: ScanResult
    overnight: Optional[OvernightPrediction]
    dual_setup: Optional[Any] = None          # DualSetupResult when available

    # Combined signal
    combined_bias: str                         # STRONG_BULL, BULL, NEUTRAL, BEAR, STRONG_BEAR
    combined_confidence: float

    # Dynamic weighting (V2)
    mtf_weight: float = 0.60
    overnight_weight: float = 0.40
    weight_reason: str = ""

    # Scenarios
    high_scenario_prob: float = 0.0
    low_scenario_prob: float = 0.0
    chop_scenario_prob: float = 0.0

    # Key levels
    key_levels: Dict[str, float] = field(default_factory=dict)

    # Trade plan
    trade_plan: Optional[Dict[str, Any]] = None

    # V2 context summary
    v2_context: Optional[object] = None
    squeeze_active: bool = False
    squeeze_days: int = 0
    weekly_trend: str = ""
    iv_regime: str = ""
    vp_shape: str = ""

    # Notes
    notes: List[str] = field(default_factory=list)


# =============================================================================
# OVERNIGHT SIGNAL STRENGTH CLASSIFIER
# =============================================================================

def _classify_overnight_strength(overnight: Optional[OvernightPrediction],
                                  squeeze_active: bool = False) -> Tuple[float, float, str]:
    """
    Dynamic MTF/Overnight weighting based on overnight signal strength.

    Returns:
        (mtf_weight, overnight_weight, reason)

    Rules:
        - Squeeze break gap (squeeze active + large gap): overnight 50-60%
        - Strong directional gap (>0.5%): standard 60/40
        - Moderate gap (0.2-0.5%): slightly less overnight 65/35
        - Flat/no gap (<0.2%): MTF dominates 75-80/20-25
        - No overnight data: MTF only 100/0
    """
    if overnight is None:
        return 1.0, 0.0, "No overnight data — MTF only"

    # Extract gap info — handle both V1 and V2 structures
    gap_pct = 0.0
    gap_is_large = False
    bias_strength = "NEUTRAL"

    if _overnight_v2:
        # V2 OvernightPrediction
        if hasattr(overnight, 'gap') and overnight.gap:
            gap_pct = abs(getattr(overnight.gap, 'gap_pct', 0) or 0)
            gap_is_large = gap_pct >= 0.75
        bias_val = getattr(overnight, 'bias', 'NEUTRAL')
        if isinstance(bias_val, str):
            bias_strength = bias_val
        else:
            bias_strength = getattr(bias_val, 'value', str(bias_val))
        confidence = getattr(overnight, 'confidence', 50)
    else:
        # V1 OvernightPrediction
        if hasattr(overnight, 'gap') and overnight.gap:
            gap_pct = abs(getattr(overnight.gap, 'gap_pct', 0) or 0)
            gt = getattr(overnight.gap, 'gap_type', None)
            if gt:
                gap_is_large = gt in [GapType.GAP_UP_LARGE, GapType.GAP_DOWN_LARGE]
        bias_val = getattr(overnight, 'bias', None)
        if bias_val:
            bias_strength = getattr(bias_val, 'value', str(bias_val))
        confidence = 50

    # Squeeze break gap — the spring-loaded scenario
    if squeeze_active and gap_is_large:
        # Squeeze + big gap = overnight signal is very meaningful
        if gap_pct >= 1.0:
            return 0.40, 0.60, f"SQUEEZE BREAK GAP ({gap_pct:.1f}%) — overnight dominates"
        else:
            return 0.50, 0.50, f"Squeeze gap ({gap_pct:.1f}%) — balanced weight"

    # Strong directional gap
    if gap_pct >= 0.75:
        return 0.55, 0.45, f"Large gap ({gap_pct:.1f}%) — overnight significant"

    if gap_pct >= 0.5:
        return 0.60, 0.40, f"Moderate gap ({gap_pct:.1f}%) — standard weight"

    if gap_pct >= 0.2:
        return 0.65, 0.35, f"Small gap ({gap_pct:.1f}%) — MTF slightly favored"

    # Flat/no gap
    if bias_strength in ("NEUTRAL", "FLAT"):
        return 0.80, 0.20, f"No gap, neutral overnight — MTF dominates"

    return 0.75, 0.25, f"Tiny gap ({gap_pct:.2f}%) — MTF leads"


# =============================================================================
# INTEGRATED SCANNER V2
# =============================================================================

class IntegratedScanner:
    """
    Combines MTF Auction + Overnight/Gap + Dual Setup + V2 Context
    into a single analysis engine.

    For Traders:
    -----------
    1. What happened overnight? (Gap, direction, squeeze break?)
    2. Where are we in the auction? (Value area, flow, momentum)
    3. What's the weekly/squeeze context? (V2 enrichment)
    4. What's the combined bias? (Dynamic-weighted synthesis)
    5. What's the trade plan? (Entries, stops, targets, options)

    For Programmers:
    ---------------
    Orchestrates V2 scanners with dynamic weighting:
    - Squeeze break gaps → overnight signal carries more weight
    - Flat overnights → MTF auction dominates
    - V2Context flows through to all sub-scanners

    Usage:
        scanner = IntegratedScanner()

        # V1 compatible
        analysis = scanner.analyze(df, "META")

        # V2 enriched
        analysis = scanner.analyze(df, "META", v2_context=ctx)

        # V2 with auto-built context
        analysis = scanner.analyze_enriched(df, "META")
    """

    def __init__(self):
        self.mtf_scanner = MTFAuctionScanner()
        self.overnight_engine = OvernightPredictionEngine()
        self.vp_engine = VolumeProfileEngine()

        if _dual_v2:
            self.dual_generator = DualSetupGeneratorV2()
        else:
            self.dual_generator = None

        if _v2ctx_available:
            self.base_scanner = BaseMarketScanner()
        else:
            self.base_scanner = None

    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================

    def analyze(self,
                df: pd.DataFrame,
                symbol: str = "UNKNOWN",
                v2_context: Optional[object] = None) -> IntegratedAnalysis:
        """
        Run complete integrated analysis.

        Args:
            df: OHLCV dataframe with datetime index
            symbol: Ticker symbol
            v2_context: Optional V2Context (squeeze/weekly/IV/VP)

        Returns:
            IntegratedAnalysis with dynamic-weighted combined signal
        """
        # Extract V2 context fields for use throughout
        squeeze_active = False
        squeeze_days = 0
        weekly_trend = ""
        iv_regime = ""
        vp_shape = ""

        if v2_context is not None and _v2ctx_available:
            if hasattr(v2_context, 'squeeze') and v2_context.squeeze:
                squeeze_active = getattr(v2_context.squeeze, 'is_squeezed', False)
                squeeze_days = getattr(v2_context.squeeze, 'squeeze_days', 0)
            if hasattr(v2_context, 'weekly') and v2_context.weekly:
                weekly_trend = getattr(v2_context.weekly, 'trend', '')
            if hasattr(v2_context, 'iv') and v2_context.iv:
                iv_regime = getattr(v2_context.iv, 'iv_regime', '')
            if hasattr(v2_context, 'vp') and v2_context.vp:
                vp_shape = getattr(v2_context.vp, 'profile_shape', '')

        # --- Run MTF scan (with V2 context if available) ---
        if _mtf_v2 and v2_context is not None:
            mtf_result = self.mtf_scanner.scan(df, symbol=symbol, v2_context=v2_context)
        else:
            mtf_result = self.mtf_scanner.scan(df, symbol=symbol)

        # --- Get prior day VP for overnight analysis ---
        prior_poc, prior_vah, prior_val = self._get_prior_day_vp(df)

        # --- Run overnight prediction ---
        if _overnight_v2:
            overnight = self.overnight_engine.analyze(df, symbol=symbol)
        else:
            overnight = self.overnight_engine.predict(
                df, symbol=symbol,
                prior_poc=prior_poc, prior_vah=prior_vah, prior_val=prior_val
            )

        # --- Dynamic weighting ---
        mtf_weight, overnight_weight, weight_reason = _classify_overnight_strength(
            overnight, squeeze_active
        )

        # --- Combine signals ---
        combined_bias, combined_conf = self._combine_signals(
            mtf_result, overnight, mtf_weight, overnight_weight, v2_context
        )

        # --- Scenario probabilities ---
        high_prob, low_prob, chop_prob = self._calculate_scenarios(
            mtf_result, overnight, combined_bias
        )

        # --- Key levels ---
        key_levels = self._compile_key_levels(mtf_result, overnight)

        # --- Dual setup (V2) ---
        dual_setup = None
        if self.dual_generator is not None:
            try:
                dual_setup = self.dual_generator.analyze(df, symbol)
            except Exception:
                dual_setup = None

        # --- Trade plan ---
        trade_plan = self._generate_trade_plan(
            mtf_result, overnight, combined_bias, combined_conf,
            key_levels, dual_setup, v2_context
        )

        # --- Notes ---
        notes = self._compile_notes(
            mtf_result, overnight, combined_bias,
            mtf_weight, overnight_weight, weight_reason,
            v2_context
        )

        return IntegratedAnalysis(
            symbol=symbol,
            analysis_time=datetime.now(),
            mtf_scan=mtf_result,
            overnight=overnight,
            dual_setup=dual_setup,
            combined_bias=combined_bias,
            combined_confidence=combined_conf,
            mtf_weight=mtf_weight,
            overnight_weight=overnight_weight,
            weight_reason=weight_reason,
            high_scenario_prob=high_prob,
            low_scenario_prob=low_prob,
            chop_scenario_prob=chop_prob,
            key_levels=key_levels,
            trade_plan=trade_plan,
            v2_context=v2_context,
            squeeze_active=squeeze_active,
            squeeze_days=squeeze_days,
            weekly_trend=weekly_trend,
            iv_regime=iv_regime,
            vp_shape=vp_shape,
            notes=notes,
        )

    def analyze_enriched(self,
                          df: pd.DataFrame,
                          symbol: str = "UNKNOWN") -> IntegratedAnalysis:
        """
        V2 enriched: auto-builds V2Context then runs full analysis.
        Requires market_scanner_v2 + Polygon API key.
        Falls back to analyze() without V2 context if unavailable.
        """
        v2_context = None
        if self.base_scanner is not None:
            try:
                v2_context = self.base_scanner.build_v2_context(symbol)
            except Exception:
                v2_context = None

        return self.analyze(df, symbol, v2_context=v2_context)

    # =========================================================================
    # PRIOR DAY VP
    # =========================================================================

    def _get_prior_day_vp(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Extract prior day's VP levels"""
        try:
            daily = df.resample('D').agg({
                'open': 'first', 'high': 'max',
                'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()

            if len(daily) >= 2:
                prior_date = daily.index[-2].date()
                prior_day_df = df[df.index.date == prior_date]
                if len(prior_day_df) > 10:
                    vp = self.vp_engine.calculate(prior_day_df)
                    return vp.poc, vp.vah, vp.val
        except Exception:
            pass
        return None, None, None

    # =========================================================================
    # SIGNAL COMBINATION (DYNAMIC WEIGHTING)
    # =========================================================================

    def _combine_signals(self,
                          mtf: ScanResult,
                          overnight: Optional[OvernightPrediction],
                          mtf_weight: float,
                          overnight_weight: float,
                          v2_context: Optional[object] = None) -> Tuple[str, float]:
        """Combine MTF and overnight signals with dynamic weighting"""

        # --- MTF bull score (0-100) ---
        mtf_bull_score = 50
        if mtf.dominant_signal == SignalState.LONG_SETUP:
            mtf_bull_score = 80 + (mtf.confluence_score / 5)
        elif mtf.dominant_signal == SignalState.YELLOW:
            mtf_bull_score = 50 + (mtf.high_scenario_prob - 0.5) * 30
        elif mtf.dominant_signal == SignalState.SHORT_SETUP:
            mtf_bull_score = 20 - (mtf.confluence_score / 5)
        elif mtf.dominant_signal == SignalState.NEUTRAL:
            mtf_bull_score = 50

        # --- Overnight bull score (0-100) ---
        overnight_bull_score = 50
        if overnight is not None:
            overnight_bull_score = self._extract_overnight_bull_score(overnight)

        # --- Weighted combination ---
        combined_score = (mtf_bull_score * mtf_weight) + (overnight_bull_score * overnight_weight)

        # --- V2 context adjustments ---
        if v2_context is not None and _v2ctx_available:
            # Weekly trend nudge (±5 pts)
            if hasattr(v2_context, 'weekly') and v2_context.weekly:
                wt = getattr(v2_context.weekly, 'trend', '')
                if wt in ('STRONG_UPTREND', 'UPTREND'):
                    combined_score += 5
                elif wt in ('STRONG_DOWNTREND', 'DOWNTREND'):
                    combined_score -= 5

            # Squeeze tension bonus — reinforces whatever direction is dominant
            if hasattr(v2_context, 'squeeze') and v2_context.squeeze:
                if getattr(v2_context.squeeze, 'is_squeezed', False):
                    mom_dir = getattr(v2_context.squeeze, 'momentum_direction', 'NEUTRAL')
                    if mom_dir == 'UP' and combined_score > 50:
                        combined_score += 5
                    elif mom_dir == 'DOWN' and combined_score < 50:
                        combined_score -= 5

        combined_score = max(0, min(100, combined_score))

        # --- Determine bias ---
        if combined_score >= 75:
            bias = "STRONG_BULL"
        elif combined_score >= 60:
            bias = "BULL"
        elif combined_score <= 25:
            bias = "STRONG_BEAR"
        elif combined_score <= 40:
            bias = "BEAR"
        else:
            bias = "NEUTRAL"

        # --- Confidence ---
        agreement_bonus = 0
        if overnight is not None:
            mtf_bullish = mtf.dominant_signal == SignalState.LONG_SETUP
            mtf_bearish = mtf.dominant_signal == SignalState.SHORT_SETUP

            on_bullish = self._overnight_is_bullish(overnight)
            on_bearish = self._overnight_is_bearish(overnight)

            if (mtf_bullish and on_bullish) or (mtf_bearish and on_bearish):
                agreement_bonus = 15
            elif (mtf_bullish and on_bearish) or (mtf_bearish and on_bullish):
                agreement_bonus = -15

        confidence = min(95, max(20, abs(combined_score - 50) * 2 + agreement_bonus))

        return bias, confidence

    def _extract_overnight_bull_score(self, overnight: OvernightPrediction) -> float:
        """Extract bull score from overnight prediction (handles V1 and V2)"""
        if _overnight_v2:
            # V2: prediction_score is 0-100 (50=neutral, >50=bullish)
            return getattr(overnight, 'prediction_score', 50)
        else:
            # V1: OvernightBias enum
            bias = getattr(overnight, 'bias', None)
            if bias is None:
                return 50
            bias_map = {
                OvernightBias.STRONG_BULLISH: 90,
                OvernightBias.BULLISH: 70,
                OvernightBias.NEUTRAL: 50,
                OvernightBias.BEARISH: 30,
                OvernightBias.STRONG_BEARISH: 10,
            }
            score = bias_map.get(bias, 50)

            # V1 gap adjustment
            if hasattr(overnight, 'gap') and overnight.gap:
                gt = getattr(overnight.gap, 'gap_type', None)
                gfp = getattr(overnight.gap, 'gap_fill_probability', None)
                if gt in [GapType.GAP_UP_LARGE, GapType.GAP_UP_SMALL]:
                    if gfp == GapFillProbability.LOW:
                        score += 10
                elif gt in [GapType.GAP_DOWN_LARGE, GapType.GAP_DOWN_SMALL]:
                    if gfp == GapFillProbability.LOW:
                        score -= 10
            return max(0, min(100, score))

    def _overnight_is_bullish(self, overnight: OvernightPrediction) -> bool:
        if _overnight_v2:
            return getattr(overnight, 'prediction_score', 50) >= 65
        else:
            bias = getattr(overnight, 'bias', None)
            return bias in [OvernightBias.STRONG_BULLISH, OvernightBias.BULLISH]

    def _overnight_is_bearish(self, overnight: OvernightPrediction) -> bool:
        if _overnight_v2:
            return getattr(overnight, 'prediction_score', 50) <= 35
        else:
            bias = getattr(overnight, 'bias', None)
            return bias in [OvernightBias.STRONG_BEARISH, OvernightBias.BEARISH]

    # =========================================================================
    # SCENARIO PROBABILITIES
    # =========================================================================

    def _calculate_scenarios(self,
                              mtf: ScanResult,
                              overnight: Optional[OvernightPrediction],
                              bias: str) -> Tuple[float, float, float]:
        """Calculate HIGH/LOW/CHOP scenario probabilities"""

        high_prob = mtf.high_scenario_prob
        low_prob = mtf.low_scenario_prob

        if overnight is not None:
            if self._overnight_is_bullish(overnight):
                high_prob *= 1.2
                low_prob *= 0.8
            elif self._overnight_is_bearish(overnight):
                high_prob *= 0.8
                low_prob *= 1.2

            # Gap fill → adds chop
            if _overnight_v2:
                gap_fill = getattr(overnight.gap, 'fill_probability', 'moderate') if overnight.gap else 'moderate'
                chop_prob = 0.25 if gap_fill == 'high' else 0.10
            else:
                gfp = getattr(overnight.gap, 'gap_fill_probability', None) if overnight and overnight.gap else None
                chop_prob = 0.25 if gfp == GapFillProbability.HIGH else 0.10
        else:
            chop_prob = getattr(mtf, 'neutral_prob', 0.15)

        # Normalize
        total = high_prob + low_prob + chop_prob
        if total > 0:
            high_prob /= total
            low_prob /= total
            chop_prob /= total

        return high_prob, low_prob, chop_prob

    # =========================================================================
    # KEY LEVELS
    # =========================================================================

    def _compile_key_levels(self,
                             mtf: ScanResult,
                             overnight: Optional[OvernightPrediction]) -> Dict[str, float]:
        levels = {}

        # From MTF (highest available timeframe)
        for tf in [Timeframe.H4, Timeframe.H2, Timeframe.H1, Timeframe.M30]:
            if tf in mtf.timeframe_analyses:
                analysis = mtf.timeframe_analyses[tf]
                if hasattr(analysis, 'volume_profile') and analysis.volume_profile:
                    levels['session_vah'] = analysis.volume_profile.vah
                    levels['session_poc'] = analysis.volume_profile.poc
                    levels['session_val'] = analysis.volume_profile.val
                if hasattr(analysis, 'vwap') and analysis.vwap:
                    levels['vwap'] = analysis.vwap.vwap
                    if hasattr(analysis.vwap, 'upper_band_1'):
                        levels['vwap_upper_1sd'] = analysis.vwap.upper_band_1
                        levels['vwap_lower_1sd'] = analysis.vwap.lower_band_1
                break

        # From overnight
        if overnight is not None:
            if _overnight_v2:
                kl = getattr(overnight, 'key_levels', {})
                if isinstance(kl, dict):
                    for k, v in kl.items():
                        if isinstance(v, (int, float)) and v > 0:
                            levels[k] = v
                # Also extract from sub-structures
                if hasattr(overnight, 'overnight') and overnight.overnight:
                    o = overnight.overnight
                    for attr, label in [
                        ('overnight_high', 'overnight_high'),
                        ('overnight_low', 'overnight_low'),
                        ('overnight_midpoint', 'overnight_mid'),
                    ]:
                        val = getattr(o, attr, None)
                        if val and val > 0:
                            levels[label] = val
                if hasattr(overnight, 'prior_day') and overnight.prior_day:
                    pd_ctx = overnight.prior_day
                    for attr, label in [
                        ('prior_close', 'prior_close'),
                        ('prior_high', 'prior_high'),
                        ('prior_low', 'prior_low'),
                        ('prior_poc', 'prior_poc'),
                        ('prior_vah', 'prior_vah'),
                        ('prior_val', 'prior_val'),
                    ]:
                        val = getattr(pd_ctx, attr, None)
                        if val and val > 0:
                            levels[label] = val
                if hasattr(overnight, 'gap') and overnight.gap:
                    gfl = getattr(overnight.gap, 'gap_fill_level', None)
                    if gfl and gfl > 0:
                        levels['gap_fill'] = gfl
            else:
                # V1 overnight structure
                try:
                    levels.update({
                        'overnight_high': overnight.overnight.overnight_high,
                        'overnight_low': overnight.overnight.overnight_low,
                        'overnight_mid': overnight.overnight.overnight_midpoint,
                        'prior_close': overnight.prior_day.prior_close,
                        'prior_high': overnight.prior_day.prior_high,
                        'prior_low': overnight.prior_day.prior_low,
                        'prior_poc': overnight.prior_day.prior_poc,
                        'prior_vah': overnight.prior_day.prior_vah,
                        'prior_val': overnight.prior_day.prior_val,
                        'gap_fill': overnight.gap.gap_fill_level,
                    })
                except (AttributeError, TypeError):
                    pass

        return {k: v for k, v in levels.items() if v is not None and v > 0}

    # =========================================================================
    # TRADE PLAN
    # =========================================================================

    def _generate_trade_plan(self,
                              mtf: ScanResult,
                              overnight: Optional[OvernightPrediction],
                              bias: str,
                              confidence: float,
                              levels: Dict[str, float],
                              dual_setup: Optional[Any] = None,
                              v2_context: Optional[object] = None) -> Optional[Dict]:
        """Generate trade plan if conditions warrant"""

        if confidence < 50 or bias == "NEUTRAL":
            return None

        # Get current price
        try:
            first_tf = list(mtf.timeframe_analyses.keys())[0]
            current_price = mtf.timeframe_analyses[first_tf].current_price
        except (IndexError, AttributeError):
            return None

        if bias in ("STRONG_BULL", "BULL"):
            direction = "LONG"
            entry_zone_low = levels.get('session_val', current_price * 0.99)
            entry_zone_high = current_price
            stop = levels.get('overnight_low', levels.get('prior_low', current_price * 0.97))
            target1 = levels.get('session_vah', current_price * 1.02)
            target2 = levels.get('prior_high', current_price * 1.04)
        else:
            direction = "SHORT"
            entry_zone_low = current_price
            entry_zone_high = levels.get('session_vah', current_price * 1.01)
            stop = levels.get('overnight_high', levels.get('prior_high', current_price * 1.03))
            target1 = levels.get('session_val', current_price * 0.98)
            target2 = levels.get('prior_low', current_price * 0.96)

        risk = abs(current_price - stop)
        reward = abs(target1 - current_price)
        rr_ratio = reward / risk if risk > 0 else 0

        plan = {
            'direction': direction,
            'bias_strength': bias,
            'confidence': confidence,
            'entry_zone': f"${entry_zone_low:.2f} - ${entry_zone_high:.2f}",
            'stop_loss': stop,
            'target_1': target1,
            'target_2': target2,
            'risk_per_share': risk,
            'reward_risk_ratio': rr_ratio,
            'position_notes': [
                f"Entry on pullback to {entry_zone_low:.2f}" if direction == "LONG"
                    else f"Entry on rally to {entry_zone_high:.2f}",
                f"Stop {'below' if direction == 'LONG' else 'above'} overnight "
                    f"{'low' if direction == 'LONG' else 'high'}",
                "Scale out 50% at Target 1",
            ],
        }

        # Enrich from dual setup if available
        if dual_setup is not None and hasattr(dual_setup, 'preferred_direction'):
            plan['dual_verdict'] = getattr(dual_setup, 'preferred_direction', '')
            plan['dual_verdict_reason'] = getattr(dual_setup, 'verdict_reason', '')
            if hasattr(dual_setup, 'options_strategy') and dual_setup.options_strategy:
                plan['options_strategy'] = dual_setup.options_strategy

        # V2 context notes
        if v2_context is not None and _v2ctx_available:
            if hasattr(v2_context, 'squeeze') and v2_context.squeeze:
                if getattr(v2_context.squeeze, 'is_squeezed', False):
                    days = getattr(v2_context.squeeze, 'squeeze_days', 0)
                    plan['position_notes'].append(
                        f"🔥 SQUEEZE active ({days}d) — expect expansion move"
                    )
            if hasattr(v2_context, 'iv') and v2_context.iv:
                regime = getattr(v2_context.iv, 'iv_regime', '')
                if regime == 'extreme':
                    plan['position_notes'].append(
                        "⚡ IV EXTREME — consider selling premium or widening stops"
                    )
                elif regime == 'elevated':
                    plan['position_notes'].append(
                        "📈 IV elevated — premium rich, consider spreads"
                    )

        return plan

    # =========================================================================
    # NOTES
    # =========================================================================

    def _compile_notes(self,
                        mtf: ScanResult,
                        overnight: Optional[OvernightPrediction],
                        bias: str,
                        mtf_weight: float,
                        overnight_weight: float,
                        weight_reason: str,
                        v2_context: Optional[object] = None) -> List[str]:
        notes = []

        # MTF signal
        notes.append(
            f"MTF Signal: {mtf.dominant_signal.emoji} {mtf.dominant_signal.value} "
            f"({mtf.confluence_score:.0f}% confluence)"
        )

        # Overnight
        if overnight is not None:
            if _overnight_v2:
                bias_str = getattr(overnight, 'bias', 'NEUTRAL')
                bias_emoji = getattr(overnight, 'bias_emoji', '🟡')
                notes.append(f"Overnight: {bias_emoji} {bias_str}")
                if hasattr(overnight, 'gap') and overnight.gap:
                    gap_pct = getattr(overnight.gap, 'gap_pct', 0)
                    notes.append(f"Gap: {gap_pct:+.2f}%")
            else:
                notes.append(f"Overnight: {overnight.bias.emoji} {overnight.bias.value}")
                if overnight.gap:
                    notes.append(f"Gap: {overnight.gap.gap_type.emoji} {overnight.gap.gap_pct:+.2f}%")
                    if overnight.gap.gap_fill_probability == GapFillProbability.HIGH:
                        notes.append("⚠️ Gap likely to fill — watch for fade")
                    elif overnight.gap.gap_fill_probability == GapFillProbability.LOW:
                        notes.append("✅ Gap likely to hold — continuation expected")

        # Dynamic weight
        notes.append(f"Weighting: MTF {mtf_weight:.0%} / Overnight {overnight_weight:.0%} ({weight_reason})")

        # Agreement/conflict
        if overnight is not None:
            mtf_bull = mtf.dominant_signal == SignalState.LONG_SETUP
            mtf_bear = mtf.dominant_signal == SignalState.SHORT_SETUP
            on_bull = self._overnight_is_bullish(overnight)
            on_bear = self._overnight_is_bearish(overnight)

            if (mtf_bull and on_bull) or (mtf_bear and on_bear):
                notes.append("✅ MTF and Overnight AGREE — higher confidence")
            elif (mtf_bull and on_bear) or (mtf_bear and on_bull):
                notes.append("⚠️ MTF and Overnight CONFLICT — reduced confidence")

        # V2 context
        if v2_context is not None and _v2ctx_available:
            ctx_parts = []
            if hasattr(v2_context, 'squeeze') and v2_context.squeeze:
                if getattr(v2_context.squeeze, 'is_squeezed', False):
                    ctx_parts.append(f"SQUEEZE({getattr(v2_context.squeeze, 'squeeze_days', '?')}d)")
            if hasattr(v2_context, 'weekly') and v2_context.weekly:
                ctx_parts.append(f"Weekly:{getattr(v2_context.weekly, 'trend', '?')}")
            if hasattr(v2_context, 'iv') and v2_context.iv:
                ctx_parts.append(f"IV:{getattr(v2_context.iv, 'iv_regime', '?')}")
            if hasattr(v2_context, 'vp') and v2_context.vp:
                ctx_parts.append(f"VP:{getattr(v2_context.vp, 'profile_shape', '?')}")
            if ctx_parts:
                notes.append(f"V2 Context: {' | '.join(ctx_parts)}")

        notes.append(f"Combined Bias: {bias}")

        return notes

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def print_analysis(self, analysis: IntegratedAnalysis) -> str:
        lines = []

        lines.append("=" * 80)
        lines.append(f"🎯 INTEGRATED ANALYSIS: {analysis.symbol}")
        lines.append(f"   Time: {analysis.analysis_time.strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 80)

        # Combined signal
        bias_emoji = {
            "STRONG_BULL": "🟢🟢", "BULL": "🟢",
            "NEUTRAL": "🟡",
            "BEAR": "🔴", "STRONG_BEAR": "🔴🔴"
        }.get(analysis.combined_bias, "⚪")

        lines.append(f"\n{bias_emoji} COMBINED BIAS: {analysis.combined_bias}")
        lines.append(f"   Confidence: {analysis.combined_confidence:.0f}%")
        lines.append(f"   Weighting: MTF {analysis.mtf_weight:.0%} / Overnight {analysis.overnight_weight:.0%}")
        lines.append(f"   Reason: {analysis.weight_reason}")

        # V2 context summary
        if analysis.squeeze_active or analysis.weekly_trend or analysis.iv_regime:
            lines.append(f"\n🔬 V2 CONTEXT:")
            if analysis.squeeze_active:
                lines.append(f"   🔥 SQUEEZE: {analysis.squeeze_days}d active")
            if analysis.weekly_trend:
                lines.append(f"   📈 Weekly: {analysis.weekly_trend}")
            if analysis.iv_regime:
                lines.append(f"   📊 IV: {analysis.iv_regime}")
            if analysis.vp_shape:
                lines.append(f"   📊 VP: {analysis.vp_shape}")

        # Scenarios
        lines.append(f"\n📊 SCENARIO PROBABILITIES:")
        lines.append(f"   HIGH (Bullish): {analysis.high_scenario_prob:.0%}")
        lines.append(f"   LOW (Bearish):  {analysis.low_scenario_prob:.0%}")
        lines.append(f"   CHOP (Range):   {analysis.chop_scenario_prob:.0%}")

        # Component summaries
        lines.append(f"\n📈 MTF ANALYSIS:")
        lines.append(f"   Signal: {analysis.mtf_scan.dominant_signal.emoji} {analysis.mtf_scan.dominant_signal.value}")
        lines.append(f"   Confluence: {analysis.mtf_scan.confluence_score:.0f}%")

        if analysis.overnight is not None:
            lines.append(f"\n🌙 OVERNIGHT ANALYSIS:")
            if _overnight_v2:
                lines.append(f"   Bias: {getattr(analysis.overnight, 'bias_emoji', '🟡')} {getattr(analysis.overnight, 'bias', 'NEUTRAL')}")
                if analysis.overnight.gap:
                    lines.append(f"   Gap: {getattr(analysis.overnight.gap, 'gap_pct', 0):+.2f}%")
            else:
                lines.append(f"   Bias: {analysis.overnight.bias.emoji} {analysis.overnight.bias.value}")
                if analysis.overnight.gap:
                    lines.append(f"   Gap: {analysis.overnight.gap.gap_type.emoji} {analysis.overnight.gap.gap_pct:+.2f}%")
                    lines.append(f"   Fill Prob: {analysis.overnight.gap.gap_fill_probability.value}")

        # Dual setup
        if analysis.dual_setup is not None:
            lines.append(f"\n⚔️ DUAL SETUP:")
            pref = getattr(analysis.dual_setup, 'preferred_direction', '?')
            reason = getattr(analysis.dual_setup, 'verdict_reason', '')
            lines.append(f"   Preferred: {pref}")
            if reason:
                lines.append(f"   Reason: {reason}")

        # Key levels
        lines.append(f"\n📍 KEY LEVELS:")
        sorted_levels = sorted(analysis.key_levels.items(), key=lambda x: x[1], reverse=True)
        for name, level in sorted_levels[:12]:
            lines.append(f"   {name:<18}: ${level:.2f}")

        # Trade plan
        if analysis.trade_plan:
            tp = analysis.trade_plan
            lines.append(f"\n{'=' * 80}")
            lines.append(f"📋 TRADE PLAN: {tp['direction']}")
            lines.append(f"{'=' * 80}")
            lines.append(f"   Bias: {tp['bias_strength']} ({tp['confidence']:.0f}% conf)")
            lines.append(f"   Entry Zone: {tp['entry_zone']}")
            lines.append(f"   Stop Loss:  ${tp['stop_loss']:.2f}")
            lines.append(f"   Target 1:   ${tp['target_1']:.2f}")
            lines.append(f"   Target 2:   ${tp['target_2']:.2f}")
            lines.append(f"   Risk/Share: ${tp['risk_per_share']:.2f}")
            lines.append(f"   R:R Ratio:  {tp['reward_risk_ratio']:.2f}")

            if 'dual_verdict' in tp:
                lines.append(f"   Dual Verdict: {tp['dual_verdict']}")
            if 'options_strategy' in tp:
                lines.append(f"   Options: {tp['options_strategy']}")

            lines.append(f"\n   Notes:")
            for note in tp['position_notes']:
                lines.append(f"     • {note}")
        else:
            lines.append(f"\n⏸️ NO TRADE PLAN — Wait for higher confidence setup")

        # Analysis notes
        lines.append(f"\n🔍 NOTES:")
        for note in analysis.notes:
            lines.append(f"   • {note}")

        lines.append("=" * 80)
        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    try:
        from overnight_model_v2 import generate_overnight_demo_data
    except ImportError:
        from overnight_model import generate_overnight_demo_data

    print("Generating demo data...")
    df = generate_overnight_demo_data()
    print(f"Data shape: {df.shape}")

    scanner = IntegratedScanner()
    analysis = scanner.analyze(df, symbol="DEMO")
    print(scanner.print_analysis(analysis))
