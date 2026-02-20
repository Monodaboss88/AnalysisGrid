"""
Enhanced Squeeze Detector v2 — C.O.R.E. Methodology
=====================================================
Detects volatility compression setups aligned with Rob's auction theory
trading approach: compression reversals at key volume profile levels.

ORIGINAL FACTORS (retained):
- TTM Squeeze (BB inside Keltner) - Gold standard
- ATR Compression - Current ATR vs 20-period average
- ADX - Directional movement strength
- RSI - Now enhanced with exhaustion detection
- Range vs ATR - Now uses 3-day average instead of single day
- Squeeze Duration - How long compression has been active
- Relative Volume - Volume contraction

NEW IN V2:
- Volume Profile Context - Where is price relative to VAH/POC/VAL?
- Weekly Structure - Is weekly trend supporting the setup?
- RSI Exhaustion Signals - 2hr-equivalent < 32 (long) or > 68 (short)
- Squeeze Release Detection - First bar breaking Keltner after squeeze
- IV Percentile - Cheap options = better risk/reward (optional)
- Auction Theory Direction - Drift toward VAL/VAH, weekly close signals
- Multi-Day Range Averaging - 3-day range vs ATR (less noisy)
- VP Shape Context - Normal vs extreme distribution

Tiers:
- 50-64: FORMING - Squeeze is developing, monitor
- 65-79: ACTIVE - Squeeze is tight, prepare entries
- 80-89: PRIME - High probability setup, look for entry trigger
- 90+: TEXTBOOK - All factors aligned, act on exhaustion signal

Author: Rob's Trading Systems (Strategic Edge Flow)
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
from datetime import datetime


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VolumeProfileLevels:
    """Volume profile key levels"""
    poc: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    price_zone: str = "unknown"       # 'above_vah', 'vah_poc', 'at_poc', 'poc_val', 'below_val'
    proximity_to_val: float = 0.0     # % distance from VAL
    proximity_to_vah: float = 0.0     # % distance from VAH
    proximity_to_poc: float = 0.0     # % distance from POC
    vp_shape: str = "unknown"         # 'normal', 'wide', 'narrow', 'extreme'
    at_key_level: bool = False        # Price within 1% of VAH/POC/VAL


@dataclass
class WeeklyContext:
    """Weekly structure context for MTF awareness"""
    trend: str = "NEUTRAL"            # STRONG_UPTREND, UPTREND, NEUTRAL, DOWNTREND, STRONG_DOWNTREND
    last_week_structure: str = ""     # HH+HL, LH+LL, etc.
    weekly_close_position: float = 0.5  # 0=closed at low, 1=closed at high
    weekly_close_signal: str = ""     # BULLISH_REVERSAL, BEARISH_REVERSAL, etc.
    weekly_compression: float = 1.0   # Current week range / 8-week avg range
    supports_long: bool = False
    supports_short: bool = False


@dataclass
class SqueezeRelease:
    """Squeeze release/fire detection"""
    is_firing: bool = False           # Squeeze just released
    fire_direction: str = "none"      # 'long', 'short', 'none'
    bars_since_release: int = 0       # How many bars since squeeze ended
    momentum_histogram: float = 0.0   # TTM momentum histogram value


@dataclass 
class SqueezeMetrics:
    """Complete squeeze analysis result — V2"""
    symbol: str
    score: int
    tier: str                          # FORMING, ACTIVE, PRIME, TEXTBOOK
    quality_grade: str                 # A+, A, B, C, D
    factors: List[str]
    
    # === CORE SQUEEZE INDICATORS ===
    ttm_squeeze: bool
    ttm_score: int
    atr_compression: float             # Ratio of current ATR to average ATR
    atr_score: int
    adx: float
    adx_score: int
    rsi: float
    rsi_score: int
    rsi_zone: str                      # 'exhausted_long', 'reversal_long', 'neutral', 'reversal_short', 'exhausted_short'
    range_vs_atr: float                # 3-day avg range as % of ATR
    range_score: int
    rvol: float
    rvol_score: int
    squeeze_duration: int              # Days in squeeze
    duration_score: int
    
    # === NEW V2: VOLUME PROFILE CONTEXT ===
    volume_profile: VolumeProfileLevels = None
    vp_score: int = 0                  # 0-15 points
    
    # === NEW V2: WEEKLY STRUCTURE ===
    weekly: WeeklyContext = None
    weekly_score: int = 0              # 0-10 points
    
    # === NEW V2: SQUEEZE RELEASE ===
    release: SqueezeRelease = None
    
    # === NEW V2: IV CONTEXT ===
    iv_percentile: float = 0.0         # 0-100 (estimated from HV)
    iv_score: int = 0                  # 0-10 points
    
    # === DIRECTION BIAS (ENHANCED) ===
    direction_bias: str = "neutral"    # 'long', 'short', 'neutral'
    bias_score: int = 0                # Confidence 0-100
    bias_reasons: List[str] = field(default_factory=list)
    
    # === TRADE SETUP ===
    setup_type: str = ""               # 'compression_reversal_long', 'compression_reversal_short', 'squeeze_breakout', etc.
    entry_trigger: str = ""            # What to watch for
    
    # === PRICE LEVELS ===
    current_price: float = 0.0
    upper_band: float = 0.0
    lower_band: float = 0.0
    atr: float = 0.0
    avg_daily_range: float = 0.0
    
    timestamp: str = ""


# =============================================================================
# SQUEEZE DETECTOR V2
# =============================================================================

class SqueezeDetectorV2:
    """
    Enhanced squeeze detector aligned with C.O.R.E. methodology.
    Integrates volume profile, weekly structure, RSI exhaustion,
    and auction theory for high-probability compression setups.
    """
    
    def __init__(self, scanner=None):
        """
        Args:
            scanner: Optional FinnhubScanner instance for reusing its 
                     TechnicalCalculator methods. If None, we calculate locally.
        """
        self.scanner = scanner
        
        # TTM Squeeze parameters
        self.bb_period = 20
        self.bb_std = 2.0
        self.kc_period = 20
        self.kc_mult = 1.5
        self.atr_period = 14
        self.adx_period = 14
        
        # Volume Profile parameters
        self.vp_num_bins = 50
        self.vp_value_area_pct = 0.70
        
        # Key level proximity threshold (1% of price)
        self.proximity_threshold = 0.01
    
    # =========================================================================
    # CORE TECHNICAL CALCULATIONS
    # =========================================================================
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(window=self.bb_period).mean()
        std = df['close'].rolling(window=self.bb_period).std()
        upper = sma + (self.bb_std * std)
        lower = sma - (self.bb_std * std)
        return upper, sma, lower
    
    def calculate_keltner_channels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        ema = df['close'].ewm(span=self.kc_period, adjust=False).mean()
        atr = self.calculate_atr_series(df, self.kc_period)
        upper = ema + (self.kc_mult * atr)
        lower = ema - (self.kc_mult * atr)
        return upper, ema, lower
    
    def calculate_atr_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR as a series"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = df['high']
        low = df['low']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        plus_dm[(plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < plus_dm)] = 0
        
        atr = self.calculate_atr_series(df, period)
        
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI series using Wilder's smoothing"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # =========================================================================
    # NEW: VOLUME PROFILE ANALYSIS
    # =========================================================================
    
    def calculate_volume_profile(self, df: pd.DataFrame, 
                                  current_price: float) -> VolumeProfileLevels:
        """
        Calculate POC, VAH, VAL and determine where price sits relative
        to the value area. This is the core of the C.O.R.E. methodology —
        squeezes at key VP levels are the highest probability setups.
        """
        if len(df) < 10:
            return VolumeProfileLevels()
        
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        if price_max == price_min:
            return VolumeProfileLevels(poc=price_max, vah=price_max, val=price_min)
        
        bin_size = (price_max - price_min) / self.vp_num_bins
        bins = np.arange(price_min, price_max + bin_size, bin_size)
        
        volume_profile = np.zeros(len(bins) - 1)
        
        for _, row in df.iterrows():
            bar_low = row['low']
            bar_high = row['high']
            bar_volume = row['volume']
            
            for i in range(len(bins) - 1):
                bin_low = bins[i]
                bin_high = bins[i + 1]
                
                overlap_low = max(bar_low, bin_low)
                overlap_high = min(bar_high, bin_high)
                
                if overlap_high > overlap_low:
                    bar_range = bar_high - bar_low
                    if bar_range > 0:
                        overlap_pct = (overlap_high - overlap_low) / bar_range
                        volume_profile[i] += bar_volume * overlap_pct
        
        # POC = highest volume bin
        poc_idx = np.argmax(volume_profile)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Value Area (70% of volume)
        total_volume = volume_profile.sum()
        target_volume = total_volume * self.vp_value_area_pct
        
        va_volume = volume_profile[poc_idx]
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        
        while va_volume < target_volume:
            expand_low = va_low_idx > 0
            expand_high = va_high_idx < len(volume_profile) - 1
            
            if not expand_low and not expand_high:
                break
            
            low_vol = volume_profile[va_low_idx - 1] if expand_low else 0
            high_vol = volume_profile[va_high_idx + 1] if expand_high else 0
            
            if low_vol >= high_vol and expand_low:
                va_low_idx -= 1
                va_volume += low_vol
            elif expand_high:
                va_high_idx += 1
                va_volume += high_vol
            elif expand_low:
                va_low_idx -= 1
                va_volume += low_vol
        
        val = bins[va_low_idx]
        vah = bins[va_high_idx + 1]
        
        # Determine price zone
        if current_price > vah * 1.005:
            price_zone = "above_vah"
        elif current_price >= poc * 1.005:
            price_zone = "vah_poc"
        elif abs(current_price - poc) / poc < 0.005:
            price_zone = "at_poc"
        elif current_price >= val * 0.995:
            price_zone = "poc_val"
        else:
            price_zone = "below_val"
        
        # Proximity calculations
        prox_val = abs(current_price - val) / current_price if current_price > 0 else 1.0
        prox_vah = abs(current_price - vah) / current_price if current_price > 0 else 1.0
        prox_poc = abs(current_price - poc) / current_price if current_price > 0 else 1.0
        
        at_key_level = (prox_val < self.proximity_threshold or 
                        prox_vah < self.proximity_threshold or 
                        prox_poc < self.proximity_threshold)
        
        # VP Shape — narrow value area = more compressed = better
        value_area_width = (vah - val) / current_price if current_price > 0 else 0.1
        if value_area_width < 0.03:
            vp_shape = "extreme"      # Very narrow — extreme compression
        elif value_area_width < 0.06:
            vp_shape = "narrow"       # Tight value area
        elif value_area_width < 0.12:
            vp_shape = "normal"       # Standard distribution
        else:
            vp_shape = "wide"         # Wide value area — less useful
        
        return VolumeProfileLevels(
            poc=round(poc, 2),
            vah=round(vah, 2),
            val=round(val, 2),
            price_zone=price_zone,
            proximity_to_val=round(prox_val * 100, 2),
            proximity_to_vah=round(prox_vah * 100, 2),
            proximity_to_poc=round(prox_poc * 100, 2),
            vp_shape=vp_shape,
            at_key_level=at_key_level
        )
    
    # =========================================================================
    # NEW: WEEKLY STRUCTURE ANALYSIS
    # =========================================================================
    
    def calculate_weekly_context(self, symbol: str) -> WeeklyContext:
        """
        Fetch weekly data and calculate structure.
        Weekly candles = complete auction cycles, so structure is high-conviction.
        """
        try:
            from polygon_data import get_bars
            df_w = get_bars(symbol, period="6mo", interval="1wk")
            
            if df_w.empty or len(df_w) < 6:
                return WeeklyContext()
            
            df_w.columns = [c.lower() for c in df_w.columns]
            
            weeks = df_w.tail(8)
            
            # Count structure patterns
            ll_count = hh_count = lh_count = hl_count = 0
            last_structure = ""
            
            for i in range(1, len(weeks)):
                curr = weeks.iloc[i]
                prev = weeks.iloc[i - 1]
                
                structure = ""
                if curr['high'] > prev['high'] * 1.001:
                    structure += "HH"
                    hh_count += 1
                elif curr['high'] < prev['high'] * 0.999:
                    structure += "LH"
                    lh_count += 1
                else:
                    structure += "EQ"
                
                if curr['low'] > prev['low'] * 1.001:
                    structure += "+HL"
                    hl_count += 1
                elif curr['low'] < prev['low'] * 0.999:
                    structure += "+LL"
                    ll_count += 1
                else:
                    structure += "+EQ"
                
                last_structure = structure
            
            # Trend classification
            bearish_signals = ll_count + lh_count
            bullish_signals = hh_count + hl_count
            
            if bearish_signals >= 8 and bullish_signals <= 2:
                trend = "STRONG_DOWNTREND"
            elif bearish_signals >= 5 and bearish_signals > bullish_signals * 2:
                trend = "DOWNTREND"
            elif bullish_signals >= 8 and bearish_signals <= 2:
                trend = "STRONG_UPTREND"
            elif bullish_signals >= 5 and bullish_signals > bearish_signals * 2:
                trend = "UPTREND"
            else:
                trend = "NEUTRAL"
            
            # Weekly close position (last completed week)
            last_week = weeks.iloc[-2]  # Use second-to-last (completed week)
            lw_range = last_week['high'] - last_week['low']
            weekly_close_pos = (last_week['close'] - last_week['low']) / lw_range if lw_range > 0 else 0.5
            
            # Weekly close signal
            weekly_close_signal = ""
            if "LL" in last_structure and weekly_close_pos > 0.70:
                weekly_close_signal = "BULLISH_REVERSAL"
            elif "HH" in last_structure and weekly_close_pos < 0.30:
                weekly_close_signal = "BEARISH_REVERSAL"
            elif weekly_close_pos > 0.75:
                weekly_close_signal = "STRONG_BULL_CLOSE"
            elif weekly_close_pos < 0.25:
                weekly_close_signal = "STRONG_BEAR_CLOSE"
            
            # Weekly compression: current week range vs 8-week average
            current_week_range = weeks.iloc[-1]['high'] - weeks.iloc[-1]['low']
            avg_week_range = (weeks['high'] - weeks['low']).mean()
            weekly_compression = current_week_range / avg_week_range if avg_week_range > 0 else 1.0
            
            # Determine directional support
            supports_long = (
                trend in ("UPTREND", "STRONG_UPTREND") or
                weekly_close_signal in ("BULLISH_REVERSAL", "STRONG_BULL_CLOSE") or
                (trend == "NEUTRAL" and weekly_close_pos > 0.6)
            )
            supports_short = (
                trend in ("DOWNTREND", "STRONG_DOWNTREND") or
                weekly_close_signal in ("BEARISH_REVERSAL", "STRONG_BEAR_CLOSE") or
                (trend == "NEUTRAL" and weekly_close_pos < 0.4)
            )
            
            return WeeklyContext(
                trend=trend,
                last_week_structure=last_structure,
                weekly_close_position=round(weekly_close_pos, 2),
                weekly_close_signal=weekly_close_signal,
                weekly_compression=round(weekly_compression, 2),
                supports_long=supports_long,
                supports_short=supports_short
            )
            
        except Exception as e:
            print(f"Weekly context error for {symbol}: {e}")
            return WeeklyContext()
    
    # =========================================================================
    # NEW: SQUEEZE RELEASE / FIRE DETECTION
    # =========================================================================
    
    def detect_squeeze_release(self, df: pd.DataFrame) -> SqueezeRelease:
        """
        Detect if a squeeze has just fired (BB expanding outside KC).
        The first candle breaking outside Keltner after a multi-day squeeze
        is often the entry signal.
        """
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger_bands(df)
        kc_upper, kc_mid, kc_lower = self.calculate_keltner_channels(df)
        
        # Build squeeze state series
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        
        if len(squeeze) < 5:
            return SqueezeRelease()
        
        # Check if squeeze just released (was in squeeze recently, now isn't)
        is_firing = False
        fire_direction = "none"
        bars_since_release = 0
        
        # Look back up to 5 bars for recent squeeze release
        for i in range(len(squeeze) - 1, max(len(squeeze) - 6, 0), -1):
            if squeeze.iloc[i]:
                # Found the squeeze — everything after this is the release
                bars_since_release = len(squeeze) - 1 - i
                if bars_since_release > 0 and bars_since_release <= 3:
                    is_firing = True
                    # Direction based on price movement since release
                    release_close = df['close'].iloc[i]
                    current_close = df['close'].iloc[-1]
                    if current_close > release_close:
                        fire_direction = "long"
                    elif current_close < release_close:
                        fire_direction = "short"
                break
        
        # TTM Momentum histogram (simplified: momentum of midline)
        momentum = bb_mid - kc_mid
        mom_hist = float(momentum.iloc[-1]) if len(momentum) > 0 else 0.0
        
        return SqueezeRelease(
            is_firing=is_firing,
            fire_direction=fire_direction,
            bars_since_release=bars_since_release,
            momentum_histogram=round(mom_hist, 4)
        )
    
    # =========================================================================
    # NEW: IV PERCENTILE (Historical Volatility Based)
    # =========================================================================
    
    def estimate_iv_percentile(self, df: pd.DataFrame) -> float:
        """
        Estimate IV percentile using historical volatility rank.
        True IV requires options chain data (available via Tradier in the server),
        but for scanning purposes we use HV percentile as a proxy.
        Low HV percentile during squeeze = cheap options = better R:R.
        """
        if len(df) < 60:
            return 50.0
        
        # Calculate 20-day rolling HV (annualized)
        log_returns = np.log(df['close'] / df['close'].shift(1))
        hv_series = log_returns.rolling(window=20).std() * np.sqrt(252) * 100
        
        hv_series = hv_series.dropna()
        if len(hv_series) < 20:
            return 50.0
        
        current_hv = hv_series.iloc[-1]
        
        # Percentile rank of current HV over last 60 bars
        lookback = hv_series.tail(60)
        percentile = (lookback < current_hv).sum() / len(lookback) * 100
        
        return round(percentile, 1)
    
    # =========================================================================
    # NEW: ENHANCED DIRECTION BIAS (AUCTION THEORY)
    # =========================================================================
    
    def calculate_direction_bias_v2(self, df: pd.DataFrame, 
                                     vp: VolumeProfileLevels,
                                     weekly: WeeklyContext) -> Dict:
        """
        Enhanced direction bias using auction theory:
        - Price drift toward VAL = bullish reversal potential
        - Price drift toward VAH = bearish reversal potential  
        - Weekly structure alignment
        - Volume on up vs down moves
        - Recent candle momentum
        """
        if len(df) < 10:
            return {'bias': 'neutral', 'score': 0, 'reasons': []}
        
        recent = df.tail(10)
        bias_score = 0
        reasons = []
        
        # === 1. VOLUME PROFILE ZONE (+/- 25 points) ===
        # Squeeze near VAL = bullish bias (reversal zone)
        # Squeeze near VAH = bearish bias (rejection zone)
        if vp.price_zone == "below_val":
            bias_score += 25
            reasons.append(f"Price below VAL ${vp.val} (reversal zone)")
        elif vp.price_zone == "poc_val" and vp.proximity_to_val < 2.0:
            bias_score += 15
            reasons.append(f"Price near VAL ${vp.val}")
        elif vp.price_zone == "above_vah":
            bias_score -= 25
            reasons.append(f"Price above VAH ${vp.vah} (rejection zone)")
        elif vp.price_zone == "vah_poc" and vp.proximity_to_vah < 2.0:
            bias_score -= 15
            reasons.append(f"Price near VAH ${vp.vah}")
        
        # === 2. WEEKLY STRUCTURE (+/- 20 points) ===
        if weekly.supports_long:
            bias_score += 20
            reasons.append(f"Weekly supports long ({weekly.trend})")
        elif weekly.supports_short:
            bias_score -= 20
            reasons.append(f"Weekly supports short ({weekly.trend})")
        
        if weekly.weekly_close_signal == "BULLISH_REVERSAL":
            bias_score += 15
            reasons.append("Weekly bullish reversal signal")
        elif weekly.weekly_close_signal == "BEARISH_REVERSAL":
            bias_score -= 15
            reasons.append("Weekly bearish reversal signal")
        
        # === 3. VOLUME ON UP VS DOWN MOVES (+/- 20 points) ===
        up_volume = 0
        down_volume = 0
        for i in range(1, len(recent)):
            if recent['close'].iloc[i] > recent['close'].iloc[i-1]:
                up_volume += recent['volume'].iloc[i]
            else:
                down_volume += recent['volume'].iloc[i]
        
        total_volume = up_volume + down_volume
        if total_volume > 0:
            up_pct = up_volume / total_volume
            if up_pct > 0.65:
                bias_score += 20
                reasons.append("Accumulation (volume favors up)")
            elif up_pct < 0.35:
                bias_score -= 20
                reasons.append("Distribution (volume favors down)")
        
        # === 4. PRICE DRIFT (+/- 15 points) ===
        first_half = recent['close'].iloc[:5].mean()
        second_half = recent['close'].iloc[5:].mean()
        drift_pct = (second_half - first_half) / first_half * 100
        
        if drift_pct > 0.5:
            bias_score += min(15, int(abs(drift_pct) * 8))
            reasons.append(f"Price drifting up ({drift_pct:.1f}%)")
        elif drift_pct < -0.5:
            bias_score -= min(15, int(abs(drift_pct) * 8))
            reasons.append(f"Price drifting down ({drift_pct:.1f}%)")
        
        # === 5. RECENT CANDLE MOMENTUM (+/- 20 points) ===
        last_3_bullish = sum(1 for i in range(-3, 0) 
                            if recent['close'].iloc[i] > recent['open'].iloc[i])
        if last_3_bullish == 3:
            bias_score += 20
            reasons.append("3 consecutive bullish candles")
        elif last_3_bullish == 0:
            bias_score -= 20
            reasons.append("3 consecutive bearish candles")
        elif last_3_bullish >= 2:
            bias_score += 10
        elif last_3_bullish <= 1:
            bias_score -= 10
        
        # Determine direction
        if bias_score > 20:
            direction = 'long'
        elif bias_score < -20:
            direction = 'short'
        else:
            direction = 'neutral'
        
        return {
            'bias': direction,
            'score': min(100, max(0, abs(bias_score))),
            'reasons': reasons
        }
    
    # =========================================================================
    # TTM SQUEEZE DETECTION
    # =========================================================================
    
    def detect_ttm_squeeze(self, df: pd.DataFrame) -> Tuple[bool, int]:
        """TTM Squeeze: BB inside Keltner Channels"""
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger_bands(df)
        kc_upper, kc_mid, kc_lower = self.calculate_keltner_channels(df)
        
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        
        squeeze_days = 0
        for i in range(len(squeeze) - 1, -1, -1):
            if squeeze.iloc[i]:
                squeeze_days += 1
            else:
                break
        
        is_squeeze_now = bool(squeeze.iloc[-1]) if len(squeeze) > 0 else False
        return is_squeeze_now, squeeze_days
    
    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================
    
    def analyze(self, symbol: str) -> Optional[SqueezeMetrics]:
        """
        Full squeeze analysis with C.O.R.E. methodology integration.
        
        Scoring (max 120 points, normalized to 100):
        - TTM Squeeze:        0-25 pts (gold standard)
        - ATR Compression:    0-20 pts
        - ADX (no trend):     0-15 pts
        - RSI Zone:           0-15 pts (enhanced: rewards exhaustion)
        - Range vs ATR:       0-15 pts (now 3-day average)
        - Low Volume:         0-10 pts
        - Squeeze Duration:   0-15 pts
        - Volume Profile:     0-15 pts (NEW)
        - Weekly Alignment:   0-10 pts (NEW)
        - IV Percentile:      0-10 pts (NEW)
        """
        try:
            # =================================================================
            # FETCH DATA
            # =================================================================
            from polygon_data import get_bars
            df = get_bars(symbol, period="6mo", interval="1d")
            
            if df.empty or len(df) < 30:
                return None
            
            df.columns = [c.lower() for c in df.columns]
            
            current_price = float(df['close'].iloc[-1])
            
            # =================================================================
            # CORE INDICATORS
            # =================================================================
            bb_upper, bb_mid, bb_lower = self.calculate_bollinger_bands(df)
            kc_upper, kc_mid, kc_lower = self.calculate_keltner_channels(df)
            atr_series = self.calculate_atr_series(df, self.atr_period)
            adx_series = self.calculate_adx(df, self.adx_period)
            rsi_series = self.calculate_rsi(df)
            
            current_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0
            avg_atr_20 = float(atr_series.tail(20).mean()) if len(atr_series) >= 20 else current_atr
            current_adx = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 25
            current_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
            
            # 3-day average range (V2 improvement — less noisy than single day)
            recent_ranges = (df['high'].tail(3) - df['low'].tail(3))
            avg_3d_range = float(recent_ranges.mean())
            range_vs_atr = avg_3d_range / current_atr if current_atr > 0 else 1.0
            
            # Average daily range (5-day)
            avg_daily_range = float((df['high'].tail(5) - df['low'].tail(5)).mean())
            
            # Relative volume
            avg_volume = float(df['volume'].tail(20).mean())
            current_volume = float(df['volume'].iloc[-1])
            rvol = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # TTM Squeeze
            is_ttm_squeeze, squeeze_days = self.detect_ttm_squeeze(df)
            
            # =================================================================
            # NEW V2: VOLUME PROFILE
            # =================================================================
            # Use last 30 days for tactical VP (aligns with swing timeframe)
            df_vp = df.tail(30)
            vp = self.calculate_volume_profile(df_vp, current_price)
            
            # =================================================================
            # NEW V2: WEEKLY STRUCTURE
            # =================================================================
            weekly = self.calculate_weekly_context(symbol)
            
            # =================================================================
            # NEW V2: SQUEEZE RELEASE
            # =================================================================
            release = self.detect_squeeze_release(df)
            
            # =================================================================
            # NEW V2: IV PERCENTILE
            # =================================================================
            iv_pct = self.estimate_iv_percentile(df)
            
            # =================================================================
            # NEW V2: ENHANCED DIRECTION BIAS
            # =================================================================
            bias_data = self.calculate_direction_bias_v2(df, vp, weekly)
            
            # =================================================================
            # RSI ZONE CLASSIFICATION
            # =================================================================
            if current_rsi < 32:
                rsi_zone = "exhausted_long"      # Prime long entry zone
            elif current_rsi < 40:
                rsi_zone = "reversal_long"        # Approaching exhaustion
            elif current_rsi <= 60:
                rsi_zone = "neutral"
            elif current_rsi < 68:
                rsi_zone = "reversal_short"       # Approaching short exhaustion
            else:
                rsi_zone = "exhausted_short"      # Prime short entry zone
            
            # =================================================================
            # SCORING
            # =================================================================
            factors = []
            
            # --- 1. TTM Squeeze (0-25 pts) - GRADUATED by duration ---
            ttm_score = 0
            if is_ttm_squeeze:
                if squeeze_days >= 6:
                    ttm_score = 25
                    factors.append(f"TTM Squeeze ({squeeze_days}d extended)")
                elif squeeze_days >= 4:
                    ttm_score = 20
                    factors.append(f"TTM Squeeze ({squeeze_days}d)")
                elif squeeze_days >= 2:
                    ttm_score = 15
                    factors.append(f"TTM Squeeze ({squeeze_days}d forming)")
            
            # --- 2. ATR Compression (0-20 pts) ---
            atr_compression = current_atr / avg_atr_20 if avg_atr_20 > 0 else 1.0
            atr_score = 0
            if atr_compression < 0.6:
                atr_score = 20
                factors.append("ATR heavily compressed")
            elif atr_compression < 0.7:
                atr_score = 15
                factors.append("ATR compressed")
            elif atr_compression < 0.8:
                atr_score = 10
                factors.append("Low ATR")
            
            # --- 3. ADX (0-15 pts) ---
            adx_score = 0
            if current_adx < 15:
                adx_score = 15
                factors.append("No trend (ADX)")
            elif current_adx < 20:
                adx_score = 10
                factors.append("Weak trend")
            elif current_adx < 25:
                adx_score = 5
            
            # --- 4. RSI (0-15 pts) — ENHANCED: rewards exhaustion ---
            rsi_score = 0
            if rsi_zone == "exhausted_long":
                rsi_score = 15
                factors.append(f"RSI exhaustion LONG ({current_rsi:.0f})")
            elif rsi_zone == "exhausted_short":
                rsi_score = 15
                factors.append(f"RSI exhaustion SHORT ({current_rsi:.0f})")
            elif rsi_zone == "reversal_long":
                rsi_score = 12
                factors.append(f"RSI reversal zone ({current_rsi:.0f})")
            elif rsi_zone == "reversal_short":
                rsi_score = 12
                factors.append(f"RSI reversal zone ({current_rsi:.0f})")
            elif 45 <= current_rsi <= 55:
                rsi_score = 8
                factors.append("RSI coiling")
            elif 40 <= current_rsi <= 60:
                rsi_score = 5
            
            # --- 5. Range vs ATR (0-15 pts) — now uses 3-day average ---
            range_score = 0
            if range_vs_atr < 0.5:
                range_score = 15
                factors.append("Very tight range")
            elif range_vs_atr < 0.7:
                range_score = 10
                factors.append("Tight range")
            elif range_vs_atr < 0.9:
                range_score = 5
                factors.append("Narrow range")
            
            # --- 6. Low Volume (0-10 pts) ---
            rvol_score = 0
            if rvol < 0.5:
                rvol_score = 10
                factors.append("Very low volume")
            elif rvol < 0.7:
                rvol_score = 7
                factors.append("Low volume")
            elif rvol < 0.9:
                rvol_score = 4
            
            # --- 7. Squeeze Duration (0-15 pts) ---
            duration_score = 0
            if squeeze_days >= 6:
                duration_score = 15
                factors.append(f"{squeeze_days}d squeeze (extended)")
            elif squeeze_days >= 4:
                duration_score = 10
                factors.append(f"{squeeze_days}d squeeze")
            elif squeeze_days >= 2:
                duration_score = 5
            
            # --- 8. NEW: Volume Profile Context (0-18 pts) - BOOSTED weight ---
            vp_score = 0
            
            # Squeeze at key VP level = highest probability
            if vp.at_key_level:
                vp_score += 10
                factors.append(f"At key VP level ({vp.price_zone})")
            
            # VP shape bonus — narrow/extreme = more compressed auction
            if vp.vp_shape == "extreme":
                vp_score += 8
                factors.append("Extreme VP compression")
            elif vp.vp_shape == "narrow":
                vp_score += 6
                factors.append("Narrow value area")
            elif vp.vp_shape == "normal":
                vp_score += 2
            
            vp_score = min(18, vp_score)
            
            # --- 9. NEW: Weekly Alignment (0-10 pts) ---
            weekly_score = 0
            
            # Weekly structure confirms direction bias
            direction = bias_data['bias']
            if direction == 'long' and weekly.supports_long:
                weekly_score = 10
                factors.append(f"Weekly confirms long ({weekly.trend})")
            elif direction == 'short' and weekly.supports_short:
                weekly_score = 10
                factors.append(f"Weekly confirms short ({weekly.trend})")
            elif weekly.weekly_compression < 0.6:
                weekly_score = 5
                factors.append("Weekly also compressing")
            
            # --- 10. NEW: IV Percentile (0-10 pts) ---
            iv_score = 0
            if iv_pct < 20:
                iv_score = 10
                factors.append(f"Low IV ({iv_pct:.0f}%ile) — cheap options")
            elif iv_pct < 35:
                iv_score = 7
                factors.append(f"Below-avg IV ({iv_pct:.0f}%ile)")
            elif iv_pct < 50:
                iv_score = 3
            
            # =================================================================
            # TOTAL SCORE & TIER
            # =================================================================
            raw_total = (ttm_score + atr_score + adx_score + rsi_score + 
                        range_score + rvol_score + duration_score +
                        vp_score + weekly_score + iv_score)
            
            # Normalize: max possible is 153 (VP boosted 15→18), scale to 100
            total_score = min(100, int(raw_total * 100 / 153))
            
            # Tier classification (updated for v2 - adjusted thresholds)
            if total_score >= 90:
                tier = "TEXTBOOK"
            elif total_score >= 80:
                tier = "PRIME"
            elif total_score >= 60:
                tier = "ACTIVE"
            elif total_score >= 45:
                tier = "FORMING"
            else:
                tier = "NONE"
            
            # Quality grade
            if total_score >= 90:
                quality_grade = "A+"
            elif total_score >= 80:
                quality_grade = "A"
            elif total_score >= 70:
                quality_grade = "B"
            elif total_score >= 60:
                quality_grade = "C"
            elif total_score >= 50:
                quality_grade = "D"
            else:
                quality_grade = "F"
            
            # =================================================================
            # SETUP TYPE & ENTRY TRIGGER
            # =================================================================
            setup_type = ""
            entry_trigger = ""
            
            if is_ttm_squeeze and vp.at_key_level:
                if direction == 'long' and rsi_zone in ("exhausted_long", "reversal_long"):
                    setup_type = "compression_reversal_long"
                    entry_trigger = f"Wait for RSI < 32 at VAL ${vp.val}, enter 50% size"
                elif direction == 'short' and rsi_zone in ("exhausted_short", "reversal_short"):
                    setup_type = "compression_reversal_short"
                    entry_trigger = f"Wait for RSI > 68 at VAH ${vp.vah}, enter 50% size"
                else:
                    setup_type = "squeeze_at_key_level"
                    entry_trigger = f"Watch for RSI exhaustion at {vp.price_zone}"
            elif is_ttm_squeeze:
                setup_type = "squeeze_breakout"
                if direction != 'neutral':
                    entry_trigger = f"Watch for squeeze fire {direction}, enter on Keltner break"
                else:
                    entry_trigger = "Monitor for directional break of Keltner channel"
            elif release.is_firing:
                setup_type = f"squeeze_fire_{release.fire_direction}"
                entry_trigger = f"Squeeze fired {release.fire_direction} — confirm with volume"
            
            return SqueezeMetrics(
                symbol=symbol.upper(),
                score=total_score,
                tier=tier,
                quality_grade=quality_grade,
                factors=factors[:6],
                
                ttm_squeeze=is_ttm_squeeze,
                ttm_score=ttm_score,
                atr_compression=round(atr_compression, 2),
                atr_score=atr_score,
                adx=round(current_adx, 1),
                adx_score=adx_score,
                rsi=round(current_rsi, 1),
                rsi_score=rsi_score,
                rsi_zone=rsi_zone,
                range_vs_atr=round(range_vs_atr, 2),
                range_score=range_score,
                rvol=round(rvol, 2),
                rvol_score=rvol_score,
                squeeze_duration=squeeze_days,
                duration_score=duration_score,
                
                volume_profile=vp,
                vp_score=vp_score,
                
                weekly=weekly,
                weekly_score=weekly_score,
                
                release=release,
                
                iv_percentile=iv_pct,
                iv_score=iv_score,
                
                direction_bias=bias_data['bias'],
                bias_score=bias_data['score'],
                bias_reasons=bias_data['reasons'],
                
                setup_type=setup_type,
                entry_trigger=entry_trigger,
                
                current_price=round(current_price, 2),
                upper_band=round(float(kc_upper.iloc[-1]), 2),
                lower_band=round(float(kc_lower.iloc[-1]), 2),
                atr=round(current_atr, 2),
                avg_daily_range=round(avg_daily_range, 2),
                
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"Squeeze V2 analysis error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def to_dict(self, metrics: SqueezeMetrics) -> Dict:
        """Convert SqueezeMetrics to serializable dict for API responses"""
        d = asdict(metrics)
        return d


# =============================================================================
# BATCH SCANNER
# =============================================================================

def scan_for_squeezes_v2(symbols: List[str], 
                          min_tier: str = "FORMING") -> List[SqueezeMetrics]:
    """
    Scan multiple symbols for squeeze setups.
    Returns list sorted by score (highest first).
    
    Args:
        symbols: List of ticker symbols
        min_tier: Minimum tier to include ('FORMING', 'ACTIVE', 'PRIME', 'TEXTBOOK')
    """
    tier_order = {"NONE": 0, "FORMING": 1, "ACTIVE": 2, "PRIME": 3, "TEXTBOOK": 4}
    min_tier_val = tier_order.get(min_tier, 1)
    
    detector = SqueezeDetectorV2()
    results = []
    
    for symbol in symbols:
        try:
            metrics = detector.analyze(symbol)
            if metrics and tier_order.get(metrics.tier, 0) >= min_tier_val:
                results.append(metrics)
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
    
    results.sort(key=lambda x: x.score, reverse=True)
    return results


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    test_symbols = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'META', 'TSLA', 'AMZN']
    
    print("=" * 70)
    print("  SQUEEZE DETECTOR V2 — C.O.R.E. Methodology")
    print("=" * 70)
    
    detector = SqueezeDetectorV2()
    
    for symbol in test_symbols:
        result = detector.analyze(symbol)
        if result:
            print(f"\n{'='*60}")
            print(f"  {symbol}: Score {result.score}/100 — {result.tier} ({result.quality_grade})")
            print(f"{'='*60}")
            
            print(f"  Setup: {result.setup_type or 'No active setup'}")
            if result.entry_trigger:
                print(f"  Entry: {result.entry_trigger}")
            
            print(f"\n  --- Core Squeeze ---")
            print(f"  TTM Squeeze: {'YES' if result.ttm_squeeze else 'NO'} ({result.squeeze_duration}d)")
            print(f"  ATR Compression: {result.atr_compression}x (score: {result.atr_score})")
            print(f"  ADX: {result.adx} (score: {result.adx_score})")
            print(f"  RSI: {result.rsi} [{result.rsi_zone}] (score: {result.rsi_score})")
            print(f"  Range/ATR: {result.range_vs_atr} (score: {result.range_score})")
            print(f"  RVol: {result.rvol} (score: {result.rvol_score})")
            
            print(f"\n  --- Volume Profile ---")
            if result.volume_profile:
                vp = result.volume_profile
                print(f"  VAH: ${vp.vah}  |  POC: ${vp.poc}  |  VAL: ${vp.val}")
                print(f"  Zone: {vp.price_zone} | Shape: {vp.vp_shape} | At Key Level: {vp.at_key_level}")
                print(f"  VP Score: {result.vp_score}")
            
            print(f"\n  --- Weekly Structure ---")
            if result.weekly:
                w = result.weekly
                print(f"  Trend: {w.trend} | Last Week: {w.last_week_structure}")
                print(f"  Close Position: {w.weekly_close_position} | Signal: {w.weekly_close_signal or 'none'}")
                print(f"  Weekly Compression: {w.weekly_compression}x")
                print(f"  Weekly Score: {result.weekly_score}")
            
            print(f"\n  --- Release / IV ---")
            if result.release and result.release.is_firing:
                print(f"  SQUEEZE FIRING {result.release.fire_direction.upper()}! ({result.release.bars_since_release} bars)")
            else:
                print(f"  Release: Not firing")
            print(f"  IV Percentile: {result.iv_percentile}% (score: {result.iv_score})")
            
            print(f"\n  --- Direction ---")
            print(f"  Bias: {result.direction_bias} ({result.bias_score}% confidence)")
            for reason in result.bias_reasons:
                print(f"    • {reason}")
            
            print(f"\n  Factors: {', '.join(result.factors)}")
        else:
            print(f"\n{symbol}: No data or error")
    
    print(f"\n{'='*70}")
    print("  BATCH SCAN — ACTIVE+ Only")
    print(f"{'='*70}")
    
    actives = scan_for_squeezes_v2(test_symbols, min_tier="ACTIVE")
    if actives:
        for r in actives:
            print(f"  {r.symbol}: {r.score} ({r.tier}) — {r.setup_type or 'monitoring'}")
    else:
        print("  No ACTIVE+ squeezes found in test symbols")
