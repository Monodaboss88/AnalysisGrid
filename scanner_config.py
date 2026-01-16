"""
Scanner Configuration - Tunable Parameters
==========================================
Adjust these settings to match your trading style and risk tolerance.

Author: Rob's Trading Systems
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import List
from enum import Enum


class TradingStyle(Enum):
    """Pre-configured trading styles"""
    CONSERVATIVE = "conservative"  # Fewer signals, higher quality
    BALANCED = "balanced"          # Default settings
    AGGRESSIVE = "aggressive"      # More signals, accept more risk


@dataclass
class ScoringConfig:
    """
    Scoring thresholds for signal generation
    
    For Brokers:
    -----------
    These control how picky the scanner is about generating signals:
    - Higher thresholds = fewer but higher-confidence signals
    - Lower thresholds = more signals but some may be marginal
    
    For Programmers:
    ---------------
    Adjust these to tune sensitivity. The gap requirement prevents
    signals when bull and bear scores are too close.
    """
    
    # Signal thresholds
    strong_threshold: float = 65       # Score needed for confirmed setup
    moderate_threshold: float = 45     # Score needed for "leaning" signal
    min_score_gap: float = 20          # Minimum gap between bull/bear scores
    
    # Confluence requirements
    min_confluence_actionable: float = 50  # Minimum % of timeframes agreeing
    require_higher_tf_agreement: bool = True  # 4hr must confirm direction
    
    @classmethod
    def conservative(cls) -> 'ScoringConfig':
        """Conservative: High bar for signals"""
        return cls(
            strong_threshold=75,
            moderate_threshold=55,
            min_score_gap=30,
            min_confluence_actionable=75,
            require_higher_tf_agreement=True
        )
    
    @classmethod
    def balanced(cls) -> 'ScoringConfig':
        """Balanced: Default settings"""
        return cls()
    
    @classmethod
    def aggressive(cls) -> 'ScoringConfig':
        """Aggressive: More signals"""
        return cls(
            strong_threshold=55,
            moderate_threshold=40,
            min_score_gap=15,
            min_confluence_actionable=50,
            require_higher_tf_agreement=False
        )


@dataclass
class VolumeProfileConfig:
    """
    Volume Profile calculation settings
    
    For Brokers:
    -----------
    - value_area_pct: Standard is 70% (one standard deviation)
      - Use 80% for wider "fair value" range
      - Use 60% for tighter, more precise levels
    
    For Programmers:
    ---------------
    num_bins affects resolution of the price histogram.
    More bins = more precise POC but slower calculation.
    """
    
    value_area_pct: float = 0.70       # % of volume defining value area
    num_bins: int = 50                 # Price histogram resolution
    
    @classmethod
    def tight(cls) -> 'VolumeProfileConfig':
        """Tight value area for precise levels"""
        return cls(value_area_pct=0.60, num_bins=75)
    
    @classmethod
    def standard(cls) -> 'VolumeProfileConfig':
        """Standard 70% value area"""
        return cls()
    
    @classmethod
    def wide(cls) -> 'VolumeProfileConfig':
        """Wide value area for bigger picture"""
        return cls(value_area_pct=0.80, num_bins=40)


@dataclass
class RSIConfig:
    """
    RSI calculation and interpretation settings
    
    For Brokers:
    -----------
    - period: 14 is standard, lower = more sensitive
    - overbought/oversold: Traditional is 70/30
      - Use 80/20 to only catch extreme moves
      - Use 65/35 for earlier signals
    
    For Programmers:
    ---------------
    These define the zone boundaries used in scoring.
    """
    
    period: int = 14
    slope_period: int = 3
    overbought: float = 70
    oversold: float = 30
    bullish_zone: float = 60
    bearish_zone: float = 40
    
    @classmethod
    def standard(cls) -> 'RSIConfig':
        return cls()
    
    @classmethod
    def sensitive(cls) -> 'RSIConfig':
        """More sensitive RSI"""
        return cls(
            period=10,
            overbought=65,
            oversold=35,
            bullish_zone=55,
            bearish_zone=45
        )
    
    @classmethod
    def extreme_only(cls) -> 'RSIConfig':
        """Only flag extreme RSI"""
        return cls(
            period=14,
            overbought=80,
            oversold=20,
            bullish_zone=65,
            bearish_zone=35
        )


@dataclass
class FlowConfig:
    """
    Flow/Delta analysis settings
    
    For Brokers:
    -----------
    - imbalance_threshold: How much buyer/seller imbalance matters
    - momentum_period: How many bars to measure acceleration
    
    For Programmers:
    ---------------
    These tune the sensitivity of flow detection.
    """
    
    momentum_period: int = 5
    strong_imbalance: float = 0.30     # Flow imbalance for "strong" signal
    moderate_imbalance: float = 0.15   # Flow imbalance for "moderate"
    mild_imbalance: float = 0.05       # Flow imbalance for "mild"
    
    @classmethod
    def standard(cls) -> 'FlowConfig':
        return cls()
    
    @classmethod
    def sensitive(cls) -> 'FlowConfig':
        """More sensitive to flow changes"""
        return cls(
            momentum_period=3,
            strong_imbalance=0.20,
            moderate_imbalance=0.10,
            mild_imbalance=0.03
        )


@dataclass
class TimeframeConfig:
    """
    Timeframe selection and weighting
    
    For Brokers:
    -----------
    - enabled_timeframes: Which timeframes to analyze
    - weights: How much each timeframe matters in aggregate
      - Higher weight on 4hr = more emphasis on bigger picture
      - Equal weights = pure democracy across timeframes
    
    For Programmers:
    ---------------
    Weights are normalized in aggregation so they don't need to sum to 1.
    """
    
    enabled_timeframes: List[str] = None  # None = all
    weights: dict = None  # e.g., {"30min": 1.0, "4hour": 2.0}
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "30min": 1.0,
                "1hour": 1.5,
                "2hour": 1.5,
                "4hour": 2.0  # Higher timeframe = more weight
            }
    
    @classmethod
    def equal_weight(cls) -> 'TimeframeConfig':
        """Equal weight across all timeframes"""
        return cls(weights={
            "30min": 1.0, "1hour": 1.0, "2hour": 1.0, "4hour": 1.0
        })
    
    @classmethod
    def higher_tf_bias(cls) -> 'TimeframeConfig':
        """Heavily favor higher timeframes"""
        return cls(weights={
            "30min": 0.5, "1hour": 1.0, "2hour": 1.5, "4hour": 3.0
        })
    
    @classmethod
    def lower_tf_focus(cls) -> 'TimeframeConfig':
        """Focus on shorter timeframes (faster signals)"""
        return cls(
            enabled_timeframes=["30min", "1hour"],
            weights={"30min": 1.0, "1hour": 1.0}
        )


@dataclass 
class SwingTradeConfig:
    """
    Complete configuration for 3-5 day swing trading
    
    For Brokers:
    -----------
    This bundles all the settings optimized for your style:
    - 3-5 day holding period
    - Entry based on value area + flow confirmation
    - Explicit YELLOW states for uncertain periods
    
    For Programmers:
    ---------------
    Use this as the master config passed to the scanner.
    """
    
    scoring: ScoringConfig = None
    volume_profile: VolumeProfileConfig = None
    rsi: RSIConfig = None
    flow: FlowConfig = None
    timeframes: TimeframeConfig = None
    
    # Trade management
    default_stop_atr_mult: float = 1.5    # Stop = entry ± (ATR × this)
    target1_atr_mult: float = 2.0         # First target
    target2_atr_mult: float = 4.0         # Second target
    max_hold_days: int = 5                # Max days to hold
    
    def __post_init__(self):
        if self.scoring is None:
            self.scoring = ScoringConfig()
        if self.volume_profile is None:
            self.volume_profile = VolumeProfileConfig()
        if self.rsi is None:
            self.rsi = RSIConfig()
        if self.flow is None:
            self.flow = FlowConfig()
        if self.timeframes is None:
            self.timeframes = TimeframeConfig()
    
    @classmethod
    def conservative_swing(cls) -> 'SwingTradeConfig':
        """Conservative 3-5 day swing setup"""
        return cls(
            scoring=ScoringConfig.conservative(),
            volume_profile=VolumeProfileConfig.standard(),
            rsi=RSIConfig.extreme_only(),
            flow=FlowConfig.standard(),
            timeframes=TimeframeConfig.higher_tf_bias(),
            default_stop_atr_mult=2.0,
            target1_atr_mult=2.5,
            target2_atr_mult=5.0
        )
    
    @classmethod
    def balanced_swing(cls) -> 'SwingTradeConfig':
        """Balanced swing setup (default)"""
        return cls()
    
    @classmethod
    def active_swing(cls) -> 'SwingTradeConfig':
        """More active swing trading"""
        return cls(
            scoring=ScoringConfig.aggressive(),
            volume_profile=VolumeProfileConfig.tight(),
            rsi=RSIConfig.sensitive(),
            flow=FlowConfig.sensitive(),
            timeframes=TimeframeConfig.equal_weight(),
            default_stop_atr_mult=1.0,
            target1_atr_mult=1.5,
            target2_atr_mult=3.0
        )


# =============================================================================
# DISPLAY CONFIG
# =============================================================================

def print_config(config: SwingTradeConfig):
    """Pretty print current configuration"""
    lines = []
    lines.append("=" * 60)
    lines.append("CURRENT SCANNER CONFIGURATION")
    lines.append("=" * 60)
    
    lines.append("\n📊 SCORING THRESHOLDS:")
    lines.append(f"   Strong signal threshold:    {config.scoring.strong_threshold}")
    lines.append(f"   Moderate signal threshold:  {config.scoring.moderate_threshold}")
    lines.append(f"   Minimum bull/bear gap:      {config.scoring.min_score_gap}")
    lines.append(f"   Confluence for actionable:  {config.scoring.min_confluence_actionable}%")
    
    lines.append("\n📈 VOLUME PROFILE:")
    lines.append(f"   Value area percentage:      {config.volume_profile.value_area_pct:.0%}")
    lines.append(f"   Histogram bins:             {config.volume_profile.num_bins}")
    
    lines.append("\n📉 RSI:")
    lines.append(f"   Period:                     {config.rsi.period}")
    lines.append(f"   Overbought/Oversold:        {config.rsi.overbought}/{config.rsi.oversold}")
    lines.append(f"   Bullish/Bearish zones:      >{config.rsi.bullish_zone} / <{config.rsi.bearish_zone}")
    
    lines.append("\n🌊 FLOW:")
    lines.append(f"   Momentum period:            {config.flow.momentum_period}")
    lines.append(f"   Strong/Mod/Mild imbalance:  {config.flow.strong_imbalance}/{config.flow.moderate_imbalance}/{config.flow.mild_imbalance}")
    
    lines.append("\n⏱️ TIMEFRAMES:")
    for tf, weight in config.timeframes.weights.items():
        lines.append(f"   {tf:>8}: weight {weight}")
    
    lines.append("\n💼 TRADE MANAGEMENT:")
    lines.append(f"   Stop ATR multiplier:        {config.default_stop_atr_mult}x")
    lines.append(f"   Target 1 ATR multiplier:    {config.target1_atr_mult}x")
    lines.append(f"   Target 2 ATR multiplier:    {config.target2_atr_mult}x")
    lines.append(f"   Max hold days:              {config.max_hold_days}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# =============================================================================
# QUICK PRESETS
# =============================================================================

# Ready-to-use configurations
CONSERVATIVE = SwingTradeConfig.conservative_swing()
BALANCED = SwingTradeConfig.balanced_swing()
ACTIVE = SwingTradeConfig.active_swing()


if __name__ == "__main__":
    print("\n🔧 AVAILABLE CONFIGURATIONS:\n")
    
    print("1. CONSERVATIVE (fewer signals, higher quality)")
    print(print_config(CONSERVATIVE))
    
    print("\n2. BALANCED (default)")
    print(print_config(BALANCED))
    
    print("\n3. ACTIVE (more signals)")
    print(print_config(ACTIVE))
