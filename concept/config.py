"""
SEF Trading System - Configuration
Multi-Timeframe Volume Profile + VWAP System
"""

from dataclasses import dataclass, field
from typing import List, Dict
from datetime import time
import os

@dataclass
class APIConfig:
    """API credentials and endpoints"""
    # Polygon.io
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")
    polygon_base_url: str = "https://api.polygon.io"
    
    # Tradier
    tradier_api_key: str = os.getenv("TRADIER_API_KEY", "")
    tradier_account_id: str = os.getenv("TRADIER_ACCOUNT_ID", "")
    tradier_base_url: str = "https://api.tradier.com/v1"  # Production
    tradier_sandbox_url: str = "https://sandbox.tradier.com/v1"  # Sandbox
    use_sandbox: bool = True  # Start with sandbox


@dataclass
class SessionConfig:
    """Trading session parameters"""
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    timezone: str = "America/New_York"
    use_rth_only: bool = True  # Regular Trading Hours only for VP


@dataclass
class VolumeProfileConfig:
    """Volume Profile calculation parameters"""
    num_bins: int = 50  # Price bins for VP distribution
    resolution_minutes: int = 5  # Bar size for VP calculation
    value_area_pct: float = 0.70  # 70% of volume for VAH/VAL
    poc_smoothing: bool = True  # Smooth POC to avoid spikes
    poc_smoothing_window: int = 3  # Bins to smooth over


@dataclass
class VWAPConfig:
    """VWAP calculation parameters"""
    anchors: List[str] = field(default_factory=lambda: [
        "monthly",  # Anchored to month start
        "weekly",   # Anchored to week start
        "daily",    # Anchored to day start
        "rolling"   # Rolling N-period
    ])
    rolling_periods: int = 20  # For rolling VWAP
    include_bands: bool = True  # VWAP standard deviation bands
    band_multipliers: List[float] = field(default_factory=lambda: [1.0, 2.0])


@dataclass 
class CandleConfig:
    """Primary timeframe candle configuration"""
    primary_timeframe_minutes: int = 120  # 2-hour candles
    use_session_breaks: bool = True  # Reset at session boundaries


@dataclass
class ExtensionConfig:
    """Extension duration predictor configuration"""
    # Trigger thresholds (number of consecutive candles)
    watching_threshold: int = 1    # Start watching
    alert_threshold: int = 2       # 4 hours - prepare
    high_prob_threshold: int = 3   # 6 hours - look for entry
    extreme_threshold: int = 4     # 8+ hours - high conviction
    
    # Extension detection
    min_extension_atr: float = 0.5  # Minimum ATR distance to count as extended
    
    # Snap-back probabilities by candle count
    snap_back_probs: Dict[int, float] = field(default_factory=lambda: {
        1: 0.45,
        2: 0.55,
        3: 0.65,
        4: 0.75,
        5: 0.80,
        6: 0.85,
    })
    
    # Adjustments
    rejection_candle_bonus: float = 0.10  # Add to snap-back prob
    declining_volume_bonus: float = 0.05  # Add if volume declining
    extreme_extension_bonus: float = 0.05  # Add if > 2 ATR extended


@dataclass
class SignalConfig:
    """Signal generation parameters"""
    # Mean Reversion
    mr_enabled: bool = True
    mr_min_extension_atr: float = 1.5  # Min ATR extension from value
    mr_rejection_wick_ratio: float = 0.6  # Wick must be 60% of candle range
    mr_target_level: str = "poc"  # Target: poc, vwap, or opposite_va
    
    # Trend Following
    trend_enabled: bool = True
    trend_vwap_slope_periods: int = 5  # Periods to calc VWAP slope
    trend_min_slope: float = 0.001  # Min slope for trend confirmation
    trend_poc_alignment_required: bool = True  # M > W > D POC alignment
    
    # Hybrid Filter
    require_bias_confirmation: bool = True  # POC stack must confirm direction
    mixed_bias_mr_only: bool = True  # Only MR trades when bias is mixed
    
    # Extension Duration Integration
    use_extension_duration: bool = True  # Use time × extension predictor
    min_extension_candles: int = 2  # Minimum candles for extension signal
    extension_quality_threshold: float = 50  # Min quality score


@dataclass
class RiskConfig:
    """Position sizing and risk management"""
    max_risk_per_trade_pct: float = 0.01  # 1% of account per trade
    max_position_pct: float = 0.10  # Max 10% of account in one position
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0  # Stop at 2x ATR
    scale_out_levels: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0])
    scale_out_pcts: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])


@dataclass
class SystemConfig:
    """Master system configuration"""
    api: APIConfig = field(default_factory=APIConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    volume_profile: VolumeProfileConfig = field(default_factory=VolumeProfileConfig)
    vwap: VWAPConfig = field(default_factory=VWAPConfig)
    candles: CandleConfig = field(default_factory=CandleConfig)
    extension: ExtensionConfig = field(default_factory=ExtensionConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    # Watchlist
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT"])
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/trading_system.log"


# Default configuration instance
DEFAULT_CONFIG = SystemConfig()
