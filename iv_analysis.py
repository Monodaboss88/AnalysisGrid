"""
Implied Volatility Analysis & Metrics
=====================================
Calculates IV Rank, IV Percentile, and volatility regime indicators.
Helps determine optimal options strategies based on IV environment.

Author: Strategic Edge Flow
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime, timedelta


@dataclass
class IVMetrics:
    """IV analysis metrics"""
    current_iv: float
    iv_rank: float          # 0-100 (current IV vs 52-week high/low)
    iv_percentile: float    # 0-100 (% of days IV was lower)
    hv_20: float           # 20-day historical volatility
    hv_vs_iv: float        # HV - IV (negative = IV > HV = expensive options)
    iv_regime: str         # LOW, NORMAL, ELEVATED, HIGH, EXTREME
    z_score: float         # Standard deviations from mean
    
    # Context
    iv_52w_high: float
    iv_52w_low: float
    iv_mean_30d: float
    
    # Recommendations
    strategy_bias: str     # BUY_OPTIONS, SELL_OPTIONS, NEUTRAL
    warnings: List[str]


class IVAnalyzer:
    """Analyze implied volatility metrics"""
    
    def calculate_historical_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """Calculate historical volatility from price data"""
        if len(df) < window:
            return 0.0
            
        # Log returns
        log_returns = np.log(df['close'] / df['close'].shift(1))
        
        # Annualized volatility
        volatility = log_returns.rolling(window=window).std() * np.sqrt(252)
        
        return volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0.0
        
    def calculate_iv_metrics(
        self,
        symbol: str,
        current_iv: float,
        historical_df: Optional[pd.DataFrame] = None
    ) -> IVMetrics:
        """
        Calculate comprehensive IV metrics
        
        Args:
            symbol: Stock symbol
            current_iv: Current implied volatility (as decimal, e.g., 0.50 for 50%)
            historical_df: Optional pre-fetched historical data
        """
        
        # Fetch historical data if not provided
        if historical_df is None:
            from polygon_data import get_bars
            historical_df = get_bars(symbol, period="1y", interval="1d")
            if historical_df.empty:
                return self._get_default_metrics(current_iv)
                
        historical_df.columns = [c.lower() for c in historical_df.columns]
        
        # Calculate historical volatility
        hv_20 = self.calculate_historical_volatility(historical_df, window=20)
        
        # Get options chain to build IV history (approximate)
        iv_history = self._estimate_iv_history(symbol, historical_df)
        
        if len(iv_history) < 30:
            return self._get_default_metrics(current_iv, hv_20)
            
        # Calculate IV Rank (52-week high/low)
        iv_52w_high = iv_history[-252:].max() if len(iv_history) >= 252 else iv_history.max()
        iv_52w_low = iv_history[-252:].min() if len(iv_history) >= 252 else iv_history.min()
        
        if iv_52w_high == iv_52w_low:
            iv_rank = 50.0
        else:
            iv_rank = ((current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low)) * 100
            
        # Calculate IV Percentile (% of days IV was lower)
        iv_percentile = (iv_history[-252:] < current_iv).sum() / min(len(iv_history), 252) * 100
        
        # Calculate mean and z-score
        iv_mean_30d = iv_history[-30:].mean() if len(iv_history) >= 30 else iv_history.mean()
        iv_std_30d = iv_history[-30:].std() if len(iv_history) >= 30 else iv_history.std()
        z_score = (current_iv - iv_mean_30d) / iv_std_30d if iv_std_30d > 0 else 0
        
        # HV vs IV comparison
        hv_vs_iv = hv_20 - current_iv
        
        # Determine IV regime
        iv_regime = self._classify_iv_regime(current_iv, iv_rank, iv_percentile)
        
        # Strategy bias
        strategy_bias = self._determine_strategy_bias(iv_rank, iv_percentile, hv_vs_iv)
        
        # Generate warnings
        warnings = self._generate_warnings(current_iv, iv_rank, hv_vs_iv, z_score)
        
        return IVMetrics(
            current_iv=round(current_iv, 4),
            iv_rank=round(iv_rank, 1),
            iv_percentile=round(iv_percentile, 1),
            hv_20=round(hv_20, 4),
            hv_vs_iv=round(hv_vs_iv, 4),
            iv_regime=iv_regime,
            z_score=round(z_score, 2),
            iv_52w_high=round(iv_52w_high, 4),
            iv_52w_low=round(iv_52w_low, 4),
            iv_mean_30d=round(iv_mean_30d, 4),
            strategy_bias=strategy_bias,
            warnings=warnings
        )
        
    def _estimate_iv_history(self, symbol: str, df: pd.DataFrame) -> np.ndarray:
        """
        Estimate historical IV using realized volatility + implied component
        (Simplified - real implementation would use actual IV data from options chain history)
        """
        # Calculate rolling HV as proxy for IV
        log_returns = np.log(df['close'] / df['close'].shift(1))
        rolling_hv = log_returns.rolling(window=20).std() * np.sqrt(252)
        
        # Add volatility premium (IV typically > HV)
        # Estimate based on market regime
        vol_premium = rolling_hv * 0.15  # IV typically 15% higher than HV
        
        estimated_iv = rolling_hv + vol_premium
        
        return estimated_iv.dropna().values
        
    def _classify_iv_regime(self, current_iv: float, iv_rank: float, iv_percentile: float) -> str:
        """Classify the current IV environment"""
        
        # Use both IV Rank and absolute level
        if current_iv > 1.5:  # 150%+
            return "EXTREME"
        elif current_iv > 1.0:  # 100-150%
            return "HIGH"
        elif iv_rank > 70 or iv_percentile > 80:
            return "ELEVATED"
        elif iv_rank < 30 or iv_percentile < 20:
            return "LOW"
        else:
            return "NORMAL"
            
    def _determine_strategy_bias(
        self,
        iv_rank: float,
        iv_percentile: float,
        hv_vs_iv: float
    ) -> str:
        """Determine whether to buy or sell options"""
        
        # High IV + overpriced = sell options (credit spreads, iron condors)
        if iv_rank > 60 and hv_vs_iv < -0.10:
            return "SELL_OPTIONS"
            
        # Low IV + underpriced = buy options (long calls/puts, debit spreads)
        elif iv_rank < 40 and hv_vs_iv > 0:
            return "BUY_OPTIONS"
            
        # Middle ground
        else:
            return "NEUTRAL"
            
    def _generate_warnings(
        self,
        current_iv: float,
        iv_rank: float,
        hv_vs_iv: float,
        z_score: float
    ) -> List[str]:
        """Generate warnings based on IV metrics"""
        
        warnings = []
        
        # Extreme IV
        if current_iv > 1.0:
            warnings.append(f"🚨 EXTREME IV: {current_iv:.1%} - earnings or major catalyst likely imminent")
            warnings.append("   → Post-event IV crush could destroy 40-70% of option value")
            warnings.append("   → Consider waiting until after catalyst or size down to 0.25R")
            
        # Very high IV rank
        if iv_rank > 80:
            warnings.append(f"⚠️ IV RANK: {iv_rank:.0f}th percentile - near 52-week high")
            warnings.append("   → Options are expensive - favor selling vs buying")
            warnings.append("   → If buying, expect mean reversion (IV crush)")
            
        # Very low IV rank
        if iv_rank < 20:
            warnings.append(f"💡 IV RANK: {iv_rank:.0f}th percentile - near 52-week low")
            warnings.append("   → Options are cheap - favorable for buying")
            warnings.append("   → Consider debit spreads or long options")
            
        # HV >> IV (options underpriced)
        if hv_vs_iv > 0.15:
            warnings.append(f"💡 OPTIONS CHEAP: HV {hv_vs_iv:.1%} higher than IV")
            warnings.append("   → Realized moves > implied moves = option buyers win recently")
            
        # IV >> HV (options overpriced)
        elif hv_vs_iv < -0.20:
            warnings.append(f"⚠️ OPTIONS EXPENSIVE: IV {abs(hv_vs_iv):.1%} higher than HV")
            warnings.append("   → Implied moves > realized moves = option sellers win recently")
            
        # Extreme z-score
        if abs(z_score) > 2:
            direction = "HIGH" if z_score > 0 else "LOW"
            warnings.append(f"⚠️ IV ANOMALY: {abs(z_score):.1f} std devs {direction}")
            warnings.append("   → Mean reversion likely - current IV is unsustainable")
            
        return warnings
        
    def _get_default_metrics(self, current_iv: float, hv_20: float = 0.0) -> IVMetrics:
        """Return default metrics when data is insufficient"""
        
        return IVMetrics(
            current_iv=current_iv,
            iv_rank=50.0,
            iv_percentile=50.0,
            hv_20=hv_20,
            hv_vs_iv=hv_20 - current_iv if hv_20 > 0 else 0.0,
            iv_regime="UNKNOWN",
            z_score=0.0,
            iv_52w_high=current_iv,
            iv_52w_low=current_iv,
            iv_mean_30d=current_iv,
            strategy_bias="NEUTRAL",
            warnings=["⚠️ Insufficient data for complete IV analysis"]
        )


# Example usage
if __name__ == "__main__":
    analyzer = IVAnalyzer()
    
    # Analyze the high IV case from the image (114.6%)
    metrics = analyzer.calculate_iv_metrics("HD", current_iv=1.146)
    
    print(f"Symbol: HD")
    print(f"Current IV: {metrics.current_iv:.1%}")
    print(f"IV Rank: {metrics.iv_rank:.1f} (52-week range: {metrics.iv_52w_low:.1%} - {metrics.iv_52w_high:.1%})")
    print(f"IV Percentile: {metrics.iv_percentile:.1f}%")
    print(f"Historical Vol (20d): {metrics.hv_20:.1%}")
    print(f"HV vs IV: {metrics.hv_vs_iv:+.1%}")
    print(f"\nIV Regime: {metrics.iv_regime}")
    print(f"Strategy Bias: {metrics.strategy_bias}")
    print(f"Z-Score: {metrics.z_score:.2f}")
    print(f"\nWarnings:")
    for warning in metrics.warnings:
        print(f"  {warning}")
