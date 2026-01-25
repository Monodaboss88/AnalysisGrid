"""
Chart Input Analyzer
====================
Input your chart values directly and get instant MTF analysis.
Includes alert triggers and trade tracking.

Author: Rob's Trading Systems
Version: 1.0.0
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ChartInput:
    """Single timeframe chart input"""
    price: float
    vah: float
    poc: float
    val: float
    vwap: float
    rsi: float
    timeframe: str = ""
    # Volume analysis fields
    rvol: float = 1.0  # Relative volume (1.0 = average)
    volume_trend: str = "neutral"  # "increasing", "decreasing", "neutral"
    volume_divergence: bool = False  # Price up + volume down = bearish divergence
    atr: float = 0.0  # Average True Range for extension calculations
    has_rejection: bool = False  # Whether rejection candle pattern detected
    
    def __post_init__(self):
        self.timeframe = self.timeframe.upper()


# Signal Type Classification (from concept/signal_generator.py)
class SignalType:
    """Signal classification - Mean Reversion vs Trend"""
    NONE = "none"
    LONG_MR = "long_mean_reversion"      # Extended below VAL, expect bounce
    SHORT_MR = "short_mean_reversion"    # Extended above VAH, expect rejection
    LONG_TREND = "long_trend"            # Holding above VAH, trend continuation
    SHORT_TREND = "short_trend"          # Holding below VAL, trend continuation
    LONG_SETUP = "long_setup"            # Generic bullish (legacy)
    SHORT_SETUP = "short_setup"          # Generic bearish (legacy)


@dataclass
class AnalysisResult:
    """Result from analyzing chart input"""
    timeframe: str
    signal: str
    signal_emoji: str
    bull_score: float
    bear_score: float
    confidence: float
    high_prob: float
    low_prob: float
    position: str
    vwap_zone: str
    rsi_zone: str
    notes: List[str]
    # Volume metrics
    rvol: float = 1.0
    volume_trend: str = "neutral"
    volume_divergence: bool = False
    # Signal classification (NEW from concept system)
    signal_type: str = "none"  # SignalType value
    signal_strength: str = "moderate"  # weak, moderate, strong, very_strong
    atr: float = 0.0
    extension_atr: float = 0.0  # How many ATR extended from key level
    has_rejection: bool = False


@dataclass 
class MTFResult:
    """Combined multi-timeframe result"""
    symbol: str
    timestamp: str
    dominant_signal: str
    signal_emoji: str
    confluence_pct: float
    weighted_bull: float
    weighted_bear: float
    high_prob: float
    low_prob: float
    timeframe_results: Dict[str, AnalysisResult]
    key_levels: Dict[str, float]
    trade_plan: Optional[Dict] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class AlertTrigger:
    """Price alert trigger"""
    symbol: str
    level: float
    direction: str  # "above" or "below"
    action: str     # "LONG", "SHORT", "EXIT", "ALERT"
    note: str = ""
    triggered: bool = False
    created_at: str = ""
    triggered_at: str = ""


@dataclass
class TradeSetup:
    """Logged trade setup for tracking"""
    symbol: str
    timeframe: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: Optional[float] = None
    signal: str = ""
    confidence: float = 0
    rr_ratio: float = 0
    notes: str = ""
    status: str = "PENDING"  # PENDING, ACTIVE, WIN, LOSS, SCRATCH
    created_at: str = ""
    entry_time: str = ""
    exit_time: str = ""
    exit_price: float = 0
    result_pct: float = 0


# =============================================================================
# SIGNAL SCORER (Simplified standalone version)
# =============================================================================

class ChartAnalyzer:
    """
    Analyzes chart inputs and produces signals
    
    This is a standalone scorer that takes YOUR chart values
    and applies the same logic as the full scanner.
    """
    
    # Thresholds - adjusted for higher scores
    STRONG_THRESHOLD = 60  # Signal threshold
    MODERATE_THRESHOLD = 45
    MIN_SCORE_GAP = 15
    
    def analyze_single(self, chart: ChartInput) -> AnalysisResult:
        """Analyze a single timeframe"""
        
        price = chart.price
        bull_score = 0.0
        bear_score = 0.0
        notes = []
        
        # =====================================================================
        # POSITION IN VALUE (40 points max) - INCREASED
        # =====================================================================
        
        if price > chart.vah:
            position = "ABOVE_VALUE"
            bull_score += 40
            notes.append("Above value - bullish position")
        elif price < chart.val:
            position = "BELOW_VALUE"
            bear_score += 35
            # But also potential bounce
            bull_score += 10
            notes.append("Price extended below VAL - watch for bounce")
        else:
            position = "IN_VALUE"
            if price > chart.poc:
                bull_score += 25
                notes.append("Inside value, above POC")
            else:
                bear_score += 25
                notes.append("Inside value, below POC")
        
        # =====================================================================
        # VWAP ANALYSIS (25 points max) - INCREASED
        # =====================================================================
        
        vwap_dev = (price - chart.vwap) / chart.vwap * 100
        
        if vwap_dev > 2.0:
            vwap_zone = "EXTREME_ABOVE"
            bull_score += 20  # Strong bullish momentum
            notes.append(f"Price extreme above VWAP (+{vwap_dev:.1f}%) - strong momentum")
        elif vwap_dev > 0.5:
            vwap_zone = "ABOVE_1SD"
            bull_score += 25
            notes.append(f"Price above VWAP (+{vwap_dev:.1f}%) - buyers in control")
        elif vwap_dev > -0.5:
            vwap_zone = "AT_VWAP"
            bull_score += 8
            bear_score += 8
            notes.append("⚡ Price at VWAP - key decision point")
        elif vwap_dev > -2.0:
            vwap_zone = "BELOW_1SD"
            bear_score += 25
            notes.append(f"Price below VWAP ({vwap_dev:.1f}%) - sellers in control")
        else:
            vwap_zone = "EXTREME_BELOW"
            bear_score += 20  # Strong bearish
            bull_score += 8
            notes.append(f"Price extreme below VWAP ({vwap_dev:.1f}%) - bounce possible")
        
        # =====================================================================
        # RSI ANALYSIS (35 points max) - BOOSTED
        # =====================================================================
        
        rsi = chart.rsi
        
        if rsi >= 75:
            rsi_zone = "OVERBOUGHT"
            bear_score += 12  # Slight caution but trend is strong
            notes.append(f"RSI overbought ({rsi:.1f}) - momentum strong")
        elif rsi >= 65:
            rsi_zone = "NEAR_OVERBOUGHT"
            bull_score += 25  # Still bullish momentum
            notes.append(f"RSI strong ({rsi:.1f})")
        elif rsi >= 55:
            rsi_zone = "BULLISH"
            bull_score += 30
            notes.append(f"RSI bullish ({rsi:.1f})")
        elif rsi >= 45:
            rsi_zone = "NEUTRAL"
            bull_score += 5
            bear_score += 5
            notes.append(f"RSI neutral ({rsi:.1f})")
        elif rsi >= 35:
            rsi_zone = "BEARISH"
            bear_score += 30
            notes.append(f"RSI bearish ({rsi:.1f})")
        elif rsi >= 30:
            rsi_zone = "NEAR_OVERSOLD"
            bear_score += 25  # Still bearish but may bounce
            notes.append(f"RSI weak ({rsi:.1f}) - watch for bounce")
        else:
            rsi_zone = "OVERSOLD"
            bear_score += 12  # Slight caution, bounce potential
            bull_score += 8
            notes.append(f"RSI oversold ({rsi:.1f}) - bounce likely ✓")
        
        # =====================================================================
        # VOLUME ANALYSIS (15 points max) - NEW
        # =====================================================================
        
        rvol = getattr(chart, 'rvol', 1.0)
        volume_trend = getattr(chart, 'volume_trend', 'neutral')
        volume_divergence = getattr(chart, 'volume_divergence', False)
        
        # Relative Volume scoring
        if rvol >= 2.0:
            notes.append(f"🔥 High volume ({rvol:.1f}x avg) - strong conviction")
            # High volume confirms the move
            if bull_score > bear_score:
                bull_score += 15
            else:
                bear_score += 15
        elif rvol >= 1.5:
            notes.append(f"📈 Above avg volume ({rvol:.1f}x)")
            if bull_score > bear_score:
                bull_score += 10
            else:
                bear_score += 10
        elif rvol <= 0.5:
            notes.append(f"⚠️ Low volume ({rvol:.1f}x) - weak conviction")
            # Low volume = less reliable signal
            bull_score = max(0, bull_score - 5)
            bear_score = max(0, bear_score - 5)
        
        # Volume Trend scoring
        if volume_trend == "increasing":
            notes.append("📊 Volume increasing - trend strengthening")
            if bull_score > bear_score:
                bull_score += 10
            else:
                bear_score += 10
        elif volume_trend == "decreasing":
            notes.append("📉 Volume decreasing - momentum fading")
            # Fading volume = potential reversal
            if bull_score > bear_score:
                bull_score = max(0, bull_score - 3)
            else:
                bear_score = max(0, bear_score - 3)
        
        # Volume Divergence Warning (critical signal)
        if volume_divergence:
            notes.append("⚠️ VOLUME DIVERGENCE - price vs volume conflict!")
            # Divergence reduces confidence in the current direction
            if bull_score > bear_score:
                # Price up but volume down = bearish divergence
                bear_score += 10
                bull_score = max(0, bull_score - 5)
            else:
                # Price down but volume down = bullish divergence
                bull_score += 10
                bear_score = max(0, bear_score - 5)
        
        # =====================================================================
        # DETERMINE SIGNAL
        # =====================================================================
        
        bull_score = max(0, min(100, bull_score))
        bear_score = max(0, min(100, bear_score))
        
        score_gap = abs(bull_score - bear_score)
        max_score = max(bull_score, bear_score)
        min_score = min(bull_score, bear_score)
        
        if max_score < 25 and min_score < 15:
            signal = "YELLOW"
            emoji = "🟡"
            confidence = 50
            notes.append("Low signal strength - insufficient data")
        elif max_score < self.MODERATE_THRESHOLD and min_score >= 15:
            signal = "YELLOW"
            emoji = "🟡"
            if bull_score > bear_score:
                confidence = 45 + (score_gap / 2)
                notes.append("Mixed signals, slight bullish lean - YELLOW")
            elif bear_score > bull_score:
                confidence = 45 + (score_gap / 2)
                notes.append("Mixed signals, slight bearish lean - YELLOW")
            else:
                confidence = 40
                notes.append("Mixed signals - YELLOW")
        elif score_gap < self.MIN_SCORE_GAP:
            signal = "YELLOW"
            emoji = "🟡"
            confidence = 50 - (score_gap / self.MIN_SCORE_GAP * 25)
            notes.append(f"Mixed signals - gap only {score_gap:.1f} points")
        elif bull_score > bear_score and bull_score >= self.STRONG_THRESHOLD:
            signal = "LONG_SETUP"
            emoji = "🟢"
            # Confidence based on actual score strength + gap bonus
            # Score 50 = 65%, Score 70 = 80%, Score 90+ = 95%
            confidence = min(95, 40 + (bull_score * 0.5) + (score_gap * 0.1))
            notes.append("✓ Long setup confirmed")
        elif bear_score > bull_score and bear_score >= self.STRONG_THRESHOLD:
            signal = "SHORT_SETUP"
            emoji = "🔴"
            # Confidence based on actual score strength + gap bonus
            # Score 50 = 65%, Score 70 = 80%, Score 90+ = 95%
            confidence = min(95, 40 + (bear_score * 0.5) + (score_gap * 0.1))
            notes.append("✓ Short setup confirmed")
        elif bull_score > bear_score:
            signal = "YELLOW"
            emoji = "🟡"
            confidence = 40 + (score_gap / 2)
            notes.append("Leaning bullish but not confirmed - YELLOW")
        elif bear_score > bull_score:
            signal = "YELLOW"
            emoji = "🟡"
            confidence = 40 + (score_gap / 2)
            notes.append("Leaning bearish but not confirmed - YELLOW")
        else:
            signal = "YELLOW"
            emoji = "🟡"
            confidence = 50
            notes.append("Undetermined - waiting for clearer signal")
        
        # Probabilities
        total = bull_score + bear_score
        high_prob = (bull_score / total * 100) if total > 0 else 50
        low_prob = (bear_score / total * 100) if total > 0 else 50
        
        # =================================================================
        # SIGNAL CLASSIFICATION (from concept/signal_generator.py)
        # Determine if this is Mean Reversion or Trend
        # =================================================================
        signal_type = SignalType.NONE
        signal_strength = "moderate"
        atr = getattr(chart, 'atr', 0) or 0
        has_rejection = getattr(chart, 'has_rejection', False)
        extension_atr = 0.0
        
        if atr > 0:
            # Calculate extension from key levels
            ext_from_vah = (price - chart.vah) / atr if chart.vah else 0
            ext_from_val = (chart.val - price) / atr if chart.val else 0
            extension_atr = max(ext_from_vah, ext_from_val) if ext_from_vah > 0 or ext_from_val > 0 else 0
        
        if signal == "LONG_SETUP":
            if position == "BELOW_VALUE" and extension_atr >= 1.5:
                # Extended below VAL = Mean Reversion Long
                signal_type = SignalType.LONG_MR
                notes.append(f"📉 Mean Reversion: {extension_atr:.1f} ATR below value")
            elif position == "ABOVE_VALUE":
                # Above VAH = Trend Long  
                signal_type = SignalType.LONG_TREND
                notes.append("📈 Trend: holding above value area")
            else:
                signal_type = SignalType.LONG_SETUP
        elif signal == "SHORT_SETUP":
            if position == "ABOVE_VALUE" and extension_atr >= 1.5:
                # Extended above VAH = Mean Reversion Short
                signal_type = SignalType.SHORT_MR
                notes.append(f"📈 Mean Reversion: {extension_atr:.1f} ATR above value")
            elif position == "BELOW_VALUE":
                # Below VAL = Trend Short
                signal_type = SignalType.SHORT_TREND
                notes.append("📉 Trend: holding below value area")
            else:
                signal_type = SignalType.SHORT_SETUP
        
        # Determine signal strength based on extension and confirmation
        if extension_atr >= 3.0:
            signal_strength = "very_strong"
        elif extension_atr >= 2.0 or has_rejection:
            signal_strength = "strong"
        elif extension_atr >= 1.0:
            signal_strength = "moderate"
        else:
            signal_strength = "weak"
        
        return AnalysisResult(
            timeframe=chart.timeframe,
            signal=signal,
            signal_emoji=emoji,
            bull_score=bull_score,
            bear_score=bear_score,
            confidence=confidence,
            high_prob=high_prob,
            low_prob=low_prob,
            position=position,
            vwap_zone=vwap_zone,
            rsi_zone=rsi_zone,
            notes=notes,
            rvol=rvol,
            volume_trend=volume_trend,
            volume_divergence=volume_divergence,
            signal_type=signal_type,
            signal_strength=signal_strength,
            atr=atr,
            extension_atr=extension_atr,
            has_rejection=has_rejection
        )
    
    def analyze_mtf(self, 
                    symbol: str,
                    charts: Dict[str, ChartInput],
                    current_price: float = None) -> MTFResult:
        """
        Analyze multiple timeframes and produce combined signal
        
        Args:
            symbol: Ticker symbol
            charts: Dict of timeframe -> ChartInput
            current_price: Override price (uses first chart price if not provided)
        
        Returns:
            MTFResult with combined analysis
        """
        
        if not charts:
            raise ValueError("No chart data provided")
        
        # Analyze each timeframe
        results = {}
        for tf, chart in charts.items():
            chart.timeframe = tf
            results[tf] = self.analyze_single(chart)
        
        # Weights (higher TF = more weight)
        weight_map = {
            "5MIN": 0.05, "15MIN": 0.10, "30MIN": 0.15,
            "1HR": 0.20, "2HR": 0.25, "4HR": 0.30, "DAILY": 0.35
        }
        
        # Normalize weights for provided timeframes
        total_weight = sum(weight_map.get(tf, 0.15) for tf in results.keys())
        weights = {tf: weight_map.get(tf, 0.15) / total_weight for tf in results.keys()}
        
        # Weighted scores
        weighted_bull = sum(results[tf].bull_score * weights[tf] for tf in results)
        weighted_bear = sum(results[tf].bear_score * weights[tf] for tf in results)
        
        # Count signals
        signals = [r.signal for r in results.values()]
        long_count = signals.count("LONG_SETUP")
        short_count = signals.count("SHORT_SETUP")
        yellow_count = signals.count("YELLOW")
        
        # Positions
        below_value = sum(1 for r in results.values() if r.position == "BELOW_VALUE")
        above_value = sum(1 for r in results.values() if r.position == "ABOVE_VALUE")
        
        # Determine dominant signal
        notes = []
        total_tf = len(results)
        
        if long_count >= total_tf / 2:
            dominant = "LONG_SETUP"
            emoji = "🟢"
            confluence = long_count / total_tf * 100
            notes.append(f"LONG confirmed on {long_count}/{total_tf} timeframes")
        elif short_count >= total_tf / 2:
            dominant = "SHORT_SETUP"
            emoji = "🔴"
            confluence = short_count / total_tf * 100
            notes.append(f"SHORT confirmed on {short_count}/{total_tf} timeframes")
        elif below_value >= total_tf * 0.75 and weighted_bear > weighted_bull:
            dominant = "SHORT_SETUP" if weighted_bear > 50 else "YELLOW"
            emoji = "🔴" if dominant == "SHORT_SETUP" else "🟡"
            confluence = below_value / total_tf * 100
            notes.append(f"BELOW VALUE on {below_value}/{total_tf} timeframes - bearish structure")
        elif above_value >= total_tf * 0.75 and weighted_bull > weighted_bear:
            dominant = "LONG_SETUP" if weighted_bull > 50 else "YELLOW"
            emoji = "🟢" if dominant == "LONG_SETUP" else "🟡"
            confluence = above_value / total_tf * 100
            notes.append(f"ABOVE VALUE on {above_value}/{total_tf} timeframes - bullish structure")
        else:
            dominant = "YELLOW"
            emoji = "🟡"
            confluence = yellow_count / total_tf * 100
            if weighted_bull > weighted_bear:
                notes.append("Mixed signals - slight bullish lean")
            elif weighted_bear > weighted_bull:
                notes.append("Mixed signals - slight bearish lean")
            else:
                notes.append("Mixed signals - no clear direction")
        
        # Probabilities
        total = weighted_bull + weighted_bear
        high_prob = (weighted_bull / total * 100) if total > 0 else 50
        low_prob = (weighted_bear / total * 100) if total > 0 else 50
        
        # Compile key levels from all timeframes
        key_levels = {}
        for tf, chart in charts.items():
            key_levels[f"{tf}_VAH"] = chart.vah
            key_levels[f"{tf}_POC"] = chart.poc
            key_levels[f"{tf}_VAL"] = chart.val
            key_levels[f"{tf}_VWAP"] = chart.vwap
        
        # Add current price
        price = current_price or list(charts.values())[0].price
        key_levels["CURRENT"] = price
        
        return MTFResult(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            dominant_signal=dominant,
            signal_emoji=emoji,
            confluence_pct=confluence,
            weighted_bull=weighted_bull,
            weighted_bear=weighted_bear,
            high_prob=high_prob,
            low_prob=low_prob,
            timeframe_results=results,
            key_levels=key_levels,
            notes=notes
        )


# =============================================================================
# ALERT MANAGER
# =============================================================================

class AlertManager:
    """Manages price alerts and triggers"""
    
    def __init__(self, data_file: str = "alerts.json"):
        self.data_file = data_file
        self.alerts: List[AlertTrigger] = []
        self.load()
    
    def add_alert(self,
                  symbol: str,
                  level: float,
                  direction: str,
                  action: str,
                  note: str = "") -> AlertTrigger:
        """Add a new alert"""
        alert = AlertTrigger(
            symbol=symbol.upper(),
            level=level,
            direction=direction.lower(),
            action=action.upper(),
            note=note,
            created_at=datetime.now().isoformat()
        )
        self.alerts.append(alert)
        self.save()
        return alert
    
    def check_alerts(self, symbol: str, current_price: float) -> List[AlertTrigger]:
        """Check if any alerts are triggered"""
        triggered = []
        
        for alert in self.alerts:
            if alert.symbol != symbol.upper() or alert.triggered:
                continue
            
            if alert.direction == "above" and current_price >= alert.level:
                alert.triggered = True
                alert.triggered_at = datetime.now().isoformat()
                triggered.append(alert)
            elif alert.direction == "below" and current_price <= alert.level:
                alert.triggered = True
                alert.triggered_at = datetime.now().isoformat()
                triggered.append(alert)
        
        if triggered:
            self.save()
        
        return triggered
    
    def get_active_alerts(self, symbol: str = None) -> List[AlertTrigger]:
        """Get all active (non-triggered) alerts"""
        alerts = [a for a in self.alerts if not a.triggered]
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol.upper()]
        return alerts
    
    def clear_triggered(self):
        """Remove all triggered alerts"""
        self.alerts = [a for a in self.alerts if not a.triggered]
        self.save()
    
    def remove_alert(self, symbol: str, level: float) -> bool:
        """Remove a specific alert"""
        for i, alert in enumerate(self.alerts):
            if alert.symbol == symbol.upper() and abs(alert.level - level) < 0.01:
                self.alerts.pop(i)
                self.save()
                return True
        return False
    
    def save(self):
        """Save alerts to file"""
        data = [asdict(a) for a in self.alerts]
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load alerts from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                self.alerts = [AlertTrigger(**a) for a in data]
            except:
                self.alerts = []


# =============================================================================
# TRADE TRACKER
# =============================================================================

class TradeTracker:
    """Tracks trade setups and outcomes for validation"""
    
    def __init__(self, data_file: str = "trades.json"):
        self.data_file = data_file
        self.trades: List[TradeSetup] = []
        self.load()
    
    def log_setup(self,
                  symbol: str,
                  timeframe: str,
                  direction: str,
                  entry_price: float,
                  stop_loss: float,
                  target_1: float,
                  target_2: float = None,
                  signal: str = "",
                  confidence: float = 0,
                  notes: str = "") -> TradeSetup:
        """Log a new trade setup"""
        
        # Calculate R:R
        risk = abs(entry_price - stop_loss)
        reward = abs(target_1 - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        trade = TradeSetup(
            symbol=symbol.upper(),
            timeframe=timeframe,
            direction=direction.upper(),
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            signal=signal,
            confidence=confidence,
            rr_ratio=round(rr_ratio, 2),
            notes=notes,
            created_at=datetime.now().isoformat()
        )
        
        self.trades.append(trade)
        self.save()
        return trade
    
    def update_status(self, 
                      symbol: str, 
                      status: str,
                      exit_price: float = None) -> Optional[TradeSetup]:
        """Update trade status"""
        for trade in reversed(self.trades):
            if trade.symbol == symbol.upper() and trade.status == "PENDING":
                trade.status = status.upper()
                if status.upper() == "ACTIVE":
                    trade.entry_time = datetime.now().isoformat()
                elif status.upper() in ["WIN", "LOSS", "SCRATCH"]:
                    trade.exit_time = datetime.now().isoformat()
                    if exit_price:
                        trade.exit_price = exit_price
                        if trade.direction == "LONG":
                            trade.result_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
                        else:
                            trade.result_pct = (trade.entry_price - exit_price) / trade.entry_price * 100
                self.save()
                return trade
        return None
    
    def get_pending(self, symbol: str = None) -> List[TradeSetup]:
        """Get pending setups"""
        trades = [t for t in self.trades if t.status == "PENDING"]
        if symbol:
            trades = [t for t in trades if t.symbol == symbol.upper()]
        return trades
    
    def get_stats(self) -> Dict:
        """Get win/loss statistics"""
        completed = [t for t in self.trades if t.status in ["WIN", "LOSS", "SCRATCH"]]
        wins = [t for t in completed if t.status == "WIN"]
        losses = [t for t in completed if t.status == "LOSS"]
        
        return {
            "total_trades": len(completed),
            "wins": len(wins),
            "losses": len(losses),
            "scratches": len(completed) - len(wins) - len(losses),
            "win_rate": len(wins) / len(completed) * 100 if completed else 0,
            "avg_winner": sum(t.result_pct for t in wins) / len(wins) if wins else 0,
            "avg_loser": sum(t.result_pct for t in losses) / len(losses) if losses else 0,
        }
    
    def save(self):
        """Save trades to file"""
        data = [asdict(t) for t in self.trades]
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load trades from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                self.trades = [TradeSetup(**t) for t in data]
            except:
                self.trades = []


# =============================================================================
# MAIN INTERFACE
# =============================================================================

class ChartInputSystem:
    """
    Main interface for chart input analysis
    
    Usage:
        system = ChartInputSystem()
        
        # Single timeframe
        result = system.analyze("META", price=619.28, vah=667.72, 
                                poc=660.40, val=647.22, vwap=619.63, rsi=33.58)
        
        # Multi-timeframe
        result = system.analyze_mtf("META", {
            "30min": {"price": 619, "vah": 666, "poc": 660, "val": 642, "vwap": 621, "rsi": 46},
            "1hr": {"price": 619, "vah": 668, "poc": 658, "val": 648, "vwap": 621, "rsi": 44},
            ...
        })
        
        # Add alerts
        system.add_alert("META", 615, "below", "SHORT", "Break of support")
        system.add_alert("META", 647, "above", "LONG", "Reclaim VAL")
        
        # Log trade
        system.log_trade("META", "2HR", "LONG", entry=619, stop=613, target=647)
    """
    
    def __init__(self, data_dir: str = "./trade_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.analyzer = ChartAnalyzer()
        self.alerts = AlertManager(os.path.join(data_dir, "alerts.json"))
        self.tracker = TradeTracker(os.path.join(data_dir, "trades.json"))
    
    def analyze(self, 
                symbol: str,
                price: float,
                vah: float,
                poc: float,
                val: float,
                vwap: float,
                rsi: float,
                timeframe: str = "1HR",
                rvol: float = 1.0,
                volume_trend: str = "neutral",
                volume_divergence: bool = False,
                atr: float = 0.0,
                has_rejection: bool = False) -> AnalysisResult:
        """Analyze single timeframe from chart values"""
        
        chart = ChartInput(
            price=price,
            vah=vah,
            poc=poc,
            val=val,
            vwap=vwap,
            rsi=rsi,
            timeframe=timeframe,
            rvol=rvol,
            volume_trend=volume_trend,
            volume_divergence=volume_divergence,
            atr=atr,
            has_rejection=has_rejection
        )
        
        result = self.analyzer.analyze_single(chart)
        
        # Check alerts
        triggered = self.alerts.check_alerts(symbol, price)
        if triggered:
            for alert in triggered:
                result.notes.append(f"🚨 ALERT: {alert.action} trigger at ${alert.level:.2f}")
        
        return result
    
    def analyze_mtf(self,
                    symbol: str,
                    timeframes: Dict[str, Dict],
                    current_price: float = None) -> MTFResult:
        """
        Analyze multiple timeframes
        
        Args:
            symbol: Ticker symbol
            timeframes: Dict of timeframe -> {price, vah, poc, val, vwap, rsi}
        """
        
        charts = {}
        for tf, data in timeframes.items():
            charts[tf.upper()] = ChartInput(
                price=data.get("price", data.get("close", 0)),
                vah=data["vah"],
                poc=data["poc"],
                val=data["val"],
                vwap=data["vwap"],
                rsi=data["rsi"],
                timeframe=tf
            )
        
        result = self.analyzer.analyze_mtf(symbol, charts, current_price)
        
        # Check alerts
        price = current_price or list(charts.values())[0].price
        triggered = self.alerts.check_alerts(symbol, price)
        if triggered:
            for alert in triggered:
                result.notes.append(f"🚨 ALERT: {alert.action} trigger at ${alert.level:.2f}")
        
        return result
    
    def add_alert(self,
                  symbol: str,
                  level: float,
                  direction: str,
                  action: str,
                  note: str = "") -> AlertTrigger:
        """Add price alert"""
        return self.alerts.add_alert(symbol, level, direction, action, note)
    
    def get_alerts(self, symbol: str = None) -> List[AlertTrigger]:
        """Get active alerts"""
        return self.alerts.get_active_alerts(symbol)
    
    def log_trade(self,
                  symbol: str,
                  timeframe: str,
                  direction: str,
                  entry: float,
                  stop: float,
                  target: float,
                  target2: float = None,
                  signal: str = "",
                  confidence: float = 0,
                  notes: str = "") -> TradeSetup:
        """Log a trade setup"""
        return self.tracker.log_setup(
            symbol, timeframe, direction, entry, stop, target,
            target2, signal, confidence, notes
        )
    
    def update_trade(self, symbol: str, status: str, exit_price: float = None):
        """Update trade status (ACTIVE, WIN, LOSS, SCRATCH)"""
        return self.tracker.update_status(symbol, status, exit_price)
    
    def get_pending_trades(self, symbol: str = None) -> List[TradeSetup]:
        """Get pending trade setups"""
        return self.tracker.get_pending(symbol)
    
    def get_trade_stats(self) -> Dict:
        """Get trading statistics"""
        return self.tracker.get_stats()
    
    # =========================================================================
    # DISPLAY METHODS
    # =========================================================================
    
    def print_result(self, result: AnalysisResult) -> str:
        """Format single timeframe result"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"📊 {result.timeframe} ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"\n{result.signal_emoji} SIGNAL: {result.signal}")
        lines.append(f"   Confidence: {result.confidence:.1f}%")
        lines.append(f"\n   Bull Score: {result.bull_score:.1f}")
        lines.append(f"   Bear Score: {result.bear_score:.1f}")
        lines.append(f"\n   Position:  {result.position}")
        lines.append(f"   VWAP Zone: {result.vwap_zone}")
        lines.append(f"   RSI Zone:  {result.rsi_zone}")
        lines.append(f"\n📈 Scenarios:")
        lines.append(f"   HIGH: {result.high_prob:.0f}%")
        lines.append(f"   LOW:  {result.low_prob:.0f}%")
        lines.append(f"\n📝 Notes:")
        for note in result.notes:
            lines.append(f"   • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def print_mtf_result(self, result: MTFResult) -> str:
        """Format MTF result"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"🎯 {result.symbol} - MULTI-TIMEFRAME ANALYSIS")
        lines.append(f"   {result.timestamp}")
        lines.append("=" * 70)
        
        # Individual timeframes
        lines.append("\n📊 TIMEFRAME BREAKDOWN:")
        lines.append("-" * 70)
        lines.append(f"{'TF':<8} {'Signal':<12} {'Bull':<8} {'Bear':<8} {'Position':<15} {'RSI'}")
        lines.append("-" * 70)
        
        for tf, r in result.timeframe_results.items():
            lines.append(f"{tf:<8} {r.signal_emoji} {r.signal:<10} {r.bull_score:<8.0f} {r.bear_score:<8.0f} {r.position:<15} {r.rsi_zone}")
        
        # Combined result
        lines.append(f"\n{'=' * 70}")
        lines.append(f"🎯 COMBINED SIGNAL: {result.signal_emoji} {result.dominant_signal}")
        lines.append(f"   Confluence: {result.confluence_pct:.0f}%")
        lines.append(f"   Weighted Bull: {result.weighted_bull:.1f}")
        lines.append(f"   Weighted Bear: {result.weighted_bear:.1f}")
        lines.append(f"\n📈 SCENARIOS:")
        lines.append(f"   HIGH: {result.high_prob:.0f}%")
        lines.append(f"   LOW:  {result.low_prob:.0f}%")
        
        lines.append(f"\n📝 NOTES:")
        for note in result.notes:
            lines.append(f"   • {note}")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def print_alerts(self, symbol: str = None) -> str:
        """Format active alerts"""
        alerts = self.get_alerts(symbol)
        
        if not alerts:
            return "No active alerts"
        
        lines = []
        lines.append("🚨 ACTIVE ALERTS:")
        lines.append("-" * 50)
        for a in alerts:
            lines.append(f"   {a.symbol} {a.direction.upper()} ${a.level:.2f} → {a.action}")
            if a.note:
                lines.append(f"      Note: {a.note}")
        
        return "\n".join(lines)
    
    def print_pending_trades(self, symbol: str = None) -> str:
        """Format pending trades"""
        trades = self.get_pending_trades(symbol)
        
        if not trades:
            return "No pending trades"
        
        lines = []
        lines.append("📋 PENDING SETUPS:")
        lines.append("-" * 60)
        for t in trades:
            lines.append(f"\n   {t.symbol} {t.direction} ({t.timeframe})")
            lines.append(f"   Entry: ${t.entry_price:.2f} | Stop: ${t.stop_loss:.2f} | Target: ${t.target_1:.2f}")
            lines.append(f"   R:R: {t.rr_ratio:.1f}:1 | Signal: {t.signal}")
        
        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    # Initialize system
    system = ChartInputSystem()
    
    print("=" * 70)
    print("CHART INPUT SYSTEM - DEMO")
    print("=" * 70)
    
    # Single timeframe analysis
    print("\n📊 SINGLE TIMEFRAME (META 2HR):")
    result = system.analyze(
        symbol="META",
        price=619.28,
        vah=667.72,
        poc=660.40,
        val=647.22,
        vwap=619.63,
        rsi=33.58,
        timeframe="2HR"
    )
    print(system.print_result(result))
    
    # Multi-timeframe analysis
    print("\n📊 MULTI-TIMEFRAME ANALYSIS:")
    mtf_result = system.analyze_mtf("META", {
        "30min": {"price": 619.76, "vah": 666.77, "poc": 659.96, "val": 642.24, "vwap": 621.84, "rsi": 46.45},
        "1hr":   {"price": 619.99, "vah": 667.99, "poc": 658.22, "val": 648.44, "vwap": 621.84, "rsi": 44.93},
        "2hr":   {"price": 618.32, "vah": 670.71, "poc": 657.35, "val": 633.30, "vwap": 587.33, "rsi": 41.59},
        "4hr":   {"price": 613.44, "vah": 669.20, "poc": 655.80, "val": 615.59, "vwap": 607.75, "rsi": 63.34},
    })
    print(system.print_mtf_result(mtf_result))
    
    # Add alerts
    print("\n🚨 ADDING ALERTS:")
    system.add_alert("META", 615, "below", "SHORT", "Break of 4hr VAL support")
    system.add_alert("META", 647, "above", "LONG", "Reclaim 2hr VAL")
    print(system.print_alerts("META"))
    
    # Log trade setup
    print("\n📋 LOGGING TRADE:")
    trade = system.log_trade(
        symbol="META",
        timeframe="2HR",
        direction="LONG",
        entry=619,
        stop=613,
        target=647,
        signal="YELLOW",
        confidence=52,
        notes="Bounce setup from oversold RSI at VWAP"
    )
    print(system.print_pending_trades("META"))
    
    print("\n✅ System ready!")
