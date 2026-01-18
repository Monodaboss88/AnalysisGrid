"""
SEF Trading System - Extension Dashboard
Real-time display of extension status across levels and timeframes

Quick visual reference:
┌─────────────────────────────────────────────────────────────┐
│ EXTENSION STATUS: SPY                      2024-01-15 14:30 │
├─────────────────────────────────────────────────────────────┤
│ Level          │ Direction │ Candles │ Hours │ Trigger     │
├─────────────────────────────────────────────────────────────┤
│ VWAP Daily     │ ABOVE ▲   │ 3       │ 6.0   │ HIGH_PROB   │
│ POC Daily      │ ABOVE ▲   │ 2       │ 4.0   │ ALERT       │
│ VAH            │ ABOVE ▲   │ 3       │ 6.0   │ HIGH_PROB   │
│ VWAP Weekly    │ ABOVE ▲   │ 1       │ 2.0   │ WATCHING    │
├─────────────────────────────────────────────────────────────┤
│ HOTTEST: VWAP Daily - 75% snap-back probability            │
│ ACTION: Mean reversion SHORT setup forming                  │
└─────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from extension_predictor import (
    ExtensionDurationPredictor, 
    ExtensionStreak, 
    ExtensionAlert,
    TriggerLevel,
    ResolutionBias,
    CandleInExtension
)


@dataclass
class DashboardRow:
    """Single row in the extension dashboard"""
    level_name: str
    direction: str  # "ABOVE ▲" or "BELOW ▼"
    candle_count: int
    hours: float
    trigger: str
    snap_back_prob: float
    is_actionable: bool


class ExtensionDashboard:
    """
    Generates formatted extension status displays
    """
    
    # Trigger level display formatting
    TRIGGER_DISPLAY = {
        TriggerLevel.NONE: ("·", ""),
        TriggerLevel.WATCHING: ("👀", "WATCHING"),
        TriggerLevel.ALERT: ("⚠️", "ALERT"),
        TriggerLevel.HIGH_PROB: ("🔥", "HIGH_PROB"),
        TriggerLevel.EXTREME: ("💥", "EXTREME"),
    }
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.predictor = ExtensionDurationPredictor(candle_minutes=120)
        self._last_update: Optional[datetime] = None
        self._alerts: List[ExtensionAlert] = []
    
    def update(self, candle: CandleInExtension) -> List[ExtensionAlert]:
        """Update with new candle and return any alerts"""
        self._last_update = candle.timestamp
        self._alerts = self.predictor.update(candle)
        
        # Add symbol to alerts
        for alert in self._alerts:
            alert.symbol = self.symbol
        
        return self._alerts
    
    def get_status_text(self) -> str:
        """Generate formatted status text"""
        streaks = self.predictor.get_active_streaks()
        
        lines = []
        lines.append(self._header())
        lines.append(self._separator())
        lines.append(self._column_headers())
        lines.append(self._separator())
        
        if not streaks:
            lines.append("│ No active extensions - price at fair value".ljust(60) + "│")
        else:
            for streak in sorted(streaks.values(), key=lambda s: s.streak_count, reverse=True):
                lines.append(self._format_streak_row(streak))
        
        lines.append(self._separator())
        lines.append(self._summary())
        lines.append(self._footer())
        
        return "\n".join(lines)
    
    def get_compact_status(self) -> str:
        """One-line compact status"""
        streaks = self.predictor.get_active_streaks()
        
        if not streaks:
            return f"{self.symbol}: At value ✓"
        
        # Find hottest
        hottest = max(streaks.values(), key=lambda s: s.streak_count)
        emoji, trigger_name = self.TRIGGER_DISPLAY.get(hottest.trigger, ("", ""))
        
        direction = "▲" if hottest.direction == "above" else "▼"
        
        return (
            f"{self.symbol}: {hottest.level_name} {direction} "
            f"{hottest.streak_count}c/{hottest.hours_extended:.1f}h "
            f"{emoji} {trigger_name} "
            f"[{hottest.snap_back_probability:.0%}]"
        )
    
    def get_alert_text(self) -> Optional[str]:
        """Get formatted alert text if any alerts active"""
        if not self._alerts:
            return None
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"🚨 EXTENSION ALERT: {self.symbol}")
        lines.append("=" * 60)
        
        for alert in self._alerts:
            direction_arrow = "▲" if alert.direction == "above" else "▼"
            
            lines.append(f"Level: {alert.level_name} {direction_arrow}")
            lines.append(f"Duration: {alert.candle_count} candles ({alert.hours_extended:.1f} hours)")
            lines.append(f"Extension: {alert.extension_atr:.2f} ATR from level")
            lines.append(f"")
            lines.append(f"Snap-back probability: {alert.snap_back_probability:.0%}")
            lines.append(f"Target: ${alert.snap_back_target:.2f}")
            lines.append(f"Stop: ${alert.stop_loss:.2f}")
            lines.append(f"Risk/Reward: {alert.risk_reward:.2f}")
            lines.append(f"")
            lines.append(f"Quality Score: {alert.quality_score:.0f}/100")
            lines.append("-" * 40)
        
        # Recommendation
        best_alert = max(self._alerts, key=lambda a: a.quality_score)
        if best_alert.quality_score >= 70:
            action = "SHORT" if best_alert.direction == "above" else "LONG"
            lines.append(f"")
            lines.append(f"💡 RECOMMENDATION: {action} setup - HIGH quality")
            lines.append(f"   Entry: ${best_alert.current_price:.2f}")
            lines.append(f"   Target: ${best_alert.snap_back_target:.2f}")
            lines.append(f"   Stop: ${best_alert.stop_loss:.2f}")
        elif best_alert.quality_score >= 50:
            lines.append(f"")
            lines.append(f"👀 WATCHING: Setup developing, wait for confirmation")
        
        return "\n".join(lines)
    
    def _header(self) -> str:
        time_str = self._last_update.strftime("%Y-%m-%d %H:%M") if self._last_update else "---"
        header = f"│ EXTENSION STATUS: {self.symbol}".ljust(42) + f"{time_str} │"
        return "┌" + "─" * 60 + "┐\n" + header
    
    def _separator(self) -> str:
        return "├" + "─" * 60 + "┤"
    
    def _footer(self) -> str:
        return "└" + "─" * 60 + "┘"
    
    def _column_headers(self) -> str:
        return "│ Level          │ Dir   │ Candles │ Hours │ Status      │"
    
    def _format_streak_row(self, streak: ExtensionStreak) -> str:
        direction = "▲ UP  " if streak.direction == "above" else "▼ DOWN"
        emoji, trigger_name = self.TRIGGER_DISPLAY.get(streak.trigger, ("", "---"))
        
        level = streak.level_name[:14].ljust(14)
        candles = str(streak.streak_count).center(7)
        hours = f"{streak.hours_extended:.1f}".center(5)
        status = f"{emoji} {trigger_name}".ljust(11)
        
        return f"│ {level} │ {direction} │ {candles} │ {hours} │ {status} │"
    
    def _summary(self) -> str:
        streaks = self.predictor.get_active_streaks()
        
        if not streaks:
            return "│ ACTION: None - wait for extension to develop".ljust(61) + "│"
        
        # Find hottest setup
        actionable = [s for s in streaks.values() if s.is_actionable]
        
        if actionable:
            hottest = max(actionable, key=lambda s: s.snap_back_probability)
            action = "SHORT" if hottest.direction == "above" else "LONG"
            return f"│ ACTION: {action} setup @ {hottest.level_name} [{hottest.snap_back_probability:.0%} snap-back]".ljust(61) + "│"
        else:
            watching = max(streaks.values(), key=lambda s: s.streak_count)
            return f"│ WATCHING: {watching.level_name} ({watching.streak_count}c) - not yet actionable".ljust(61) + "│"


class MultiSymbolDashboard:
    """
    Dashboard for multiple symbols
    """
    
    def __init__(self, symbols: List[str]):
        self.dashboards = {sym: ExtensionDashboard(sym) for sym in symbols}
    
    def update(self, symbol: str, candle: CandleInExtension) -> List[ExtensionAlert]:
        """Update specific symbol"""
        if symbol not in self.dashboards:
            self.dashboards[symbol] = ExtensionDashboard(symbol)
        
        return self.dashboards[symbol].update(candle)
    
    def get_all_status(self) -> str:
        """Get compact status for all symbols"""
        lines = ["=" * 60]
        lines.append("EXTENSION MONITOR")
        lines.append("=" * 60)
        
        for sym, dash in self.dashboards.items():
            lines.append(dash.get_compact_status())
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def get_actionable(self) -> List[str]:
        """Get symbols with actionable setups"""
        actionable = []
        for sym, dash in self.dashboards.items():
            if dash.predictor.get_actionable_setups():
                actionable.append(sym)
        return actionable
    
    def get_alerts(self) -> Dict[str, str]:
        """Get alert text for all symbols with alerts"""
        alerts = {}
        for sym, dash in self.dashboards.items():
            alert_text = dash.get_alert_text()
            if alert_text:
                alerts[sym] = alert_text
        return alerts


# ============ Quick Reference Card ============

QUICK_REFERENCE = """
┌──────────────────────────────────────────────────────────────┐
│           EXTENSION DURATION QUICK REFERENCE                  │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  CANDLES   HOURS   TRIGGER      SNAP-BACK     ACTION         │
│  ───────   ─────   ──────────   ──────────    ──────         │
│     1       2h     WATCHING       45%         Wait           │
│     2       4h     ALERT          55%         Prepare        │
│     3       6h     HIGH_PROB      65%         Look for entry │
│     4+      8h+    EXTREME        75%+        High conviction│
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  CONFIRMATION SIGNALS:                                        │
│  ✓ Rejection candle (long wick toward value)                 │
│  ✓ Declining volume in extension                             │
│  ✓ Multiple levels aligned (VWAP + POC + VAH all extended)   │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  TRADE SETUP (Mean Reversion):                               │
│                                                               │
│  IF price ABOVE VWAP for 3+ candles:                         │
│     Entry: SHORT on rejection candle                          │
│     Target: VWAP (snap-back)                                  │
│     Stop: Above high of extension                             │
│                                                               │
│  IF price BELOW VWAP for 3+ candles:                         │
│     Entry: LONG on rejection candle                           │
│     Target: VWAP (snap-back)                                  │
│     Stop: Below low of extension                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
"""


def print_quick_reference():
    print(QUICK_REFERENCE)


# ============ Example ============

def demo():
    """Demo the dashboard"""
    from datetime import timedelta
    
    # Create dashboard
    dashboard = ExtensionDashboard("SPY")
    
    # Reference levels
    vwap = 585.00
    poc = 584.50
    vah = 586.00
    val = 583.00
    atr = 2.50
    
    base_time = datetime(2024, 1, 15, 9, 30)
    
    print("=" * 60)
    print("EXTENSION DURATION PREDICTOR DEMO")
    print("=" * 60)
    print()
    
    # Simulate 4 candles in extension
    candles = [
        # Candle 1: Breaks above VWAP
        (0, 585.20, 586.80, 585.00, 586.50, 1000000),
        # Candle 2: Continues higher
        (2, 586.50, 587.50, 586.00, 587.20, 900000),
        # Candle 3: Extended but rejection forming
        (4, 587.20, 588.00, 586.20, 586.80, 800000),
        # Candle 4: More rejection
        (6, 586.80, 587.20, 585.80, 586.00, 700000),
    ]
    
    for hours, o, h, l, c, v in candles:
        candle = CandleInExtension(
            timestamp=base_time + timedelta(hours=hours),
            open=o, high=h, low=l, close=c, volume=v,
            vwap=vwap, poc=poc, vah=vah, val=val, atr=atr
        )
        
        alerts = dashboard.update(candle)
        
        print(f"\n--- After {hours}h ({(hours//2)+1} candles) ---")
        print(dashboard.get_compact_status())
        
        if alerts:
            print()
            print(dashboard.get_alert_text())
    
    print("\n")
    print(dashboard.get_status_text())
    print("\n")
    print_quick_reference()


if __name__ == "__main__":
    demo()
