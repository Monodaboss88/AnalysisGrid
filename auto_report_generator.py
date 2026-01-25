"""
Auto Report Generator
=====================
Automatically generates analysis reports from scanner results.
These reports feed the AI knowledge base for continuous learning.
Now stores reports in Firestore for persistence across deploys.

Author: Rob's Trading Systems
Version: 1.1.0
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json

# Try to import Firestore
try:
    import firebase_admin
    from firebase_admin import firestore
    firestore_available = True
except ImportError:
    firestore_available = False


class ReportStore:
    """
    Stores reports in Firestore for persistence.
    Falls back to local files if Firestore unavailable.
    """
    
    def __init__(self):
        self.db = None
        self._init_firestore()
    
    def _init_firestore(self):
        """Initialize Firestore connection"""
        if not firestore_available:
            return
        
        try:
            # Get existing Firebase app
            app = firebase_admin.get_app()
            self.db = firestore.client()
        except Exception as e:
            print(f"ReportStore: Firestore not available: {e}")
    
    def save_report(self, symbol: str, date_str: str, content: str, report_type: str = "analysis") -> str:
        """Save report to Firestore"""
        if self.db:
            try:
                doc_id = f"{symbol}_{date_str}_{datetime.now().strftime('%H%M%S')}"
                self.db.collection('reports').document(doc_id).set({
                    'symbol': symbol.upper(),
                    'date': date_str,
                    'content': content,
                    'type': report_type,
                    'created_at': datetime.now().isoformat()
                })
                return doc_id
            except Exception as e:
                print(f"Error saving report to Firestore: {e}")
        
        # Fallback to local file
        return self._save_local(symbol, date_str, content)
    
    def _save_local(self, symbol: str, date_str: str, content: str) -> str:
        """Fallback: save to local file"""
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{symbol}_Analysis_{date_str}.md"
        filepath = reports_dir / filename
        
        if filepath.exists():
            time_suffix = datetime.now().strftime('%H%M%S')
            filename = f"{symbol}_Analysis_{date_str}_{time_suffix}.md"
            filepath = reports_dir / filename
        
        filepath.write_text(content, encoding='utf-8')
        return str(filepath)
    
    def get_reports(self, symbol: str = None, limit: int = 50) -> List[Dict]:
        """Get reports from Firestore"""
        if not self.db:
            return []
        
        try:
            query = self.db.collection('reports')
            if symbol:
                query = query.where('symbol', '==', symbol.upper())
            query = query.order_by('created_at', direction=firestore.Query.DESCENDING).limit(limit)
            
            docs = query.stream()
            return [{'id': doc.id, **doc.to_dict()} for doc in docs]
        except Exception as e:
            print(f"Error getting reports: {e}")
            return []
    
    def get_report_content(self, doc_id: str) -> Optional[str]:
        """Get a specific report's content"""
        if not self.db:
            return None
        
        try:
            doc = self.db.collection('reports').document(doc_id).get()
            if doc.exists:
                return doc.to_dict().get('content')
        except Exception as e:
            print(f"Error getting report: {e}")
        return None


class AutoReportGenerator:
    """
    Generates markdown analysis reports from scanner data.
    Reports are saved to Firestore for persistence and AI learning.
    """
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.store = ReportStore()
    
    def generate_report(self, scanner_data: Dict, trade_plan: Dict = None) -> str:
        """
        Generate a full analysis report from scanner data.
        
        Args:
            scanner_data: Dict with symbol, price, levels, scores, etc.
            trade_plan: Optional trade plan dict from rule engine
            
        Returns:
            Path to the generated report file
        """
        symbol = scanner_data.get('symbol', 'UNKNOWN')
        date_str = datetime.now().strftime('%Y-%m-%d')
        time_str = datetime.now().strftime('%H:%M:%S')
        
        # Extract data with defaults
        price = scanner_data.get('current_price') or scanner_data.get('price', 0)
        vah = scanner_data.get('vah', price * 1.02)
        poc = scanner_data.get('poc', price)
        val = scanner_data.get('val', price * 0.98)
        vwap = scanner_data.get('vwap', price)
        rsi = scanner_data.get('rsi', 50)
        rvol = scanner_data.get('rvol', 1.0)
        bull_score = scanner_data.get('bull_score', 0)
        bear_score = scanner_data.get('bear_score', 0)
        confidence = scanner_data.get('confidence', 50)
        signal = scanner_data.get('signal', 'UNKNOWN')
        direction = scanner_data.get('direction', 'neutral')
        notes = scanner_data.get('notes', [])
        timeframe = scanner_data.get('timeframe', '1HR')
        
        # Determine signal emoji and bias
        if bull_score > bear_score + 10:
            signal_emoji = "🟢"
            bias = "Bullish"
        elif bear_score > bull_score + 10:
            signal_emoji = "🔴"
            bias = "Bearish"
        else:
            signal_emoji = "🟡"
            bias = "Neutral/Mixed"
        
        # Build the report
        report = f"""# {symbol} Analysis Report
**Generated:** {date_str} {time_str} | **Timeframe:** {timeframe} | **Auto-Generated**

---

## Executive Summary

**Current Price:** ${price:.2f}
**Signal:** {signal_emoji} {signal}
**Bias:** {bias}
**Confidence:** {confidence:.0f}%

{self._generate_summary(scanner_data, trade_plan)}

---

## Technical Analysis

### Current Snapshot
| Metric | Value |
|--------|-------|
| Price | ${price:.2f} |
| RSI | {rsi:.1f} |
| RVOL | {rvol:.1f}x |
| Bull Score | {bull_score:.0f} |
| Bear Score | {bear_score:.0f} |

### Volume Profile Levels ({timeframe})
| Level | Price | Current Position |
|-------|-------|------------------|
| VAH | ${vah:.2f} | {"📍 AT" if abs(price - vah) / price < 0.005 else "Above ↑" if price > vah else "Below ↓"} |
| POC | ${poc:.2f} | {"📍 AT" if abs(price - poc) / price < 0.005 else "Above ↑" if price > poc else "Below ↓"} |
| VAL | ${val:.2f} | {"📍 AT" if abs(price - val) / price < 0.005 else "Above ↑" if price > val else "Below ↓"} |
| VWAP | ${vwap:.2f} | {"📍 AT" if abs(price - vwap) / price < 0.005 else "Above ↑" if price > vwap else "Below ↓"} |

### Price Position Analysis
{self._analyze_position(price, vah, poc, val, vwap)}

---

## Signal Analysis

### Scoring Breakdown
| Factor | Bull | Bear | Notes |
|--------|------|------|-------|
{self._generate_scoring_table(scanner_data)}

### Scanner Notes
{self._format_notes(notes)}

---

## Trade Setup

{self._generate_trade_setup(scanner_data, trade_plan)}

---

## Risk Factors

{self._identify_risks(scanner_data)}

---

## Key Levels to Watch

| Level | Price | Significance |
|-------|-------|--------------|
| Resistance 1 | ${vah:.2f} | VAH - Value Area High |
| Pivot | ${poc:.2f} | POC - Point of Control |
| Support 1 | ${val:.2f} | VAL - Value Area Low |
| VWAP | ${vwap:.2f} | Intraday anchor |

---

## Conclusion

{self._generate_conclusion(scanner_data, trade_plan)}

---

*Report auto-generated by Scanner System*
*Timeframe: {timeframe} | Generated: {datetime.now().isoformat()}*
"""
        
        # Save the report to Firestore (with local fallback)
        doc_id = self.store.save_report(symbol, date_str, report, "analysis")
        
        return doc_id
    
    def _generate_summary(self, data: Dict, plan: Dict = None) -> str:
        """Generate executive summary text"""
        price = data.get('current_price') or data.get('price', 0)
        bull = data.get('bull_score', 0)
        bear = data.get('bear_score', 0)
        rsi = data.get('rsi', 50)
        
        parts = []
        
        if bull > bear + 15:
            parts.append(f"Strong bullish bias with bull score {bull:.0f} vs bear {bear:.0f}.")
        elif bear > bull + 15:
            parts.append(f"Strong bearish bias with bear score {bear:.0f} vs bull {bull:.0f}.")
        else:
            parts.append(f"Mixed signals with bull {bull:.0f} vs bear {bear:.0f} - no clear edge.")
        
        if rsi > 70:
            parts.append("RSI overbought, potential pullback risk.")
        elif rsi < 30:
            parts.append("RSI oversold, potential bounce setup.")
        
        if plan:
            if plan.get('direction') == 'LONG':
                parts.append(f"Trade plan: LONG with stop at ${plan.get('stop_loss', 0):.2f}.")
            elif plan.get('direction') == 'SHORT':
                parts.append(f"Trade plan: SHORT with stop at ${plan.get('stop_loss', 0):.2f}.")
            else:
                parts.append("No trade recommended per rule engine.")
        
        return " ".join(parts)
    
    def _analyze_position(self, price: float, vah: float, poc: float, val: float, vwap: float) -> str:
        """Analyze current price position"""
        lines = []
        
        if price > vah:
            lines.append("- **Extended Above Value**: Price trading above VAH indicates strong momentum but also extension risk.")
        elif price > poc:
            lines.append("- **Above POC in Value**: Price in upper value area, bullish positioning.")
        elif price > val:
            lines.append("- **Below POC in Value**: Price in lower value area, bearish positioning.")
        else:
            lines.append("- **Below Value Area**: Price rejected from value, bearish or potential bounce setup.")
        
        if price > vwap:
            lines.append("- **Above VWAP**: Intraday buyers in control.")
        else:
            lines.append("- **Below VWAP**: Intraday sellers in control.")
        
        return "\n".join(lines)
    
    def _generate_scoring_table(self, data: Dict) -> str:
        """Generate scoring breakdown table rows"""
        rows = []
        price = data.get('current_price') or data.get('price', 0)
        poc = data.get('poc', price)
        vwap = data.get('vwap', price)
        rsi = data.get('rsi', 50)
        
        # Position vs POC
        if price > poc:
            rows.append(f"| Price vs POC | +20 | - | Above POC (bullish) |")
        else:
            rows.append(f"| Price vs POC | - | +20 | Below POC (bearish) |")
        
        # VWAP
        if price > vwap:
            rows.append(f"| VWAP Position | +15 | - | Above VWAP |")
        else:
            rows.append(f"| VWAP Position | - | +15 | Below VWAP |")
        
        # RSI
        if rsi > 70:
            rows.append(f"| RSI | - | +10 | Overbought ({rsi:.0f}) |")
        elif rsi < 30:
            rows.append(f"| RSI | +10 | - | Oversold ({rsi:.0f}) |")
        elif rsi > 55:
            rows.append(f"| RSI | +10 | - | Bullish ({rsi:.0f}) |")
        elif rsi < 45:
            rows.append(f"| RSI | - | +10 | Bearish ({rsi:.0f}) |")
        else:
            rows.append(f"| RSI | - | - | Neutral ({rsi:.0f}) |")
        
        bull = data.get('bull_score', 0)
        bear = data.get('bear_score', 0)
        rows.append(f"| **TOTAL** | **{bull:.0f}** | **{bear:.0f}** | Gap: {abs(bull-bear):.0f} |")
        
        return "\n".join(rows)
    
    def _format_notes(self, notes: List[str]) -> str:
        """Format scanner notes as bullet list"""
        if not notes:
            return "- No specific notes from scanner"
        return "\n".join([f"- {note}" for note in notes])
    
    def _generate_trade_setup(self, data: Dict, plan: Dict = None) -> str:
        """Generate trade setup section"""
        if not plan or plan.get('direction') == 'NO_TRADE':
            return """### No Active Setup

Current conditions do not meet entry criteria. Wait for:
- Clearer directional bias (score gap > 15)
- Better risk/reward at key levels
- Volume confirmation
"""
        
        direction = plan.get('direction', 'UNKNOWN')
        emoji = "🟢" if direction == 'LONG' else "🔴"
        
        return f"""### {emoji} {direction} Setup

| Parameter | Value |
|-----------|-------|
| Entry Zone | ${plan.get('entry_zone_low', 0):.2f} - ${plan.get('entry_zone_high', 0):.2f} |
| Stop Loss | ${plan.get('stop_loss', 0):.2f} |
| Target 1 | ${plan.get('target_1', 0):.2f} ({plan.get('risk_reward_t1', 0):.1f}R) |
| Target 2 | ${plan.get('target_2', 0):.2f} ({plan.get('risk_reward_t2', 0):.1f}R) |
| Risk/Share | ${plan.get('risk_per_share', 0):.2f} |
| Position Size | {plan.get('position_size_pct', 0) * 100:.0f}% |

**Entry Reasons:**
{chr(10).join(['- ' + r for r in plan.get('entry_reasons', [])])}

**Watch For:**
{chr(10).join(['- ' + c for c in plan.get('caution_flags', [])]) if plan.get('caution_flags') else '- No major concerns'}

**Invalidation:**
{plan.get('invalidation', 'Stop loss hit')}
"""
    
    def _identify_risks(self, data: Dict) -> str:
        """Identify potential risks"""
        risks = []
        
        rsi = data.get('rsi', 50)
        rvol = data.get('rvol', 1.0)
        bull = data.get('bull_score', 0)
        bear = data.get('bear_score', 0)
        
        if rsi > 70:
            risks.append("1. **Overbought RSI** - Pullback risk elevated")
        if rsi < 30:
            risks.append("1. **Oversold RSI** - Dead cat bounce risk")
        
        if abs(bull - bear) < 15:
            risks.append("2. **Low Conviction** - Bull/bear scores too close")
        
        if rvol < 0.8:
            risks.append("3. **Low Volume** - Move may lack conviction")
        
        if rvol > 2.5:
            risks.append("3. **Extreme Volume** - Potential exhaustion")
        
        # Add generic risks if not many specific ones
        if len(risks) < 2:
            risks.append("- Monitor overall market conditions (SPY/QQQ)")
            risks.append("- Check for upcoming news/earnings")
        
        return "\n".join(risks) if risks else "- No major risks identified"
    
    def _generate_conclusion(self, data: Dict, plan: Dict = None) -> str:
        """Generate conclusion section"""
        bull = data.get('bull_score', 0)
        bear = data.get('bear_score', 0)
        symbol = data.get('symbol', 'UNKNOWN')
        
        if plan and plan.get('direction') != 'NO_TRADE':
            direction = plan.get('direction')
            stop = plan.get('stop_loss', 0)
            t1 = plan.get('target_1', 0)
            
            return f"""{symbol} presents a **{direction}** opportunity based on current analysis.

**Recommended Action:**
1. Enter in the defined entry zone
2. Set stop loss at ${stop:.2f}
3. Take partial profits at T1 (${t1:.2f})
4. Trail stop on remaining position

**Key Level to Watch:** {"Break above VAH for continuation" if direction == 'LONG' else "Break below VAL for continuation"}
"""
        else:
            return f"""{symbol} currently shows **MIXED** signals - no clear edge.

**Recommended Action:**
1. Wait for clearer directional bias
2. Watch for price reaction at key levels
3. Look for volume confirmation on breakout/breakdown

**Bullish above:** VAH breakout with volume
**Bearish below:** VAL breakdown with volume
"""
    
    def generate_batch_summary(self, results: List[Dict]) -> str:
        """
        Generate a summary report of multiple scanner results.
        
        Args:
            results: List of scanner result dicts
            
        Returns:
            Path to summary report
        """
        date_str = datetime.now().strftime('%Y-%m-%d')
        time_str = datetime.now().strftime('%H:%M:%S')
        
        # Sort by score
        sorted_results = sorted(
            results, 
            key=lambda x: max(x.get('bull_score', 0), x.get('bear_score', 0)),
            reverse=True
        )
        
        # Categorize
        strong_longs = [r for r in sorted_results if r.get('bull_score', 0) > 65 and r.get('bull_score', 0) > r.get('bear_score', 0) + 10]
        strong_shorts = [r for r in sorted_results if r.get('bear_score', 0) > 65 and r.get('bear_score', 0) > r.get('bull_score', 0) + 10]
        watchlist = [r for r in sorted_results if r not in strong_longs and r not in strong_shorts and max(r.get('bull_score', 0), r.get('bear_score', 0)) > 50]
        
        report = f"""# Scanner Summary Report
**Generated:** {date_str} {time_str}
**Symbols Scanned:** {len(results)}

---

## Top Opportunities

### 🟢 Strong Longs ({len(strong_longs)})
| Symbol | Score | Price | Signal |
|--------|-------|-------|--------|
"""
        for r in strong_longs[:10]:
            report += f"| {r.get('symbol')} | {r.get('bull_score', 0):.0f} | ${r.get('current_price', 0):.2f} | {r.get('signal', '')} |\n"
        
        report += f"""
### 🔴 Strong Shorts ({len(strong_shorts)})
| Symbol | Score | Price | Signal |
|--------|-------|-------|--------|
"""
        for r in strong_shorts[:10]:
            report += f"| {r.get('symbol')} | {r.get('bear_score', 0):.0f} | ${r.get('current_price', 0):.2f} | {r.get('signal', '')} |\n"
        
        report += f"""
### 🟡 Watchlist ({len(watchlist)})
| Symbol | Bull | Bear | Notes |
|--------|------|------|-------|
"""
        for r in watchlist[:10]:
            report += f"| {r.get('symbol')} | {r.get('bull_score', 0):.0f} | {r.get('bear_score', 0):.0f} | Mixed signals |\n"
        
        report += f"""
---

## Statistics

- Total Scanned: {len(results)}
- Strong Longs: {len(strong_longs)}
- Strong Shorts: {len(strong_shorts)}
- Watchlist: {len(watchlist)}
- No Setup: {len(results) - len(strong_longs) - len(strong_shorts) - len(watchlist)}

---

*Auto-generated summary report*
"""
        
        # Save the summary report to Firestore
        doc_id = self.store.save_report("SUMMARY", date_str, report, "summary")
        
        return doc_id


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def auto_generate_report(scanner_data: Dict, trade_plan: Dict = None) -> str:
    """Quick function to generate a report"""
    generator = AutoReportGenerator()
    return generator.generate_report(scanner_data, trade_plan)


def generate_scan_summary(results: List[Dict]) -> str:
    """Generate summary report for batch scan"""
    generator = AutoReportGenerator()
    return generator.generate_batch_summary(results)


if __name__ == "__main__":
    # Test
    test_data = {
        'symbol': 'AAPL',
        'current_price': 185.50,
        'vah': 188.00,
        'poc': 186.00,
        'val': 183.50,
        'vwap': 185.00,
        'bull_score': 72,
        'bear_score': 35,
        'rsi': 58,
        'rvol': 1.8,
        'confidence': 72,
        'signal': 'BULLISH',
        'direction': 'long',
        'notes': ['Above POC', 'Above VWAP', 'RSI bullish']
    }
    
    test_plan = {
        'direction': 'LONG',
        'entry_zone_low': 185.00,
        'entry_zone_high': 186.00,
        'stop_loss': 183.00,
        'target_1': 188.00,
        'target_2': 190.00,
        'risk_per_share': 2.50,
        'risk_reward_t1': 1.2,
        'risk_reward_t2': 2.0,
        'position_size_pct': 1.0,
        'entry_reasons': ['Bull score 72 > Bear 35', 'Above POC', 'Above VWAP'],
        'caution_flags': [],
        'invalidation': 'Close below $183.00'
    }
    
    path = auto_generate_report(test_data, test_plan)
    print(f"✅ Report generated: {path}")
