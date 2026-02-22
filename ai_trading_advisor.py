"""
AI Trading Advisor - Hedge Fund Grade Intelligence Layer
=========================================================
Integrates with MTF Auction Scanner to provide institutional-quality
AI-powered analysis, regime detection, and trade journaling.

This module does NOT replace the quantitative core - it ENHANCES it.
The LLM layer handles:
- Contextual interpretation of scanner signals
- News/sentiment synthesis
- Pattern recognition from trade history
- Regime-aware strategy selection
- Natural language trade journaling

Author: Rob's Trading Systems
Version: 1.0.0
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import sqlite3
from pathlib import Path

# Optional imports for enhanced functionality
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification for strategy selection"""
    TRENDING_BULL = "TRENDING_BULL"       # Strong uptrend, momentum strategies
    TRENDING_BEAR = "TRENDING_BEAR"       # Strong downtrend, momentum strategies
    RANGE_BOUND = "RANGE_BOUND"           # Chop, mean reversion strategies
    HIGH_VOLATILITY = "HIGH_VOLATILITY"   # VIX spike, reduce size / widen stops
    LOW_VOLATILITY = "LOW_VOLATILITY"     # Compression, breakout strategies
    TRANSITION = "TRANSITION"             # Regime change in progress, caution
    
    @property
    def strategy_notes(self) -> str:
        return {
            "TRENDING_BULL": "Favor longs, trail stops, add on pullbacks to VWAP/VAL",
            "TRENDING_BEAR": "Favor shorts, trail stops, add on rallies to VWAP/VAH",
            "RANGE_BOUND": "Fade extremes, tight stops, quick profits at POC",
            "HIGH_VOLATILITY": "Reduce size 50%, widen stops 1.5x, avoid overnight",
            "LOW_VOLATILITY": "Watch for breakout, size up on confirmation, tight initial stops",
            "TRANSITION": "Reduce exposure, wait for clarity, don't chase"
        }[self.value]


@dataclass
class RegimeAnalysis:
    """Complete regime analysis result"""
    regime: MarketRegime
    confidence: float
    vix_level: float
    trend_strength: float  # -100 to +100
    range_bound_score: float  # 0 to 100
    days_in_regime: int
    notes: List[str] = field(default_factory=list)


class RegimeDetector:
    """
    Detects current market regime using multiple factors.
    
    This runs on the INDEX level (SPY/QQQ) to set the market context
    for individual stock analysis.
    """
    
    def __init__(self):
        self.vix_thresholds = {
            'extreme_low': 12,
            'low': 15,
            'normal': 20,
            'elevated': 25,
            'high': 30,
            'extreme': 40
        }
    
    def analyze(self, 
                spy_data: Dict,
                vix_level: float,
                breadth_data: Optional[Dict] = None) -> RegimeAnalysis:
        """
        Analyze market regime.
        
        Args:
            spy_data: Dict with keys 'price', 'sma_20', 'sma_50', 'sma_200', 
                     'atr_20', 'high_20', 'low_20'
            vix_level: Current VIX value
            breadth_data: Optional market breadth (advance/decline, new highs/lows)
        
        Returns:
            RegimeAnalysis object
        """
        notes = []
        
        # Trend analysis
        price = spy_data.get('price', 0)
        sma_20 = spy_data.get('sma_20', price)
        sma_50 = spy_data.get('sma_50', price)
        sma_200 = spy_data.get('sma_200', price)
        
        # Trend strength: distance from moving averages
        trend_score = 0
        if price > sma_20: trend_score += 25
        if price > sma_50: trend_score += 25
        if price > sma_200: trend_score += 25
        if sma_20 > sma_50: trend_score += 12.5
        if sma_50 > sma_200: trend_score += 12.5
        
        # Normalize to -100 to +100
        trend_strength = (trend_score - 50) * 2
        
        # Range analysis: how tight is the recent range?
        high_20 = spy_data.get('high_20', price * 1.05)
        low_20 = spy_data.get('low_20', price * 0.95)
        atr_20 = spy_data.get('atr_20', (high_20 - low_20) / 20)
        
        range_pct = (high_20 - low_20) / price * 100 if price > 0 else 5
        range_bound_score = max(0, 100 - range_pct * 10)  # Tighter range = higher score
        
        # VIX analysis
        if vix_level >= self.vix_thresholds['extreme']:
            vol_regime = 'EXTREME'
            notes.append(f"⚠️ VIX at {vix_level:.1f} - extreme volatility")
        elif vix_level >= self.vix_thresholds['high']:
            vol_regime = 'HIGH'
            notes.append(f"VIX elevated at {vix_level:.1f}")
        elif vix_level <= self.vix_thresholds['extreme_low']:
            vol_regime = 'COMPRESSED'
            notes.append(f"VIX compressed at {vix_level:.1f} - breakout watch")
        else:
            vol_regime = 'NORMAL'
        
        # Determine regime
        if vol_regime in ['EXTREME', 'HIGH']:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = 0.85
        elif vol_regime == 'COMPRESSED' and range_bound_score > 70:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = 0.80
            notes.append("Coiled market - expect expansion")
        elif trend_strength > 50:
            regime = MarketRegime.TRENDING_BULL
            confidence = min(0.90, 0.5 + abs(trend_strength) / 200)
            notes.append(f"Bullish structure: price > 20/50/200 SMAs")
        elif trend_strength < -50:
            regime = MarketRegime.TRENDING_BEAR
            confidence = min(0.90, 0.5 + abs(trend_strength) / 200)
            notes.append(f"Bearish structure: price < 20/50/200 SMAs")
        elif range_bound_score > 60:
            regime = MarketRegime.RANGE_BOUND
            confidence = 0.70
            notes.append(f"Range-bound: {range_pct:.1f}% range over 20 days")
        else:
            regime = MarketRegime.TRANSITION
            confidence = 0.50
            notes.append("Mixed signals - regime transition likely")
        
        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            vix_level=vix_level,
            trend_strength=trend_strength,
            range_bound_score=range_bound_score,
            days_in_regime=0,  # Would need historical tracking
            notes=notes
        )


# =============================================================================
# AI COMMENTARY ENGINE
# =============================================================================

@dataclass
class TradeContext:
    """Full context for AI analysis"""
    symbol: str
    scanner_result: Dict           # Output from MTF scanner
    regime: RegimeAnalysis         # Current market regime
    recent_trades: List[Dict]      # Last 10 trades on this symbol
    win_rate: float               # Historical win rate on similar setups
    avg_r_multiple: float         # Average R on similar setups
    news_headlines: List[str]     # Recent news if available
    earnings_days: int            # Days until earnings (-1 if passed, 0 if today)


class AICommentaryEngine:
    """
    Generates institutional-quality trade analysis using LLMs.
    
    Unlike basic GPT integration, this engine:
    1. Maintains context across analyses
    2. References your historical trade patterns
    3. Adapts to current market regime
    4. Provides risk-adjusted recommendations
    """
    
    def __init__(self, provider: str = 'anthropic', model: str = None):
        """
        Initialize AI engine.
        
        Args:
            provider: 'anthropic' or 'openai'
            model: Specific model to use (defaults to best for provider)
        """
        self.provider = provider
        
        if provider == 'anthropic' and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
            self.model = model or 'claude-sonnet-4-20250514'
        elif provider == 'openai' and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
            self.model = model or 'gpt-4o'
        else:
            self.client = None
            self.model = None
    
    def _build_system_prompt(self, regime: RegimeAnalysis) -> str:
        """Build context-aware system prompt"""
        return f"""You are Rob's AI trading advisor, specializing in auction market theory and volume profile analysis. You analyze setups for 3-5 day swing trades.

CURRENT MARKET REGIME: {regime.regime.value}
REGIME STRATEGY: {regime.regime.strategy_notes}
VIX: {regime.vix_level:.1f}
CONFIDENCE IN REGIME: {regime.confidence:.0%}

YOUR ROLE:
1. Interpret the scanner's quantitative signals
2. Identify what could INVALIDATE the setup
3. Suggest specific price levels to watch
4. Adjust recommendations based on the market regime
5. Reference historical patterns when relevant

RESPONSE FORMAT:
- Lead with the actionable insight (1 sentence)
- Key levels: Entry zone, Stop, Targets
- What would CHANGE your mind
- Position sizing note based on regime

Keep responses concise (150 words max). Use trader language. Be direct about uncertainty."""
    
    def analyze(self, context: TradeContext) -> Dict:
        """
        Generate comprehensive AI analysis.
        
        Returns:
            Dict with 'commentary', 'key_levels', 'invalidation', 'sizing_note'
        """
        if self.client is None:
            return {
                'commentary': 'AI engine not configured',
                'key_levels': {},
                'invalidation': '',
                'sizing_note': ''
            }
        
        # Build the analysis prompt
        scanner = context.scanner_result
        regime = context.regime
        
        user_prompt = f"""ANALYZE THIS SETUP:

SYMBOL: {context.symbol}
SCANNER SIGNAL: {scanner.get('signal', 'UNKNOWN')} ({scanner.get('confidence', 0):.0f}% confidence)
CONFLUENCE: {scanner.get('confluence', 0):.0f}%

SCORES:
- Bull Score: {scanner.get('bull_score', 0):.1f}
- Bear Score: {scanner.get('bear_score', 0):.1f}

PRICE STRUCTURE:
- Current: ${scanner.get('price', 0):.2f}
- VAH: ${scanner.get('vah', 0):.2f}
- POC: ${scanner.get('poc', 0):.2f}
- VAL: ${scanner.get('val', 0):.2f}
- VWAP: ${scanner.get('vwap', 0):.2f}
- Position: {scanner.get('position', 'unknown')}
- VWAP Zone: {scanner.get('vwap_zone', 'unknown')}

MOMENTUM:
- RSI: {scanner.get('rsi', 50):.1f} ({scanner.get('rsi_zone', 'neutral')})
- Flow: {scanner.get('flow_imbalance', 0):+.2f} ({scanner.get('flow_state', 'balanced')})

SCANNER NOTES:
{chr(10).join(['- ' + n for n in scanner.get('notes', [])])}

HISTORICAL CONTEXT:
- Your win rate on similar setups: {context.win_rate:.0%}
- Average R-multiple: {context.avg_r_multiple:.1f}R
- Recent trades on {context.symbol}: {len(context.recent_trades)}

{"EARNINGS in " + str(context.earnings_days) + " days - CAUTION" if 0 <= context.earnings_days <= 5 else ""}

Provide your analysis:"""

        try:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._build_system_prompt(regime)},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=400,
                    temperature=0.7
                )
                commentary = response.choices[0].message.content.strip()
            
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=400,
                    system=self._build_system_prompt(regime),
                    messages=[{"role": "user", "content": user_prompt}]
                )
                commentary = response.content[0].text.strip()
            
            # Parse out key levels if mentioned
            key_levels = self._extract_key_levels(commentary, scanner)
            
            return {
                'commentary': commentary,
                'key_levels': key_levels,
                'regime': regime.regime.value,
                'regime_strategy': regime.regime.strategy_notes,
                'model_used': self.model
            }
            
        except Exception as e:
            return {
                'commentary': f'AI analysis error: {str(e)}',
                'key_levels': {},
                'error': str(e)
            }
    
    def _extract_key_levels(self, commentary: str, scanner: Dict) -> Dict:
        """Extract numeric levels from commentary or use scanner defaults"""
        # For now, use scanner levels - could enhance with NLP extraction
        return {
            'entry_zone': (scanner.get('val', 0), scanner.get('poc', 0)),
            'stop': scanner.get('val', 0) * 0.99,  # 1% below VAL
            'target_1': scanner.get('poc', 0),
            'target_2': scanner.get('vah', 0)
        }


# =============================================================================
# TRADE JOURNAL WITH AI REVIEW
# =============================================================================

@dataclass
class TradeRecord:
    """Complete trade record for journaling"""
    id: str
    symbol: str
    direction: str  # LONG or SHORT
    entry_date: datetime
    entry_price: float
    stop_price: float
    target_1: float
    target_2: Optional[float]
    
    # Scanner context at entry
    signal_at_entry: str
    confidence_at_entry: float
    regime_at_entry: str
    scanner_notes: List[str]
    ai_commentary: str
    
    # Outcome (filled after close)
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""  # "TARGET_1", "TARGET_2", "STOP", "TIME", "MANUAL"
    r_multiple: Optional[float] = None
    pnl: Optional[float] = None
    
    # Post-trade AI review
    ai_review: str = ""
    lessons: List[str] = field(default_factory=list)


class TradeJournal:
    """
    AI-enhanced trade journal that learns from your patterns.
    """
    
    def __init__(self, db_path: str = "./trade_journal.db"):
        self.db_path = db_path
        self.ai_engine = None
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                direction TEXT,
                entry_date TEXT,
                entry_price REAL,
                stop_price REAL,
                target_1 REAL,
                target_2 REAL,
                signal_at_entry TEXT,
                confidence_at_entry REAL,
                regime_at_entry TEXT,
                scanner_notes TEXT,
                ai_commentary TEXT,
                exit_date TEXT,
                exit_price REAL,
                exit_reason TEXT,
                r_multiple REAL,
                pnl REAL,
                ai_review TEXT,
                lessons TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def set_ai_engine(self, engine: AICommentaryEngine):
        """Set AI engine for reviews"""
        self.ai_engine = engine
    
    def log_entry(self, trade: TradeRecord) -> str:
        """Log trade entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (
                id, symbol, direction, entry_date, entry_price,
                stop_price, target_1, target_2, signal_at_entry,
                confidence_at_entry, regime_at_entry, scanner_notes,
                ai_commentary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.id, trade.symbol, trade.direction,
            trade.entry_date.isoformat(), trade.entry_price,
            trade.stop_price, trade.target_1, trade.target_2,
            trade.signal_at_entry, trade.confidence_at_entry,
            trade.regime_at_entry, json.dumps(trade.scanner_notes),
            trade.ai_commentary
        ))
        
        conn.commit()
        conn.close()
        
        return trade.id
    
    def log_exit(self, trade_id: str, exit_price: float, 
                 exit_reason: str, exit_date: datetime = None) -> Dict:
        """Log trade exit and calculate R-multiple"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get entry data
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        
        if not row:
            return {'error': 'Trade not found'}
        
        entry_price = row[4]
        stop_price = row[5]
        direction = row[2]
        
        # Calculate R-multiple
        risk = abs(entry_price - stop_price)
        if direction == 'LONG':
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price
        
        r_multiple = pnl / risk if risk > 0 else 0
        
        exit_date = exit_date or datetime.now()
        
        cursor.execute("""
            UPDATE trades SET
                exit_date = ?,
                exit_price = ?,
                exit_reason = ?,
                r_multiple = ?,
                pnl = ?
            WHERE id = ?
        """, (exit_date.isoformat(), exit_price, exit_reason, 
              r_multiple, pnl, trade_id))
        
        conn.commit()
        conn.close()
        
        return {
            'trade_id': trade_id,
            'r_multiple': r_multiple,
            'pnl': pnl,
            'exit_reason': exit_reason
        }
    
    def get_similar_trades(self, symbol: str = None, signal: str = None,
                          regime: str = None, limit: int = 20) -> List[Dict]:
        """Get historical trades matching criteria"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM trades WHERE exit_date IS NOT NULL"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if signal:
            query += " AND signal_at_entry = ?"
            params.append(signal)
        if regime:
            query += " AND regime_at_entry = ?"
            params.append(regime)
        
        query += f" ORDER BY entry_date DESC LIMIT {limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to dicts
        columns = ['id', 'symbol', 'direction', 'entry_date', 'entry_price',
                   'stop_price', 'target_1', 'target_2', 'signal_at_entry',
                   'confidence_at_entry', 'regime_at_entry', 'scanner_notes',
                   'ai_commentary', 'exit_date', 'exit_price', 'exit_reason',
                   'r_multiple', 'pnl', 'ai_review', 'lessons']
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_win_rate(self, signal: str = None, regime: str = None) -> Dict:
        """Calculate win rate for given criteria"""
        trades = self.get_similar_trades(signal=signal, regime=regime, limit=100)
        
        if not trades:
            return {'win_rate': 0.5, 'avg_r': 0, 'sample_size': 0}
        
        wins = sum(1 for t in trades if (t.get('r_multiple') or 0) > 0)
        total = len(trades)
        avg_r = sum(t.get('r_multiple', 0) for t in trades) / total if total > 0 else 0
        
        return {
            'win_rate': wins / total if total > 0 else 0.5,
            'avg_r': avg_r,
            'sample_size': total
        }
    
    def generate_ai_review(self, trade_id: str) -> str:
        """Generate AI review of a closed trade"""
        if self.ai_engine is None:
            return "AI engine not configured"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row or not row[13]:  # No exit_date
            return "Trade still open"
        
        # Build review prompt
        prompt = f"""Review this completed trade:

SETUP:
- Symbol: {row[1]}
- Direction: {row[2]}
- Signal: {row[8]} ({row[9]:.0f}% confidence)
- Regime: {row[10]}
- Entry: ${row[4]:.2f}
- Stop: ${row[5]:.2f}
- Target 1: ${row[6]:.2f}

OUTCOME:
- Exit: ${row[14]:.2f}
- Reason: {row[15]}
- R-Multiple: {row[16]:.1f}R
- P&L: ${row[17]:.2f}

AI Commentary at Entry: {row[12]}

Provide a brief review:
1. What worked or didn't work?
2. Was the signal/regime alignment correct?
3. One key lesson for future trades.

Keep it to 3-4 sentences."""

        try:
            if self.ai_engine.provider == 'anthropic':
                response = self.ai_engine.client.messages.create(
                    model='claude-sonnet-4-20250514',
                    max_tokens=200,
                    system="You are a trading coach reviewing completed trades. Be constructive and specific.",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5
                )
                review = response.content[0].text.strip()
            elif self.ai_engine.provider == 'openai':
                response = self.ai_engine.client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[
                        {"role": "system", "content": "You are a trading coach reviewing completed trades. Be constructive and specific."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.5
                )
                review = response.choices[0].message.content.strip()
            else:
                review = "Review generation not supported for this provider"
            
            # Save review
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE trades SET ai_review = ? WHERE id = ?", 
                          (review, trade_id))
            conn.commit()
            conn.close()
            
            return review
            
        except Exception as e:
            return f"Review generation failed: {e}"
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary with AI insights"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM trades 
            WHERE exit_date IS NOT NULL 
            AND entry_date > ?
            ORDER BY entry_date DESC
        """, (cutoff,))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {'message': 'No trades in period', 'trades': 0}
        
        # Calculate stats
        total = len(rows)
        wins = sum(1 for r in rows if (r[16] or 0) > 0)
        total_r = sum(r[16] or 0 for r in rows)
        total_pnl = sum(r[17] or 0 for r in rows)
        
        # By regime
        by_regime = {}
        for r in rows:
            regime = r[10] or 'UNKNOWN'
            if regime not in by_regime:
                by_regime[regime] = {'trades': 0, 'wins': 0, 'total_r': 0}
            by_regime[regime]['trades'] += 1
            if (r[16] or 0) > 0:
                by_regime[regime]['wins'] += 1
            by_regime[regime]['total_r'] += r[16] or 0
        
        # By signal type
        by_signal = {}
        for r in rows:
            signal = r[8] or 'UNKNOWN'
            if signal not in by_signal:
                by_signal[signal] = {'trades': 0, 'wins': 0, 'total_r': 0}
            by_signal[signal]['trades'] += 1
            if (r[16] or 0) > 0:
                by_signal[signal]['wins'] += 1
            by_signal[signal]['total_r'] += r[16] or 0
        
        return {
            'period_days': days,
            'total_trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': wins / total if total > 0 else 0,
            'total_r': total_r,
            'avg_r': total_r / total if total > 0 else 0,
            'total_pnl': total_pnl,
            'by_regime': by_regime,
            'by_signal': by_signal
        }


# =============================================================================
# NEWS SENTIMENT ANALYZER
# =============================================================================

class NewsSentimentAnalyzer:
    """
    Analyzes news headlines and generates sentiment scores.
    
    For hedge fund level, you'd integrate:
    - Benzinga Pro or NewsAPI for real-time headlines
    - SEC EDGAR for filing alerts
    - Twitter/StockTwits for social sentiment
    
    This module provides the LLM analysis layer.
    """
    
    def __init__(self, ai_engine: AICommentaryEngine):
        self.ai_engine = ai_engine
    
    def analyze_headlines(self, symbol: str, headlines: List[str]) -> Dict:
        """
        Analyze news headlines for trading impact.
        
        Args:
            symbol: Stock symbol
            headlines: List of recent news headlines
        
        Returns:
            Dict with sentiment score, key events, and trading implications
        """
        if not headlines:
            return {'sentiment': 'NEUTRAL', 'score': 0, 'events': []}
        
        if self.ai_engine.client is None:
            return {'sentiment': 'UNKNOWN', 'score': 0, 'error': 'AI not configured'}
        
        prompt = f"""Analyze these headlines for {symbol} and their potential market impact:

{chr(10).join([f"- {h}" for h in headlines[:10]])}

Provide:
1. Overall sentiment (BULLISH/BEARISH/NEUTRAL)
2. Sentiment score (-100 to +100)
3. Key events that could move the stock
4. Timeframe of impact (TODAY/THIS_WEEK/LONGER_TERM)

Respond in JSON format:
{{"sentiment": "...", "score": X, "key_events": ["..."], "timeframe": "...", "trading_implication": "..."}}"""

        try:
            if self.ai_engine.provider == 'anthropic':
                response = self.ai_engine.client.messages.create(
                    model='claude-sonnet-4-20250514',
                    max_tokens=200,
                    system="You are a financial news analyst. Analyze headlines for trading implications. Always respond in valid JSON.",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                result = response.content[0].text.strip()
            elif self.ai_engine.provider == 'openai':
                response = self.ai_engine.client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[
                        {"role": "system", "content": "You are a financial news analyst. Analyze headlines for trading implications. Always respond in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
                result = response.choices[0].message.content.strip()
                
                # Parse JSON
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return {'sentiment': 'NEUTRAL', 'score': 0, 'raw': result}
            
            return {'sentiment': 'UNKNOWN', 'score': 0, 'error': 'Provider not supported'}
            
        except Exception as e:
            return {'sentiment': 'UNKNOWN', 'score': 0, 'error': str(e)}


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

class HedgeFundAdvisor:
    """
    Main integration class that combines all AI capabilities.
    
    Usage:
        advisor = HedgeFundAdvisor()
        
        # Analyze a setup
        result = advisor.analyze_setup(
            symbol="META",
            scanner_result=scanner.scan(df, "META"),
            spy_data=spy_context,
            vix=current_vix
        )
        
        # Log a trade
        advisor.journal.log_entry(trade_record)
        
        # Get performance insights
        stats = advisor.journal.get_performance_summary(30)
    """
    
    def __init__(self, provider: str = 'anthropic', journal_path: str = "./trade_journal.db"):
        self.regime_detector = RegimeDetector()
        self.ai_engine = AICommentaryEngine(provider=provider)
        self.journal = TradeJournal(db_path=journal_path)
        self.journal.set_ai_engine(self.ai_engine)
        self.news_analyzer = NewsSentimentAnalyzer(self.ai_engine)
    
    def analyze_setup(self, 
                      symbol: str,
                      scanner_result: Dict,
                      spy_data: Dict,
                      vix: float,
                      news_headlines: List[str] = None,
                      earnings_days: int = -1) -> Dict:
        """
        Full hedge-fund level analysis of a setup.
        
        Returns comprehensive analysis including:
        - Scanner signal interpretation
        - Market regime context
        - Historical win rate on similar setups
        - AI commentary
        - News sentiment (if provided)
        - Risk-adjusted position sizing note
        """
        # Detect regime
        regime = self.regime_detector.analyze(spy_data, vix)
        
        # Get historical stats
        stats = self.journal.get_win_rate(
            signal=scanner_result.get('signal'),
            regime=regime.regime.value
        )
        
        # Build context
        context = TradeContext(
            symbol=symbol,
            scanner_result=scanner_result,
            regime=regime,
            recent_trades=self.journal.get_similar_trades(symbol=symbol, limit=5),
            win_rate=stats['win_rate'],
            avg_r_multiple=stats['avg_r'],
            news_headlines=news_headlines or [],
            earnings_days=earnings_days
        )
        
        # Get AI analysis
        ai_result = self.ai_engine.analyze(context)
        
        # Analyze news if provided
        news_sentiment = None
        if news_headlines:
            news_sentiment = self.news_analyzer.analyze_headlines(symbol, news_headlines)
        
        return {
            'symbol': symbol,
            'scanner_signal': scanner_result.get('signal'),
            'scanner_confidence': scanner_result.get('confidence'),
            'market_regime': {
                'regime': regime.regime.value,
                'strategy': regime.regime.strategy_notes,
                'vix': regime.vix_level,
                'confidence': regime.confidence
            },
            'historical_stats': {
                'win_rate': stats['win_rate'],
                'avg_r': stats['avg_r'],
                'sample_size': stats['sample_size']
            },
            'ai_analysis': ai_result,
            'news_sentiment': news_sentiment,
            'timestamp': datetime.now().isoformat()
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Demo the system
    print("=" * 70)
    print("AI TRADING ADVISOR - DEMO")
    print("=" * 70)
    
    # Initialize
    advisor = HedgeFundAdvisor(provider='openai')
    
    # Mock scanner result (would come from your MTF scanner)
    mock_scanner = {
        'signal': 'LONG_SETUP',
        'confidence': 72,
        'confluence': 75,
        'bull_score': 68,
        'bear_score': 32,
        'price': 619.28,
        'vah': 667.72,
        'poc': 660.40,
        'val': 647.22,
        'vwap': 619.63,
        'position': 'BELOW_VALUE',
        'vwap_zone': 'AT_VWAP',
        'rsi': 33.58,
        'rsi_zone': 'NEAR_OVERSOLD',
        'flow_imbalance': 0.15,
        'flow_state': 'MILD_BUY_FLOW',
        'notes': [
            'Price below VAL - bearish breakdown zone',
            'RSI near oversold - watch for reversal',
            'Flow showing mild buying pressure'
        ]
    }
    
    # Mock SPY data for regime detection
    mock_spy = {
        'price': 595.50,
        'sma_20': 590.00,
        'sma_50': 585.00,
        'sma_200': 550.00,
        'atr_20': 8.5,
        'high_20': 605.00,
        'low_20': 580.00
    }
    
    # Run analysis
    result = advisor.analyze_setup(
        symbol="META",
        scanner_result=mock_scanner,
        spy_data=mock_spy,
        vix=16.5,
        earnings_days=-5  # 5 days since earnings
    )
    
    # Print results
    print("\n📊 ANALYSIS RESULT:")
    print("-" * 70)
    print(f"Symbol: {result['symbol']}")
    print(f"Signal: {result['scanner_signal']} ({result['scanner_confidence']:.0f}%)")
    print(f"\n🌐 MARKET REGIME: {result['market_regime']['regime']}")
    print(f"   Strategy: {result['market_regime']['strategy']}")
    print(f"   VIX: {result['market_regime']['vix']:.1f}")
    print(f"\n📈 HISTORICAL STATS:")
    print(f"   Win Rate: {result['historical_stats']['win_rate']:.0%}")
    print(f"   Avg R: {result['historical_stats']['avg_r']:.1f}")
    print(f"   Sample: {result['historical_stats']['sample_size']} trades")
    print(f"\n🤖 AI ANALYSIS:")
    print(result['ai_analysis'].get('commentary', 'N/A'))
    print("=" * 70)
