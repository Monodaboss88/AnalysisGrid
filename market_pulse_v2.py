"""
Market Pulse V2 — Market Breadth + V2 Context Aggregation
===========================================================
Scans multiple stocks, aggregates signals, and builds a historical
analysis database with market breadth, sector rotation, and now
V2-enriched context (squeeze concentration, weekly alignment, IV regime).

V1 → V2 CHANGES:
- Import from market_scanner_v2 / finnhub_scanner_v2 (canonical V2)
- StockPulse gains squeeze_active, weekly_trend, iv_regime, vp_shape fields
- Database schema extended with V2 columns (auto-migrated)
- MarketPulse gains squeeze_count, weekly_bull_count, weekly_bear_count,
  avg_iv_rank, market_squeeze_pct for aggregate V2 breadth
- scan_stock_enriched() uses V2 context for richer pulse data
- _aggregate_pulses() computes V2 breadth metrics
- All V1 methods/schema preserved + backward compatible

Author: Rob's Trading Systems (Strategic Edge Flow)
Version: 2.0.0
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# V2 scanner imports
try:
    from market_scanner_v2 import MarketScanner as BaseMarketScanner, V2Context
    _v2_scanner_available = True
except ImportError:
    _v2_scanner_available = False
    class V2Context:
        pass

# Finnhub extended scanner (for range structure / order flow)
try:
    from finnhub_scanner_v2 import FinnhubScanner
    _finnhub_available = True
except ImportError:
    _finnhub_available = False

# V1 fallback
try:
    from finnhub_scanner_v2 import FinnhubScanner as FinnhubScannerV1, TechnicalCalculator as TechCalcV1
    _v1_available = True
except ImportError:
    _v1_available = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StockPulse:
    """Individual stock pulse data"""
    symbol: str
    timestamp: str
    timeframe: str
    signal: str              # GREEN, RED, YELLOW
    confidence: float
    bull_score: int
    bear_score: int
    rsi: float
    vwap_deviation: float
    position: str            # ABOVE_VALUE, BELOW_VALUE, IN_VALUE
    current_price: float
    vah: float
    val: float
    poc: float
    sector: str = ""
    # V2 enrichment fields
    squeeze_active: bool = False
    squeeze_days: int = 0
    weekly_trend: str = ""       # STRONG_UPTREND, UPTREND, NEUTRAL, DOWNTREND, STRONG_DOWNTREND
    iv_regime: str = ""          # low, normal, elevated, extreme
    iv_rank: float = 0.0        # 0-100
    vp_shape: str = ""          # normal, p-shape, b-shape, d-shape, thin
    quality_grade: str = ""     # A+, A, B+, B, C, D


@dataclass
class MarketPulse:
    """Aggregated market pulse data"""
    timestamp: str
    total_scanned: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    bullish_pct: float
    bearish_pct: float
    avg_confidence: float
    avg_rsi: float
    market_bias: str         # BULLISH, BEARISH, NEUTRAL, LEAN_BULLISH, LEAN_BEARISH
    sector_breakdown: Dict[str, Dict]
    top_bullish: List[Dict]
    top_bearish: List[Dict]
    # V2 aggregate breadth
    squeeze_count: int = 0
    squeeze_pct: float = 0.0
    weekly_bull_count: int = 0
    weekly_bear_count: int = 0
    weekly_neutral_count: int = 0
    avg_iv_rank: float = 0.0
    iv_extreme_count: int = 0
    squeeze_symbols: List[str] = None

    def __post_init__(self):
        if self.squeeze_symbols is None:
            self.squeeze_symbols = []


# =============================================================================
# SECTOR MAP
# =============================================================================

SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "META": "Technology", "NVDA": "Technology", "AMD": "Technology", "INTC": "Technology",
    "CRM": "Technology", "ORCL": "Technology", "ADBE": "Technology", "NOW": "Technology",
    "AVGO": "Technology", "QCOM": "Technology", "TXN": "Technology", "MU": "Technology",
    "AMAT": "Technology", "LRCX": "Technology", "KLAC": "Technology", "MRVL": "Technology",

    # Consumer/Retail
    "AMZN": "Consumer", "TSLA": "Consumer", "HD": "Consumer", "NKE": "Consumer",
    "SBUX": "Consumer", "MCD": "Consumer", "COST": "Consumer", "WMT": "Consumer",
    "TGT": "Consumer", "LOW": "Consumer", "TJX": "Consumer", "ROST": "Consumer",

    # Finance
    "JPM": "Finance", "BAC": "Finance", "WFC": "Finance", "GS": "Finance",
    "MS": "Finance", "C": "Finance", "BLK": "Finance", "SCHW": "Finance",
    "V": "Finance", "MA": "Finance", "AXP": "Finance", "PYPL": "Finance",

    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "LLY": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare", "ISRG": "Healthcare",

    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "MPC": "Energy", "PSX": "Energy", "VLO": "Energy",

    # Industrial
    "CAT": "Industrial", "DE": "Industrial", "BA": "Industrial", "HON": "Industrial",
    "UPS": "Industrial", "RTX": "Industrial", "LMT": "Industrial", "GE": "Industrial",

    # Communication
    "NFLX": "Communication", "DIS": "Communication", "CMCSA": "Communication",
    "T": "Communication", "VZ": "Communication", "TMUS": "Communication",
}


# =============================================================================
# MARKET PULSE ANALYZER V2
# =============================================================================

class MarketPulseAnalyzer:
    """
    Market breadth analysis with historical database + V2 context aggregation.

    V2 adds:
    - Squeeze concentration: what % of scanned stocks are in a squeeze
    - Weekly alignment: how many stocks have weekly uptrend vs downtrend
    - IV regime distribution: avg IV rank, extreme count
    - Per-stock V2 fields stored in database for historical analysis

    Usage:
        analyzer = MarketPulseAnalyzer()

        # V1 compatible
        pulse = analyzer.scan_watchlist(["AAPL", "MSFT", "META"], "4hr")

        # V2 enriched
        pulse = analyzer.scan_watchlist_enriched(["AAPL", "MSFT", "META"])
    """

    def __init__(self, db_path: str = "scanner_data/market_pulse.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._init_database()

        # Initialize V2 scanner (preferred)
        if _v2_scanner_available:
            self.scanner = BaseMarketScanner()
            self._scanner_version = "v2"
        elif _v1_available:
            finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
            self.scanner = FinnhubScannerV1(finnhub_key) if finnhub_key else None
            self._scanner_version = "v1"
        else:
            self.scanner = None
            self._scanner_version = None

    def _init_database(self):
        """Create/migrate database tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Stock pulse table
        c.execute("""
            CREATE TABLE IF NOT EXISTS stock_pulses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal TEXT,
                confidence REAL,
                bull_score INTEGER,
                bear_score INTEGER,
                rsi REAL,
                vwap_deviation REAL,
                position TEXT,
                current_price REAL,
                vah REAL,
                val REAL,
                poc REAL,
                sector TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Market pulse table
        c.execute("""
            CREATE TABLE IF NOT EXISTS market_pulses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_scanned INTEGER,
                bullish_count INTEGER,
                bearish_count INTEGER,
                neutral_count INTEGER,
                bullish_pct REAL,
                bearish_pct REAL,
                avg_confidence REAL,
                avg_rsi REAL,
                market_bias TEXT,
                sector_data TEXT,
                top_bullish TEXT,
                top_bearish TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_stock_symbol ON stock_pulses(symbol)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_stock_timestamp ON stock_pulses(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_market_timestamp ON market_pulses(timestamp)")

        # V2 migration: add columns if missing
        existing = {row[1] for row in c.execute("PRAGMA table_info(stock_pulses)").fetchall()}
        v2_columns = {
            "squeeze_active": "INTEGER DEFAULT 0",
            "squeeze_days": "INTEGER DEFAULT 0",
            "weekly_trend": "TEXT DEFAULT ''",
            "iv_regime": "TEXT DEFAULT ''",
            "iv_rank": "REAL DEFAULT 0",
            "vp_shape": "TEXT DEFAULT ''",
            "quality_grade": "TEXT DEFAULT ''",
        }
        for col, col_type in v2_columns.items():
            if col not in existing:
                c.execute(f"ALTER TABLE stock_pulses ADD COLUMN {col} {col_type}")

        mp_existing = {row[1] for row in c.execute("PRAGMA table_info(market_pulses)").fetchall()}
        mp_v2_columns = {
            "squeeze_count": "INTEGER DEFAULT 0",
            "squeeze_pct": "REAL DEFAULT 0",
            "weekly_bull_count": "INTEGER DEFAULT 0",
            "weekly_bear_count": "INTEGER DEFAULT 0",
            "avg_iv_rank": "REAL DEFAULT 0",
            "squeeze_symbols": "TEXT DEFAULT '[]'",
        }
        for col, col_type in mp_v2_columns.items():
            if col not in mp_existing:
                c.execute(f"ALTER TABLE market_pulses ADD COLUMN {col} {col_type}")

        conn.commit()
        conn.close()

    def get_sector(self, symbol: str) -> str:
        return SECTOR_MAP.get(symbol.upper(), "Other")

    # =========================================================================
    # V1 COMPATIBLE SCANNING
    # =========================================================================

    def scan_stock(self, symbol: str, timeframe: str = "4hr") -> Optional[StockPulse]:
        """V1 compatible: scan a single stock"""
        if not self.scanner:
            return None

        try:
            if self._scanner_version == "v2":
                result = self.scanner.analyze(symbol, timeframe)
                if not result:
                    return None
                pulse = StockPulse(
                    symbol=symbol.upper(),
                    timestamp=datetime.now().isoformat(),
                    timeframe=timeframe,
                    signal=self._map_signal(result.signal),
                    confidence=result.quality_score,
                    bull_score=int(result.bull_score),
                    bear_score=int(result.bear_score),
                    rsi=result.rsi if hasattr(result, 'rsi') else 50,
                    vwap_deviation=0,
                    position=result.position if hasattr(result, 'position') else "UNKNOWN",
                    current_price=result.current_price,
                    vah=result.vah, val=result.val, poc=result.poc,
                    sector=self.get_sector(symbol),
                    quality_grade=result.quality_grade if hasattr(result, 'quality_grade') else "",
                )
            else:
                # V1 path
                result = self.scanner.get_technicals(symbol, timeframe)
                if not result:
                    return None
                analysis = TechCalcV1.calculate_signal(result)
                pulse = StockPulse(
                    symbol=symbol.upper(),
                    timestamp=datetime.now().isoformat(),
                    timeframe=timeframe,
                    signal=analysis.get("signal", "YELLOW"),
                    confidence=analysis.get("confidence", 0),
                    bull_score=analysis.get("bull_score", 0),
                    bear_score=analysis.get("bear_score", 0),
                    rsi=result.get("rsi", 50),
                    vwap_deviation=result.get("vwap_deviation", 0),
                    position=analysis.get("position", "UNKNOWN"),
                    current_price=result.get("current_price", 0),
                    vah=result.get("vah", 0),
                    val=result.get("val", 0),
                    poc=result.get("poc", 0),
                    sector=self.get_sector(symbol),
                )

            self._store_stock_pulse(pulse)
            return pulse

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            return None

    def scan_stock_enriched(self, symbol: str, timeframe: str = "4hr") -> Optional[StockPulse]:
        """
        V2 enriched: scan with squeeze/weekly/IV/VP context.
        Falls back to scan_stock() if V2 not available.
        """
        if not _v2_scanner_available or not self.scanner:
            return self.scan_stock(symbol, timeframe)

        try:
            result, ctx = self.scanner.analyze_enriched(symbol, timeframe)
            if not result:
                return None

            # Extract V2 context fields
            squeeze_active = False
            squeeze_days = 0
            weekly_trend = ""
            iv_regime = ""
            iv_rank = 0.0
            vp_shape = ""

            if ctx:
                if hasattr(ctx, 'squeeze') and ctx.squeeze:
                    squeeze_active = getattr(ctx.squeeze, 'is_squeezed', False)
                    squeeze_days = getattr(ctx.squeeze, 'squeeze_days', 0)
                if hasattr(ctx, 'weekly') and ctx.weekly:
                    weekly_trend = getattr(ctx.weekly, 'trend', '')
                if hasattr(ctx, 'iv') and ctx.iv:
                    iv_regime = getattr(ctx.iv, 'iv_regime', '')
                    iv_rank = getattr(ctx.iv, 'iv_rank', 0.0)
                if hasattr(ctx, 'vp') and ctx.vp:
                    vp_shape = getattr(ctx.vp, 'profile_shape', '')

            pulse = StockPulse(
                symbol=symbol.upper(),
                timestamp=datetime.now().isoformat(),
                timeframe=timeframe,
                signal=self._map_signal(result.signal),
                confidence=result.quality_score,
                bull_score=int(result.bull_score),
                bear_score=int(result.bear_score),
                rsi=result.rsi if hasattr(result, 'rsi') else 50,
                vwap_deviation=0,
                position=result.position if hasattr(result, 'position') else "UNKNOWN",
                current_price=result.current_price,
                vah=result.vah, val=result.val, poc=result.poc,
                sector=self.get_sector(symbol),
                squeeze_active=squeeze_active,
                squeeze_days=squeeze_days,
                weekly_trend=weekly_trend,
                iv_regime=iv_regime,
                iv_rank=iv_rank,
                vp_shape=vp_shape,
                quality_grade=result.quality_grade if hasattr(result, 'quality_grade') else "",
            )

            self._store_stock_pulse(pulse)
            return pulse

        except Exception as e:
            print(f"Error enriched scan {symbol}: {e}")
            return self.scan_stock(symbol, timeframe)

    def _map_signal(self, signal: str) -> str:
        """Map V2 signal names to pulse colors"""
        signal = signal.upper() if signal else "YELLOW"
        if signal in ("LONG_SETUP", "BULLISH", "GREEN"):
            return "GREEN"
        elif signal in ("SHORT_SETUP", "BEARISH", "RED"):
            return "RED"
        return "YELLOW"

    # =========================================================================
    # WATCHLIST SCANNING
    # =========================================================================

    def scan_watchlist(self, symbols: List[str], timeframe: str = "4hr") -> MarketPulse:
        """V1 compatible: scan watchlist"""
        pulses = []
        for symbol in symbols:
            pulse = self.scan_stock(symbol, timeframe)
            if pulse:
                pulses.append(pulse)
        return self._aggregate_pulses(pulses)

    def scan_watchlist_enriched(self, symbols: List[str], timeframe: str = "4hr") -> MarketPulse:
        """V2 enriched: scan watchlist with full V2 context"""
        pulses = []
        for symbol in symbols:
            pulse = self.scan_stock_enriched(symbol, timeframe)
            if pulse:
                pulses.append(pulse)
        return self._aggregate_pulses(pulses)

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    def _aggregate_pulses(self, pulses: List[StockPulse]) -> MarketPulse:
        """Aggregate individual pulses into market pulse with V2 breadth"""
        if not pulses:
            return MarketPulse(
                timestamp=datetime.now().isoformat(),
                total_scanned=0,
                bullish_count=0, bearish_count=0, neutral_count=0,
                bullish_pct=0, bearish_pct=0,
                avg_confidence=0, avg_rsi=50,
                market_bias="NEUTRAL",
                sector_breakdown={}, top_bullish=[], top_bearish=[],
            )

        bullish = [p for p in pulses if p.signal == "GREEN"]
        bearish = [p for p in pulses if p.signal == "RED"]
        neutral = [p for p in pulses if p.signal == "YELLOW"]

        total = len(pulses)
        bullish_pct = (len(bullish) / total) * 100
        bearish_pct = (len(bearish) / total) * 100
        avg_confidence = sum(p.confidence for p in pulses) / total
        avg_rsi = sum(p.rsi for p in pulses) / total

        # Market bias
        if bullish_pct > 60:
            market_bias = "BULLISH"
        elif bearish_pct > 60:
            market_bias = "BEARISH"
        elif bullish_pct > bearish_pct + 10:
            market_bias = "LEAN_BULLISH"
        elif bearish_pct > bullish_pct + 10:
            market_bias = "LEAN_BEARISH"
        else:
            market_bias = "NEUTRAL"

        # Sector breakdown
        sector_data = defaultdict(lambda: {"bullish": 0, "bearish": 0, "neutral": 0, "total": 0})
        for p in pulses:
            sector = p.sector or "Other"
            sector_data[sector]["total"] += 1
            if p.signal == "GREEN":
                sector_data[sector]["bullish"] += 1
            elif p.signal == "RED":
                sector_data[sector]["bearish"] += 1
            else:
                sector_data[sector]["neutral"] += 1

        for sector in sector_data:
            ts = sector_data[sector]["total"]
            if ts > 0:
                sector_data[sector]["bullish_pct"] = (sector_data[sector]["bullish"] / ts) * 100
                sector_data[sector]["bearish_pct"] = (sector_data[sector]["bearish"] / ts) * 100

        # Top movers
        top_bullish = sorted(bullish, key=lambda x: x.confidence, reverse=True)[:5]
        top_bearish = sorted(bearish, key=lambda x: x.confidence, reverse=True)[:5]

        # V2 breadth metrics
        squeeze_stocks = [p for p in pulses if p.squeeze_active]
        squeeze_count = len(squeeze_stocks)
        squeeze_pct = (squeeze_count / total) * 100 if total > 0 else 0
        squeeze_symbols = [p.symbol for p in squeeze_stocks]

        weekly_bull = sum(1 for p in pulses if p.weekly_trend in ("UPTREND", "STRONG_UPTREND"))
        weekly_bear = sum(1 for p in pulses if p.weekly_trend in ("DOWNTREND", "STRONG_DOWNTREND"))
        weekly_neutral = total - weekly_bull - weekly_bear

        iv_ranks = [p.iv_rank for p in pulses if p.iv_rank > 0]
        avg_iv_rank = sum(iv_ranks) / len(iv_ranks) if iv_ranks else 0
        iv_extreme_count = sum(1 for p in pulses if p.iv_regime == "extreme")

        market_pulse = MarketPulse(
            timestamp=datetime.now().isoformat(),
            total_scanned=total,
            bullish_count=len(bullish),
            bearish_count=len(bearish),
            neutral_count=len(neutral),
            bullish_pct=bullish_pct,
            bearish_pct=bearish_pct,
            avg_confidence=avg_confidence,
            avg_rsi=avg_rsi,
            market_bias=market_bias,
            sector_breakdown=dict(sector_data),
            top_bullish=[asdict(p) for p in top_bullish],
            top_bearish=[asdict(p) for p in top_bearish],
            squeeze_count=squeeze_count,
            squeeze_pct=squeeze_pct,
            weekly_bull_count=weekly_bull,
            weekly_bear_count=weekly_bear,
            weekly_neutral_count=weekly_neutral,
            avg_iv_rank=avg_iv_rank,
            iv_extreme_count=iv_extreme_count,
            squeeze_symbols=squeeze_symbols,
        )

        self._store_market_pulse(market_pulse)
        return market_pulse

    # =========================================================================
    # DATABASE STORAGE
    # =========================================================================

    def _store_stock_pulse(self, pulse: StockPulse):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO stock_pulses
            (symbol, timestamp, timeframe, signal, confidence, bull_score, bear_score,
             rsi, vwap_deviation, position, current_price, vah, val, poc, sector,
             squeeze_active, squeeze_days, weekly_trend, iv_regime, iv_rank, vp_shape, quality_grade)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pulse.symbol, pulse.timestamp, pulse.timeframe, pulse.signal,
            pulse.confidence, pulse.bull_score, pulse.bear_score,
            pulse.rsi, pulse.vwap_deviation, pulse.position,
            pulse.current_price, pulse.vah, pulse.val, pulse.poc, pulse.sector,
            int(pulse.squeeze_active), pulse.squeeze_days,
            pulse.weekly_trend, pulse.iv_regime, pulse.iv_rank,
            pulse.vp_shape, pulse.quality_grade,
        ))
        conn.commit()
        conn.close()

    def _store_market_pulse(self, pulse: MarketPulse):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO market_pulses
            (timestamp, total_scanned, bullish_count, bearish_count, neutral_count,
             bullish_pct, bearish_pct, avg_confidence, avg_rsi, market_bias,
             sector_data, top_bullish, top_bearish,
             squeeze_count, squeeze_pct, weekly_bull_count, weekly_bear_count,
             avg_iv_rank, squeeze_symbols)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pulse.timestamp, pulse.total_scanned,
            pulse.bullish_count, pulse.bearish_count, pulse.neutral_count,
            pulse.bullish_pct, pulse.bearish_pct,
            pulse.avg_confidence, pulse.avg_rsi, pulse.market_bias,
            json.dumps(pulse.sector_breakdown),
            json.dumps(pulse.top_bullish),
            json.dumps(pulse.top_bearish),
            pulse.squeeze_count, pulse.squeeze_pct,
            pulse.weekly_bull_count, pulse.weekly_bear_count,
            pulse.avg_iv_rank,
            json.dumps(pulse.squeeze_symbols),
        ))
        conn.commit()
        conn.close()

    # =========================================================================
    # HISTORICAL QUERIES
    # =========================================================================

    def get_historical_pulses(self, days: int = 7) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        c.execute("SELECT * FROM market_pulses WHERE timestamp > ? ORDER BY timestamp DESC", (cutoff,))
        columns = [desc[0] for desc in c.description]
        results = [dict(zip(columns, row)) for row in c.fetchall()]
        conn.close()
        return results

    def get_stock_history(self, symbol: str, days: int = 30) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        c.execute(
            "SELECT * FROM stock_pulses WHERE symbol = ? AND timestamp > ? ORDER BY timestamp DESC",
            (symbol.upper(), cutoff)
        )
        columns = [desc[0] for desc in c.description]
        results = [dict(zip(columns, row)) for row in c.fetchall()]
        conn.close()
        return results

    def get_sector_performance(self, days: int = 1) -> Dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        c.execute("""
            SELECT sector, signal, COUNT(*) as count
            FROM stock_pulses
            WHERE timestamp > ? AND sector != ''
            GROUP BY sector, signal
        """, (cutoff,))
        results = c.fetchall()
        conn.close()

        sectors = defaultdict(lambda: {"bullish": 0, "bearish": 0, "neutral": 0})
        for sector, signal, count in results:
            if signal == "GREEN":
                sectors[sector]["bullish"] += count
            elif signal == "RED":
                sectors[sector]["bearish"] += count
            else:
                sectors[sector]["neutral"] += count

        for sector in sectors:
            total = sum(sectors[sector].values())
            if total > 0:
                sectors[sector]["bullish_pct"] = (sectors[sector]["bullish"] / total) * 100
                sectors[sector]["bearish_pct"] = (sectors[sector]["bearish"] / total) * 100
                sectors[sector]["total"] = total

        return dict(sectors)

    def get_squeeze_history(self, days: int = 7) -> List[Dict]:
        """V2: Get historical squeeze concentration data"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        c.execute("""
            SELECT timestamp, squeeze_count, squeeze_pct, squeeze_symbols,
                   weekly_bull_count, weekly_bear_count, avg_iv_rank
            FROM market_pulses
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        """, (cutoff,))
        columns = [desc[0] for desc in c.description]
        results = [dict(zip(columns, row)) for row in c.fetchall()]
        conn.close()
        return results

    def get_squeeze_stocks(self, days: int = 1) -> List[Dict]:
        """V2: Get stocks currently in a squeeze"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        c.execute("""
            SELECT symbol, squeeze_days, weekly_trend, iv_regime, iv_rank,
                   signal, confidence, current_price
            FROM stock_pulses
            WHERE timestamp > ? AND squeeze_active = 1
            ORDER BY squeeze_days DESC
        """, (cutoff,))
        columns = [desc[0] for desc in c.description]
        results = [dict(zip(columns, row)) for row in c.fetchall()]
        conn.close()
        return results

    def export_data(self, table: str = "stock_pulses", format: str = "json") -> str:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(f"SELECT * FROM {table}")
        columns = [desc[0] for desc in c.description]
        results = [dict(zip(columns, row)) for row in c.fetchall()]
        conn.close()

        if format == "json":
            return json.dumps(results, indent=2)
        else:
            if not results:
                return ""
            csv_lines = [",".join(columns)]
            for row in results:
                csv_lines.append(",".join(str(row.get(c, "")) for c in columns))
            return "\n".join(csv_lines)

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def print_pulse(self, pulse: MarketPulse) -> str:
        lines = [
            "=" * 60,
            f"📊 MARKET PULSE — {pulse.timestamp[:19]}",
            "=" * 60,
            f"Total Scanned: {pulse.total_scanned}",
            f"Bullish: {pulse.bullish_count} ({pulse.bullish_pct:.1f}%)",
            f"Bearish: {pulse.bearish_count} ({pulse.bearish_pct:.1f}%)",
            f"Neutral: {pulse.neutral_count}",
            f"Market Bias: {pulse.market_bias}",
            f"Avg RSI: {pulse.avg_rsi:.1f}",
            f"Avg Confidence: {pulse.avg_confidence:.1f}%",
        ]

        # V2 breadth
        if pulse.squeeze_count > 0:
            lines.append(f"\n🔥 SQUEEZE CONCENTRATION:")
            lines.append(f"   {pulse.squeeze_count} stocks ({pulse.squeeze_pct:.0f}%) in squeeze")
            if pulse.squeeze_symbols:
                lines.append(f"   Symbols: {', '.join(pulse.squeeze_symbols[:10])}")

        if pulse.weekly_bull_count or pulse.weekly_bear_count:
            lines.append(f"\n📈 WEEKLY ALIGNMENT:")
            lines.append(f"   Bull: {pulse.weekly_bull_count} | Bear: {pulse.weekly_bear_count} | Neutral: {pulse.weekly_neutral_count}")

        if pulse.avg_iv_rank > 0:
            lines.append(f"\n📊 IV REGIME:")
            lines.append(f"   Avg IV Rank: {pulse.avg_iv_rank:.0f}")
            if pulse.iv_extreme_count:
                lines.append(f"   Extreme IV: {pulse.iv_extreme_count} stocks")

        # Sector breakdown
        if pulse.sector_breakdown:
            lines.append(f"\n📊 SECTOR BREAKDOWN:")
            lines.append("-" * 40)
            for sector, data in sorted(pulse.sector_breakdown.items()):
                bull_pct = data.get("bullish_pct", 0)
                bear_pct = data.get("bearish_pct", 0)
                lines.append(f"   {sector:<15} Bull: {bull_pct:5.1f}% | Bear: {bear_pct:5.1f}%")

        # Top movers
        if pulse.top_bullish:
            lines.append(f"\n🏆 Top Bullish:")
            for stock in pulse.top_bullish[:3]:
                lines.append(f"   {stock['symbol']}: {stock['confidence']:.0f}% confidence")

        if pulse.top_bearish:
            lines.append(f"\n📉 Top Bearish:")
            for stock in pulse.top_bearish[:3]:
                lines.append(f"   {stock['symbol']}: {stock['confidence']:.0f}% confidence")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    analyzer = MarketPulseAnalyzer()
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    print("🔍 Scanning market pulse (V2 enriched)...")
    pulse = analyzer.scan_watchlist_enriched(test_symbols)
    print(analyzer.print_pulse(pulse))
