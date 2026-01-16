"""
Market Pulse Analysis System
=============================
Collects and stores market data to build your own analysis database.

Features:
- Scans multiple stocks and aggregates signals
- Market breadth analysis (% bullish vs bearish)
- Sector rotation tracking
- Historical data storage for backtesting
- Custom market pulse indicators

Author: Rob's Trading Systems
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Polygon for data
try:
    from polygon import RESTClient
    polygon_available = True
except ImportError:
    polygon_available = False

# Our scanner
from finnhub_scanner import FinnhubScanner, TechnicalCalculator


@dataclass
class StockPulse:
    """Individual stock pulse data"""
    symbol: str
    timestamp: str
    timeframe: str
    signal: str  # GREEN, RED, YELLOW
    confidence: float
    bull_score: int
    bear_score: int
    rsi: float
    vwap_deviation: float
    position: str  # ABOVE_VALUE, BELOW_VALUE, IN_VALUE
    current_price: float
    vah: float
    val: float
    poc: float
    sector: str = ""
    

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
    market_bias: str  # BULLISH, BEARISH, NEUTRAL
    sector_breakdown: Dict[str, Dict]
    top_bullish: List[Dict]
    top_bearish: List[Dict]


class MarketPulseAnalyzer:
    """
    Analyzes market breadth and stores historical data
    """
    
    # Sector mappings for common stocks
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
    
    def __init__(self, db_path: str = "scanner_data/market_pulse.db"):
        """Initialize with database path"""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        
        # Initialize scanner
        finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
        self.scanner = FinnhubScanner(finnhub_key) if finnhub_key else None
        
    def _init_database(self):
        """Create database tables for storing pulse data"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Stock pulse table - individual stock scans
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
        
        # Market pulse table - aggregated market data
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
        
        # Create indexes for fast queries
        c.execute("CREATE INDEX IF NOT EXISTS idx_stock_symbol ON stock_pulses(symbol)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_stock_timestamp ON stock_pulses(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_market_timestamp ON market_pulses(timestamp)")
        
        conn.commit()
        conn.close()
        
    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol"""
        return self.SECTOR_MAP.get(symbol.upper(), "Other")
    
    def scan_stock(self, symbol: str, timeframe: str = "4hr") -> Optional[StockPulse]:
        """Scan a single stock and return pulse data"""
        if not self.scanner:
            return None
            
        try:
            # Get technical data
            result = self.scanner.get_technicals(symbol, timeframe)
            if not result:
                return None
                
            # Calculate signal
            analysis = TechnicalCalculator.calculate_signal(result)
            
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
                sector=self.get_sector(symbol)
            )
            
            # Store in database
            self._store_stock_pulse(pulse)
            
            return pulse
            
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            return None
    
    def _store_stock_pulse(self, pulse: StockPulse):
        """Store stock pulse in database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            INSERT INTO stock_pulses 
            (symbol, timestamp, timeframe, signal, confidence, bull_score, bear_score,
             rsi, vwap_deviation, position, current_price, vah, val, poc, sector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pulse.symbol, pulse.timestamp, pulse.timeframe, pulse.signal,
            pulse.confidence, pulse.bull_score, pulse.bear_score,
            pulse.rsi, pulse.vwap_deviation, pulse.position,
            pulse.current_price, pulse.vah, pulse.val, pulse.poc, pulse.sector
        ))
        
        conn.commit()
        conn.close()
    
    def scan_watchlist(self, symbols: List[str], timeframe: str = "4hr") -> MarketPulse:
        """Scan entire watchlist and generate market pulse"""
        pulses = []
        
        for symbol in symbols:
            pulse = self.scan_stock(symbol, timeframe)
            if pulse:
                pulses.append(pulse)
        
        return self._aggregate_pulses(pulses)
    
    def _aggregate_pulses(self, pulses: List[StockPulse]) -> MarketPulse:
        """Aggregate individual pulses into market pulse"""
        if not pulses:
            return MarketPulse(
                timestamp=datetime.now().isoformat(),
                total_scanned=0,
                bullish_count=0, bearish_count=0, neutral_count=0,
                bullish_pct=0, bearish_pct=0,
                avg_confidence=0, avg_rsi=50,
                market_bias="NEUTRAL",
                sector_breakdown={},
                top_bullish=[], top_bearish=[]
            )
        
        # Count signals
        bullish = [p for p in pulses if p.signal == "GREEN"]
        bearish = [p for p in pulses if p.signal == "RED"]
        neutral = [p for p in pulses if p.signal == "YELLOW"]
        
        total = len(pulses)
        bullish_pct = (len(bullish) / total) * 100
        bearish_pct = (len(bearish) / total) * 100
        
        # Average stats
        avg_confidence = sum(p.confidence for p in pulses) / total
        avg_rsi = sum(p.rsi for p in pulses) / total
        
        # Determine market bias
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
        
        # Calculate sector percentages
        for sector in sector_data:
            total_sector = sector_data[sector]["total"]
            if total_sector > 0:
                sector_data[sector]["bullish_pct"] = (sector_data[sector]["bullish"] / total_sector) * 100
                sector_data[sector]["bearish_pct"] = (sector_data[sector]["bearish"] / total_sector) * 100
        
        # Top movers
        top_bullish = sorted(bullish, key=lambda x: x.confidence, reverse=True)[:5]
        top_bearish = sorted(bearish, key=lambda x: x.confidence, reverse=True)[:5]
        
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
            top_bearish=[asdict(p) for p in top_bearish]
        )
        
        # Store in database
        self._store_market_pulse(market_pulse)
        
        return market_pulse
    
    def _store_market_pulse(self, pulse: MarketPulse):
        """Store market pulse in database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            INSERT INTO market_pulses 
            (timestamp, total_scanned, bullish_count, bearish_count, neutral_count,
             bullish_pct, bearish_pct, avg_confidence, avg_rsi, market_bias,
             sector_data, top_bullish, top_bearish)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pulse.timestamp, pulse.total_scanned,
            pulse.bullish_count, pulse.bearish_count, pulse.neutral_count,
            pulse.bullish_pct, pulse.bearish_pct,
            pulse.avg_confidence, pulse.avg_rsi, pulse.market_bias,
            json.dumps(pulse.sector_breakdown),
            json.dumps(pulse.top_bullish),
            json.dumps(pulse.top_bearish)
        ))
        
        conn.commit()
        conn.close()
    
    def get_historical_pulses(self, days: int = 7) -> List[Dict]:
        """Get historical market pulses"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        c.execute("""
            SELECT * FROM market_pulses 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        """, (cutoff,))
        
        columns = [desc[0] for desc in c.description]
        results = [dict(zip(columns, row)) for row in c.fetchall()]
        
        conn.close()
        return results
    
    def get_stock_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical data for a specific stock"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        c.execute("""
            SELECT * FROM stock_pulses 
            WHERE symbol = ? AND timestamp > ? 
            ORDER BY timestamp DESC
        """, (symbol.upper(), cutoff))
        
        columns = [desc[0] for desc in c.description]
        results = [dict(zip(columns, row)) for row in c.fetchall()]
        
        conn.close()
        return results
    
    def get_sector_performance(self, days: int = 1) -> Dict:
        """Get sector performance over time"""
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
        
        # Aggregate by sector
        sectors = defaultdict(lambda: {"bullish": 0, "bearish": 0, "neutral": 0})
        for sector, signal, count in results:
            if signal == "GREEN":
                sectors[sector]["bullish"] += count
            elif signal == "RED":
                sectors[sector]["bearish"] += count
            else:
                sectors[sector]["neutral"] += count
        
        # Calculate percentages
        for sector in sectors:
            total = sum(sectors[sector].values())
            if total > 0:
                sectors[sector]["bullish_pct"] = (sectors[sector]["bullish"] / total) * 100
                sectors[sector]["bearish_pct"] = (sectors[sector]["bearish"] / total) * 100
                sectors[sector]["total"] = total
        
        return dict(sectors)
    
    def export_data(self, table: str = "stock_pulses", format: str = "json") -> str:
        """Export data for external analysis"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute(f"SELECT * FROM {table}")
        columns = [desc[0] for desc in c.description]
        results = [dict(zip(columns, row)) for row in c.fetchall()]
        
        conn.close()
        
        if format == "json":
            return json.dumps(results, indent=2)
        else:
            # CSV format
            if not results:
                return ""
            csv_lines = [",".join(columns)]
            for row in results:
                csv_lines.append(",".join(str(row.get(c, "")) for c in columns))
            return "\n".join(csv_lines)


# Quick test
if __name__ == "__main__":
    analyzer = MarketPulseAnalyzer()
    
    # Test with a few symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    print("🔍 Scanning market pulse...")
    pulse = analyzer.scan_watchlist(test_symbols, "4hr")
    
    print(f"\n📊 MARKET PULSE")
    print(f"=" * 50)
    print(f"Total Scanned: {pulse.total_scanned}")
    print(f"Bullish: {pulse.bullish_count} ({pulse.bullish_pct:.1f}%)")
    print(f"Bearish: {pulse.bearish_count} ({pulse.bearish_pct:.1f}%)")
    print(f"Neutral: {pulse.neutral_count}")
    print(f"Market Bias: {pulse.market_bias}")
    print(f"Avg RSI: {pulse.avg_rsi:.1f}")
    
    print(f"\n🏆 Top Bullish:")
    for stock in pulse.top_bullish[:3]:
        print(f"  {stock['symbol']}: {stock['confidence']:.0f}% confidence")
    
    print(f"\n📉 Top Bearish:")
    for stock in pulse.top_bearish[:3]:
        print(f"  {stock['symbol']}: {stock['confidence']:.0f}% confidence")
