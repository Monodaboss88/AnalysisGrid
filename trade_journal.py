"""
Trade Journal System
=====================
Log trades from scans with one click.
Track entries, exits, P&L, and learn from your history.

Author: Rob's Trading Systems
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class TradeStatus(str, Enum):
    PLANNED = "PLANNED"      # Logged from scan, not entered yet
    OPEN = "OPEN"            # Position is open
    CLOSED_WIN = "WIN"       # Closed for profit
    CLOSED_LOSS = "LOSS"     # Closed for loss
    CLOSED_BE = "BREAKEVEN"  # Closed at breakeven
    CANCELLED = "CANCELLED"  # Never entered, cancelled


@dataclass
class JournalEntry:
    """Single trade journal entry"""
    id: Optional[int]
    symbol: str
    direction: str  # LONG or SHORT
    timeframe: str
    
    # Entry details
    entry_price: float
    stop_loss: float
    target1: float
    target2: float
    
    # Risk management
    risk_reward_t1: float
    risk_reward_t2: float
    position_size: Optional[float] = None
    risk_amount: Optional[float] = None
    
    # Scan data at time of log
    signal: str = ""  # GREEN, RED, YELLOW
    confidence: float = 0
    bull_score: int = 0
    bear_score: int = 0
    ai_commentary: str = ""
    setup_grade: str = ""  # A+, A, B, C, F
    
    # Price levels at log time
    vah: float = 0
    poc: float = 0
    val: float = 0
    vwap: float = 0
    rsi: float = 0
    
    # Execution
    status: str = "PLANNED"
    actual_entry: Optional[float] = None
    actual_exit: Optional[float] = None
    exit_reason: str = ""  # hit_target, hit_stop, manual, etc.
    
    # P&L
    pnl_dollars: Optional[float] = None
    pnl_percent: Optional[float] = None
    pnl_r: Optional[float] = None  # P&L in R multiples
    
    # Notes
    notes: str = ""
    tags: str = ""  # comma-separated: "breakout,earnings,gap"
    screenshot_url: str = ""
    
    # Timestamps
    logged_at: str = ""
    entered_at: str = ""
    exited_at: str = ""


class TradeJournal:
    """
    Trade journaling system with SQLite storage
    """
    
    def __init__(self, db_path: str = "scanner_data/trade_journal.db"):
        """Initialize journal database"""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create journal tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                timeframe TEXT,
                
                entry_price REAL,
                stop_loss REAL,
                target1 REAL,
                target2 REAL,
                risk_reward_t1 REAL,
                risk_reward_t2 REAL,
                position_size REAL,
                risk_amount REAL,
                
                signal TEXT,
                confidence REAL,
                bull_score INTEGER,
                bear_score INTEGER,
                ai_commentary TEXT,
                setup_grade TEXT,
                
                vah REAL,
                poc REAL,
                val REAL,
                vwap REAL,
                rsi REAL,
                
                status TEXT DEFAULT 'PLANNED',
                actual_entry REAL,
                actual_exit REAL,
                exit_reason TEXT,
                
                pnl_dollars REAL,
                pnl_percent REAL,
                pnl_r REAL,
                
                notes TEXT,
                tags TEXT,
                screenshot_url TEXT,
                
                logged_at TEXT,
                entered_at TEXT,
                exited_at TEXT,
                
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_journal_symbol ON journal_entries(symbol)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_journal_status ON journal_entries(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_journal_logged ON journal_entries(logged_at)")
        
        conn.commit()
        conn.close()
    
    def log_trade(self, entry: JournalEntry) -> int:
        """Log a new trade from scan results"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        entry.logged_at = datetime.now().isoformat()
        
        c.execute("""
            INSERT INTO journal_entries (
                symbol, direction, timeframe,
                entry_price, stop_loss, target1, target2,
                risk_reward_t1, risk_reward_t2, position_size, risk_amount,
                signal, confidence, bull_score, bear_score,
                ai_commentary, setup_grade,
                vah, poc, val, vwap, rsi,
                status, notes, tags, logged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.symbol, entry.direction, entry.timeframe,
            entry.entry_price, entry.stop_loss, entry.target1, entry.target2,
            entry.risk_reward_t1, entry.risk_reward_t2, entry.position_size, entry.risk_amount,
            entry.signal, entry.confidence, entry.bull_score, entry.bear_score,
            entry.ai_commentary, entry.setup_grade,
            entry.vah, entry.poc, entry.val, entry.vwap, entry.rsi,
            entry.status, entry.notes, entry.tags, entry.logged_at
        ))
        
        entry_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return entry_id
    
    def update_trade(self, entry_id: int, updates: Dict) -> bool:
        """Update an existing trade entry"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Build dynamic update query
        updates["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [entry_id]
        
        c.execute(f"""
            UPDATE journal_entries 
            SET {set_clause}
            WHERE id = ?
        """, values)
        
        success = c.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def open_trade(self, entry_id: int, actual_entry: float) -> bool:
        """Mark trade as opened with actual entry price"""
        return self.update_trade(entry_id, {
            "status": TradeStatus.OPEN.value,
            "actual_entry": actual_entry,
            "entered_at": datetime.now().isoformat()
        })
    
    def close_trade(self, entry_id: int, actual_exit: float, exit_reason: str = "") -> bool:
        """Close a trade and calculate P&L"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get trade details
        c.execute("SELECT * FROM journal_entries WHERE id = ?", (entry_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return False
        
        columns = [desc[0] for desc in c.description]
        trade = dict(zip(columns, row))
        
        actual_entry = trade.get("actual_entry") or trade.get("entry_price")
        stop_loss = trade["stop_loss"]
        direction = trade["direction"]
        
        # Calculate P&L
        if direction == "LONG":
            pnl_percent = ((actual_exit - actual_entry) / actual_entry) * 100
            risk_per_share = actual_entry - stop_loss
        else:  # SHORT
            pnl_percent = ((actual_entry - actual_exit) / actual_entry) * 100
            risk_per_share = stop_loss - actual_entry
        
        # P&L in R multiples
        if risk_per_share != 0:
            if direction == "LONG":
                pnl_r = (actual_exit - actual_entry) / risk_per_share
            else:
                pnl_r = (actual_entry - actual_exit) / risk_per_share
        else:
            pnl_r = 0
        
        # Determine status
        if pnl_percent > 0.1:
            status = TradeStatus.CLOSED_WIN.value
        elif pnl_percent < -0.1:
            status = TradeStatus.CLOSED_LOSS.value
        else:
            status = TradeStatus.CLOSED_BE.value
        
        # Calculate dollar P&L if position size known
        position_size = trade.get("position_size")
        pnl_dollars = None
        if position_size:
            pnl_dollars = position_size * (pnl_percent / 100) * actual_entry
        
        self.update_trade(entry_id, {
            "status": status,
            "actual_exit": actual_exit,
            "exit_reason": exit_reason,
            "pnl_percent": round(pnl_percent, 2),
            "pnl_r": round(pnl_r, 2),
            "pnl_dollars": pnl_dollars,
            "exited_at": datetime.now().isoformat()
        })
        
        conn.close()
        return True
    
    def cancel_trade(self, entry_id: int, reason: str = "") -> bool:
        """Cancel a planned trade"""
        return self.update_trade(entry_id, {
            "status": TradeStatus.CANCELLED.value,
            "notes": reason
        })
    
    def get_trade(self, entry_id: int) -> Optional[Dict]:
        """Get a single trade entry"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT * FROM journal_entries WHERE id = ?", (entry_id,))
        row = c.fetchone()
        
        if not row:
            conn.close()
            return None
        
        columns = [desc[0] for desc in c.description]
        result = dict(zip(columns, row))
        conn.close()
        
        return result
    
    def get_trades(
        self,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
        days: int = 30,
        limit: int = 100
    ) -> List[Dict]:
        """Get trade entries with filters"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = "SELECT * FROM journal_entries WHERE logged_at > ?"
        params = [cutoff]
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())
        
        query += " ORDER BY logged_at DESC LIMIT ?"
        params.append(limit)
        
        c.execute(query, params)
        
        columns = [desc[0] for desc in c.description]
        results = [dict(zip(columns, row)) for row in c.fetchall()]
        
        conn.close()
        return results
    
    def get_stats(self, days: int = 30) -> Dict:
        """Get trading statistics"""
        trades = self.get_trades(days=days, limit=1000)
        
        if not trades:
            return {
                "period_days": days,
                "total_trades": 0,
                "message": "No trades in this period"
            }
        
        closed = [t for t in trades if t["status"] in ["WIN", "LOSS", "BREAKEVEN"]]
        wins = [t for t in closed if t["status"] == "WIN"]
        losses = [t for t in closed if t["status"] == "LOSS"]
        
        # Win rate
        win_rate = (len(wins) / len(closed) * 100) if closed else 0
        
        # Average R
        r_values = [t["pnl_r"] for t in closed if t["pnl_r"] is not None]
        avg_r = sum(r_values) / len(r_values) if r_values else 0
        
        # Average win/loss
        win_r = [t["pnl_r"] for t in wins if t["pnl_r"]]
        loss_r = [t["pnl_r"] for t in losses if t["pnl_r"]]
        avg_win_r = sum(win_r) / len(win_r) if win_r else 0
        avg_loss_r = sum(loss_r) / len(loss_r) if loss_r else 0
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win_r) + ((1 - win_rate/100) * avg_loss_r)
        
        # By setup grade
        grade_stats = {}
        for grade in ["A+", "A", "B", "C", "F"]:
            grade_trades = [t for t in closed if t.get("setup_grade") == grade]
            if grade_trades:
                grade_wins = [t for t in grade_trades if t["status"] == "WIN"]
                grade_stats[grade] = {
                    "count": len(grade_trades),
                    "win_rate": round(len(grade_wins) / len(grade_trades) * 100, 1)
                }
        
        return {
            "period_days": days,
            "total_trades": len(trades),
            "open_trades": len([t for t in trades if t["status"] == "OPEN"]),
            "planned_trades": len([t for t in trades if t["status"] == "PLANNED"]),
            "closed_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len([t for t in closed if t["status"] == "BREAKEVEN"]),
            "win_rate": round(win_rate, 1),
            "avg_r": round(avg_r, 2),
            "avg_win_r": round(avg_win_r, 2),
            "avg_loss_r": round(avg_loss_r, 2),
            "expectancy_r": round(expectancy, 2),
            "by_setup_grade": grade_stats
        }
    
    def export_journal(self, format: str = "json", days: int = 365) -> str:
        """Export journal for external analysis"""
        trades = self.get_trades(days=days, limit=10000)
        
        if format == "json":
            return json.dumps(trades, indent=2)
        else:
            # CSV
            if not trades:
                return ""
            columns = list(trades[0].keys())
            lines = [",".join(columns)]
            for t in trades:
                lines.append(",".join(str(t.get(c, "")).replace(",", ";") for c in columns))
            return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    journal = TradeJournal()
    
    # Test logging a trade
    entry = JournalEntry(
        id=None,
        symbol="AAPL",
        direction="LONG",
        timeframe="4hr",
        entry_price=185.50,
        stop_loss=183.00,
        target1=190.00,
        target2=195.00,
        risk_reward_t1=1.8,
        risk_reward_t2=3.8,
        signal="GREEN",
        confidence=72,
        bull_score=8,
        bear_score=3,
        setup_grade="A",
        notes="Strong bounce off VAL with RSI oversold"
    )
    
    entry_id = journal.log_trade(entry)
    print(f"✅ Logged trade #{entry_id}")
    
    # Get stats
    stats = journal.get_stats()
    print(f"\n📊 Stats: {json.dumps(stats, indent=2)}")
