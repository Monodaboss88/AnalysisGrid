"""
Trade Journal System
=====================
Log trades from scans with one click.
Track entries, exits, P&L, and learn from your history.

V2 UPDATE:
- Fib retracement fields added to schema (swing_high, swing_low, fib levels, zone, confluence)
- Fib-specific analytics: win rate by fib zone, golden zone performance, confluence edge
- Auto-migration for existing databases (adds fib columns without data loss)
- Squeeze/weekly context fields for V2 ecosystem integration

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
    """Single trade journal entry — V2 with Fib + context fields"""
    id: Optional[int]
    user_id: str  # Firebase UID for per-user storage
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
    
    # ═══════════════════════════════════════════════
    # V2: FIBONACCI CONTEXT AT ENTRY
    # ═══════════════════════════════════════════════
    fib_swing_high: float = 0           # Swing high used for fib calc
    fib_swing_low: float = 0            # Swing low used for fib calc
    fib_trend: str = ""                 # "UPTREND" or "DOWNTREND"
    fib_236: float = 0                  # Fib 23.6% level (active set)
    fib_382: float = 0                  # Fib 38.2% level
    fib_500: float = 0                  # Fib 50% level
    fib_618: float = 0                  # Fib 61.8% level (golden ratio)
    fib_786: float = 0                  # Fib 78.6% level
    fib_position: str = ""              # Human description of price vs fibs
    fib_zone: str = ""                  # Code: GOLDEN_ZONE, PULLBACK_ENTRY, etc.
    fib_confluence: str = ""            # VP+Fib confluences found
    fib_quality: str = ""               # STRONG, MODERATE, WEAK

    # V2: ADDITIONAL CONTEXT
    squeeze_active: bool = False        # Was a squeeze active at entry?
    squeeze_days: int = 0               # How many days into squeeze
    weekly_trend: str = ""              # Weekly candle trend at entry
    iv_regime: str = ""                 # IV regime at entry (LOW/NORMAL/HIGH/EXTREME)
    
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


# V2 columns to add during migration
_FIB_COLUMNS = [
    ("fib_swing_high", "REAL", 0),
    ("fib_swing_low", "REAL", 0),
    ("fib_trend", "TEXT", "''"),
    ("fib_236", "REAL", 0),
    ("fib_382", "REAL", 0),
    ("fib_500", "REAL", 0),
    ("fib_618", "REAL", 0),
    ("fib_786", "REAL", 0),
    ("fib_position", "TEXT", "''"),
    ("fib_zone", "TEXT", "''"),
    ("fib_confluence", "TEXT", "''"),
    ("fib_quality", "TEXT", "''"),
    ("squeeze_active", "INTEGER", 0),
    ("squeeze_days", "INTEGER", 0),
    ("weekly_trend", "TEXT", "''"),
    ("iv_regime", "TEXT", "''"),
]


class TradeJournal:
    """
    Trade journaling system with SQLite storage — V2 with Fib analytics
    """
    
    def __init__(self, db_path: str = "scanner_data/trade_journal.db"):
        """Initialize journal database"""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create journal tables and run migrations"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'anonymous',
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
                
                -- V2: Fibonacci context
                fib_swing_high REAL DEFAULT 0,
                fib_swing_low REAL DEFAULT 0,
                fib_trend TEXT DEFAULT '',
                fib_236 REAL DEFAULT 0,
                fib_382 REAL DEFAULT 0,
                fib_500 REAL DEFAULT 0,
                fib_618 REAL DEFAULT 0,
                fib_786 REAL DEFAULT 0,
                fib_position TEXT DEFAULT '',
                fib_zone TEXT DEFAULT '',
                fib_confluence TEXT DEFAULT '',
                fib_quality TEXT DEFAULT '',
                
                -- V2: Additional context
                squeeze_active INTEGER DEFAULT 0,
                squeeze_days INTEGER DEFAULT 0,
                weekly_trend TEXT DEFAULT '',
                iv_regime TEXT DEFAULT '',
                
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
        c.execute("CREATE INDEX IF NOT EXISTS idx_journal_user ON journal_entries(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_journal_fib_zone ON journal_entries(fib_zone)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_journal_fib_quality ON journal_entries(fib_quality)")
        
        # Migration: Add user_id column if it doesn't exist (for existing databases)
        self._migrate_column(c, "user_id", "TEXT NOT NULL DEFAULT 'anonymous'")
        
        # V2 Migration: Add fib + context columns
        for col_name, col_type, col_default in _FIB_COLUMNS:
            self._migrate_column(c, col_name, f"{col_type} DEFAULT {col_default}")
        
        conn.commit()
        conn.close()
    
    def _migrate_column(self, cursor, column_name: str, column_def: str):
        """Safely add a column if it doesn't exist"""
        try:
            cursor.execute(f"ALTER TABLE journal_entries ADD COLUMN {column_name} {column_def}")
            print(f"[TradeJournal] Migrated: Added {column_name} column")
        except:
            pass  # Column already exists
    
    def log_trade(self, entry: JournalEntry) -> int:
        """Log a new trade from scan results"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        entry.logged_at = entry.logged_at or datetime.now().isoformat()
        
        c.execute("""
            INSERT INTO journal_entries (
                user_id, symbol, direction, timeframe,
                entry_price, stop_loss, target1, target2,
                risk_reward_t1, risk_reward_t2, position_size, risk_amount,
                signal, confidence, bull_score, bear_score,
                ai_commentary, setup_grade,
                vah, poc, val, vwap, rsi,
                fib_swing_high, fib_swing_low, fib_trend,
                fib_236, fib_382, fib_500, fib_618, fib_786,
                fib_position, fib_zone, fib_confluence, fib_quality,
                squeeze_active, squeeze_days, weekly_trend, iv_regime,
                status, notes, tags, logged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.user_id, entry.symbol, entry.direction, entry.timeframe,
            entry.entry_price, entry.stop_loss, entry.target1, entry.target2,
            entry.risk_reward_t1, entry.risk_reward_t2, entry.position_size, entry.risk_amount,
            entry.signal, entry.confidence, entry.bull_score, entry.bear_score,
            entry.ai_commentary, entry.setup_grade,
            entry.vah, entry.poc, entry.val, entry.vwap, entry.rsi,
            entry.fib_swing_high, entry.fib_swing_low, entry.fib_trend,
            entry.fib_236, entry.fib_382, entry.fib_500, entry.fib_618, entry.fib_786,
            entry.fib_position, entry.fib_zone, entry.fib_confluence, entry.fib_quality,
            1 if entry.squeeze_active else 0, entry.squeeze_days,
            entry.weekly_trend, entry.iv_regime,
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
        user_id: str = "anonymous",
        status: Optional[str] = None,
        symbol: Optional[str] = None,
        days: int = 30,
        limit: int = 100
    ) -> List[Dict]:
        """Get trade entries with filters"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = "SELECT * FROM journal_entries WHERE user_id = ? AND logged_at > ?"
        params = [user_id, cutoff]
        
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
    
    def get_stats(self, user_id: str = "anonymous", days: int = 30) -> Dict:
        """Get trading statistics including fib analytics"""
        trades = self.get_trades(user_id=user_id, days=days, limit=1000)
        
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
        
        # ═══════════════════════════════════════════════
        # V2: FIB ANALYTICS
        # ═══════════════════════════════════════════════
        fib_stats = self._calculate_fib_stats(closed)
        
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
            "by_setup_grade": grade_stats,
            "fib_analytics": fib_stats,
        }
    
    def _calculate_fib_stats(self, closed_trades: List[Dict]) -> Dict:
        """
        Calculate Fibonacci-specific performance analytics.
        
        This is the KEY DATA that tells you:
        - Do trades in the Golden Zone actually win more?
        - Does VP+Fib confluence give you an edge?
        - Which fib zones produce the best R multiples?
        """
        # Filter trades that have fib data
        fib_trades = [t for t in closed_trades if t.get("fib_zone") and t["fib_zone"] != ""]
        
        if not fib_trades:
            return {"has_data": False, "message": "No trades with fib context yet"}
        
        # ── Win rate by fib zone ──
        zone_stats = {}
        for trade in fib_trades:
            zone = trade["fib_zone"]
            if zone not in zone_stats:
                zone_stats[zone] = {"wins": 0, "losses": 0, "total": 0, "r_values": []}
            
            zone_stats[zone]["total"] += 1
            if trade["status"] == "WIN":
                zone_stats[zone]["wins"] += 1
            elif trade["status"] == "LOSS":
                zone_stats[zone]["losses"] += 1
            
            if trade.get("pnl_r") is not None:
                zone_stats[zone]["r_values"].append(trade["pnl_r"])
        
        # Calculate rates per zone
        by_zone = {}
        for zone, data in zone_stats.items():
            closed_in_zone = data["wins"] + data["losses"]
            by_zone[zone] = {
                "total": data["total"],
                "wins": data["wins"],
                "losses": data["losses"],
                "win_rate": round(data["wins"] / closed_in_zone * 100, 1) if closed_in_zone > 0 else 0,
                "avg_r": round(sum(data["r_values"]) / len(data["r_values"]), 2) if data["r_values"] else 0,
            }
        
        # ── Golden Zone performance ──
        golden_trades = [t for t in fib_trades if t.get("fib_zone") == "GOLDEN_ZONE"]
        non_golden = [t for t in fib_trades if t.get("fib_zone") != "GOLDEN_ZONE"]
        
        golden_wins = len([t for t in golden_trades if t["status"] == "WIN"])
        golden_closed = len([t for t in golden_trades if t["status"] in ["WIN", "LOSS"]])
        non_golden_wins = len([t for t in non_golden if t["status"] == "WIN"])
        non_golden_closed = len([t for t in non_golden if t["status"] in ["WIN", "LOSS"]])
        
        golden_wr = round(golden_wins / golden_closed * 100, 1) if golden_closed > 0 else 0
        non_golden_wr = round(non_golden_wins / non_golden_closed * 100, 1) if non_golden_closed > 0 else 0
        
        golden_r = [t["pnl_r"] for t in golden_trades if t.get("pnl_r") is not None]
        non_golden_r = [t["pnl_r"] for t in non_golden if t.get("pnl_r") is not None]
        
        golden_zone = {
            "total_trades": len(golden_trades),
            "win_rate": golden_wr,
            "avg_r": round(sum(golden_r) / len(golden_r), 2) if golden_r else 0,
            "edge_vs_other": round(golden_wr - non_golden_wr, 1),
            "other_win_rate": non_golden_wr,
        }
        
        # ── Confluence edge ──
        conf_trades = [t for t in fib_trades if t.get("fib_confluence") and t["fib_confluence"] != ""]
        no_conf = [t for t in fib_trades if not t.get("fib_confluence") or t["fib_confluence"] == ""]
        
        conf_wins = len([t for t in conf_trades if t["status"] == "WIN"])
        conf_closed = len([t for t in conf_trades if t["status"] in ["WIN", "LOSS"]])
        no_conf_wins = len([t for t in no_conf if t["status"] == "WIN"])
        no_conf_closed = len([t for t in no_conf if t["status"] in ["WIN", "LOSS"]])
        
        conf_wr = round(conf_wins / conf_closed * 100, 1) if conf_closed > 0 else 0
        no_conf_wr = round(no_conf_wins / no_conf_closed * 100, 1) if no_conf_closed > 0 else 0
        
        conf_r = [t["pnl_r"] for t in conf_trades if t.get("pnl_r") is not None]
        no_conf_r = [t["pnl_r"] for t in no_conf if t.get("pnl_r") is not None]
        
        confluence_edge = {
            "with_confluence": {
                "total": len(conf_trades),
                "win_rate": conf_wr,
                "avg_r": round(sum(conf_r) / len(conf_r), 2) if conf_r else 0,
            },
            "without_confluence": {
                "total": len(no_conf),
                "win_rate": no_conf_wr,
                "avg_r": round(sum(no_conf_r) / len(no_conf_r), 2) if no_conf_r else 0,
            },
            "confluence_edge_pct": round(conf_wr - no_conf_wr, 1),
        }
        
        # ── Fib quality performance ──
        quality_stats = {}
        for quality in ["STRONG", "MODERATE", "WEAK"]:
            q_trades = [t for t in fib_trades if t.get("fib_quality") == quality]
            q_wins = len([t for t in q_trades if t["status"] == "WIN"])
            q_closed = len([t for t in q_trades if t["status"] in ["WIN", "LOSS"]])
            q_r = [t["pnl_r"] for t in q_trades if t.get("pnl_r") is not None]
            
            if q_trades:
                quality_stats[quality] = {
                    "total": len(q_trades),
                    "win_rate": round(q_wins / q_closed * 100, 1) if q_closed > 0 else 0,
                    "avg_r": round(sum(q_r) / len(q_r), 2) if q_r else 0,
                }
        
        # ── Squeeze context performance ──
        squeeze_trades = [t for t in closed_trades if t.get("squeeze_active") == 1]
        non_squeeze = [t for t in closed_trades if t.get("squeeze_active") != 1]
        
        sq_wins = len([t for t in squeeze_trades if t["status"] == "WIN"])
        sq_closed = len([t for t in squeeze_trades if t["status"] in ["WIN", "LOSS"]])
        nsq_wins = len([t for t in non_squeeze if t["status"] == "WIN"])
        nsq_closed = len([t for t in non_squeeze if t["status"] in ["WIN", "LOSS"]])
        
        squeeze_stats = {
            "with_squeeze": {
                "total": len(squeeze_trades),
                "win_rate": round(sq_wins / sq_closed * 100, 1) if sq_closed > 0 else 0,
            },
            "without_squeeze": {
                "total": len(non_squeeze),
                "win_rate": round(nsq_wins / nsq_closed * 100, 1) if nsq_closed > 0 else 0,
            },
        }
        
        return {
            "has_data": True,
            "fib_trades_count": len(fib_trades),
            "by_zone": by_zone,
            "golden_zone": golden_zone,
            "confluence_edge": confluence_edge,
            "by_quality": quality_stats,
            "squeeze_context": squeeze_stats,
        }
    
    def get_fib_report(self, user_id: str = "anonymous", days: int = 90) -> str:
        """
        Generate a human-readable Fib performance report.
        
        This is the report that tells you if Fibs are actually working
        for your trading or if they're just noise.
        """
        stats = self.get_stats(user_id=user_id, days=days)
        fib = stats.get("fib_analytics", {})
        
        if not fib.get("has_data"):
            return "📐 No trades with Fib context yet. Log trades from Complete Analysis to start tracking."
        
        lines = []
        lines.append("=" * 65)
        lines.append("📐 FIBONACCI PERFORMANCE REPORT")
        lines.append(f"   Period: {days} days | Fib Trades: {fib['fib_trades_count']}")
        lines.append("=" * 65)
        
        # Golden Zone
        gz = fib.get("golden_zone", {})
        if gz.get("total_trades", 0) > 0:
            lines.append(f"\n⭐ GOLDEN ZONE (50%-61.8%):")
            lines.append(f"   Trades: {gz['total_trades']} | Win Rate: {gz['win_rate']}% | Avg R: {gz['avg_r']}")
            edge = gz.get("edge_vs_other", 0)
            if edge > 0:
                lines.append(f"   🟢 EDGE: +{edge}% vs non-golden trades ({gz['other_win_rate']}%)")
            elif edge < -5:
                lines.append(f"   🔴 UNDERPERFORMING: {edge}% vs other zones ({gz['other_win_rate']}%)")
            else:
                lines.append(f"   🟡 NEUTRAL: {edge:+.1f}% vs other zones")
        
        # Confluence edge
        ce = fib.get("confluence_edge", {})
        with_c = ce.get("with_confluence", {})
        without_c = ce.get("without_confluence", {})
        if with_c.get("total", 0) > 0:
            lines.append(f"\n🎯 VP+FIB CONFLUENCE EDGE:")
            lines.append(f"   With Confluence:    {with_c['total']} trades | {with_c['win_rate']}% WR | {with_c['avg_r']}R avg")
            lines.append(f"   Without Confluence: {without_c['total']} trades | {without_c['win_rate']}% WR | {without_c['avg_r']}R avg")
            edge = ce.get("confluence_edge_pct", 0)
            if edge > 5:
                lines.append(f"   🟢 CONFLUENCE WORKS: +{edge}% edge")
            elif edge < -5:
                lines.append(f"   🔴 NO EDGE from confluence ({edge:+.1f}%)")
            else:
                lines.append(f"   🟡 INCONCLUSIVE ({edge:+.1f}%)")
        
        # By zone
        by_zone = fib.get("by_zone", {})
        if by_zone:
            lines.append(f"\n📊 WIN RATE BY FIB ZONE:")
            for zone, data in sorted(by_zone.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True):
                wr = data["win_rate"]
                emoji = "🟢" if wr >= 60 else "🔴" if wr < 40 else "🟡"
                lines.append(f"   {emoji} {zone:20s}: {wr:5.1f}% WR ({data['total']} trades, {data['avg_r']:+.2f}R)")
        
        # By quality
        by_q = fib.get("by_quality", {})
        if by_q:
            lines.append(f"\n📏 SWING QUALITY IMPACT:")
            for quality in ["STRONG", "MODERATE", "WEAK"]:
                if quality in by_q:
                    q = by_q[quality]
                    lines.append(f"   {quality:10s}: {q['win_rate']}% WR ({q['total']} trades, {q['avg_r']:+.2f}R)")
        
        # Squeeze context
        sq = fib.get("squeeze_context", {})
        ws = sq.get("with_squeeze", {})
        ns = sq.get("without_squeeze", {})
        if ws.get("total", 0) > 0:
            lines.append(f"\n🔧 SQUEEZE CONTEXT:")
            lines.append(f"   With Squeeze:    {ws['total']} trades | {ws['win_rate']}% WR")
            lines.append(f"   Without Squeeze: {ns['total']} trades | {ns['win_rate']}% WR")
        
        lines.append("\n" + "=" * 65)
        return "\n".join(lines)
    
    def export_journal(self, user_id: str = "anonymous", format: str = "json", days: int = 365) -> str:
        """Export journal for external analysis"""
        trades = self.get_trades(user_id=user_id, days=days, limit=10000)
        
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
    
    # Test logging a trade with fib context
    entry = JournalEntry(
        id=None,
        user_id="test_user",
        symbol="META",
        direction="LONG",
        timeframe="4hr",
        entry_price=620.00,
        stop_loss=613.00,
        target1=647.00,
        target2=665.00,
        risk_reward_t1=3.86,
        risk_reward_t2=6.43,
        signal="GREEN",
        confidence=72,
        bull_score=8,
        bear_score=3,
        setup_grade="A",
        vah=667.72,
        poc=660.40,
        val=647.22,
        # V2: Fib context
        fib_swing_high=670.50,
        fib_swing_low=615.20,
        fib_trend="DOWNTREND",
        fib_236=628.25,
        fib_382=636.33,
        fib_500=642.85,
        fib_618=649.37,
        fib_786=658.64,
        fib_position="Fib 50%-61.8% GOLDEN ZONE (best long entry)",
        fib_zone="GOLDEN_ZONE",
        fib_confluence="VAL ≈ Fib 61.8% at $647.22 (STRONG)",
        fib_quality="STRONG",
        squeeze_active=True,
        squeeze_days=5,
        weekly_trend="BULLISH",
        iv_regime="NORMAL",
        notes="Strong bounce off VAL with RSI oversold + golden zone entry"
    )
    
    entry_id = journal.log_trade(entry)
    print(f"✅ Logged trade #{entry_id} with fib context")
    
    # Get stats
    stats = journal.get_stats(user_id="test_user")
    print(f"\n📊 Stats: {json.dumps(stats, indent=2)}")
    
    # Fib report
    print(journal.get_fib_report(user_id="test_user"))
