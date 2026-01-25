"""
Firestore Data Store
====================
Per-user persistent storage for alerts, trades, and watchlists
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Firebase Admin SDK
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    firebase_available = True
except ImportError:
    firebase_available = False
    print("⚠️ firebase-admin not installed")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UserAlert:
    symbol: str
    level: float
    direction: str  # 'above' or 'below'
    action: str     # 'LONG', 'SHORT', 'ALERT', 'EXIT'
    note: str = ""
    created_at: str = ""
    triggered: bool = False
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class UserTrade:
    symbol: str
    direction: str
    entry: float
    stop: float
    target: float
    target2: float = 0
    timeframe: str = "swing"
    signal: str = "YELLOW"
    confidence: int = 50
    notes: str = ""
    status: str = "pending"  # pending, active, closed
    exit_price: float = 0
    pnl: float = 0
    created_at: str = ""
    closed_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


# =============================================================================
# FIRESTORE MANAGER
# =============================================================================

class FirestoreManager:
    """Manages per-user data in Firestore"""
    
    def __init__(self):
        self.db = None
        self._init_firestore()
    
    def _init_firestore(self):
        """Initialize Firestore connection"""
        if not firebase_available:
            print("❌ Firestore not available (firebase-admin not installed)")
            return
        
        try:
            # Check if Firebase app is already initialized
            try:
                app = firebase_admin.get_app()
            except ValueError:
                # No app initialized yet - try to initialize
                if os.getenv('FIREBASE_SERVICE_ACCOUNT'):
                    cred_dict = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT'))
                    cred = credentials.Certificate(cred_dict)
                    firebase_admin.initialize_app(cred)
                else:
                    print("⚠️ No FIREBASE_SERVICE_ACCOUNT env var found")
                    return
            
            self.db = firestore.client()
            print("✅ Firestore connected")
        except Exception as e:
            print(f"❌ Firestore init error: {e}")
    
    def is_available(self) -> bool:
        return self.db is not None
    
    # =========================================================================
    # ALERTS
    # =========================================================================
    
    def get_alerts(self, user_id: str, symbol: str = None) -> List[Dict]:
        """Get all alerts for a user"""
        if not self.db:
            return []
        
        try:
            alerts_ref = self.db.collection('users').document(user_id).collection('alerts')
            
            if symbol:
                query = alerts_ref.where('symbol', '==', symbol.upper())
            else:
                query = alerts_ref
            
            docs = query.stream()
            alerts = []
            for doc in docs:
                alert = doc.to_dict()
                alert['id'] = doc.id
                alerts.append(alert)
            
            return alerts
        except Exception as e:
            print(f"Error getting alerts: {e}")
            return []
    
    def add_alert(self, user_id: str, alert: UserAlert) -> Optional[Dict]:
        """Add an alert for a user"""
        if not self.db:
            return None
        
        try:
            alerts_ref = self.db.collection('users').document(user_id).collection('alerts')
            alert_dict = asdict(alert)
            alert_dict['symbol'] = alert_dict['symbol'].upper()
            
            doc_ref = alerts_ref.add(alert_dict)
            alert_dict['id'] = doc_ref[1].id
            return alert_dict
        except Exception as e:
            print(f"Error adding alert: {e}")
            return None
    
    def delete_alert(self, user_id: str, symbol: str, level: float) -> bool:
        """Delete an alert by symbol and level"""
        if not self.db:
            return False
        
        try:
            alerts_ref = self.db.collection('users').document(user_id).collection('alerts')
            query = alerts_ref.where('symbol', '==', symbol.upper()).where('level', '==', level)
            
            docs = query.stream()
            deleted = False
            for doc in docs:
                doc.reference.delete()
                deleted = True
            
            return deleted
        except Exception as e:
            print(f"Error deleting alert: {e}")
            return False
    
    def delete_alert_by_id(self, user_id: str, alert_id: str) -> bool:
        """Delete an alert by ID"""
        if not self.db:
            return False
        
        try:
            self.db.collection('users').document(user_id).collection('alerts').document(alert_id).delete()
            return True
        except Exception as e:
            print(f"Error deleting alert: {e}")
            return False
    
    # =========================================================================
    # TRADES
    # =========================================================================
    
    def get_trades(self, user_id: str, symbol: str = None, status: str = None) -> List[Dict]:
        """Get all trades for a user"""
        if not self.db:
            return []
        
        try:
            trades_ref = self.db.collection('users').document(user_id).collection('trades')
            query = trades_ref
            
            if symbol:
                query = query.where('symbol', '==', symbol.upper())
            if status:
                query = query.where('status', '==', status)
            
            docs = query.order_by('created_at', direction=firestore.Query.DESCENDING).stream()
            trades = []
            for doc in docs:
                trade = doc.to_dict()
                trade['id'] = doc.id
                trades.append(trade)
            
            return trades
        except Exception as e:
            print(f"Error getting trades: {e}")
            return []
    
    def add_trade(self, user_id: str, trade: UserTrade) -> Optional[Dict]:
        """Add a trade for a user"""
        if not self.db:
            return None
        
        try:
            trades_ref = self.db.collection('users').document(user_id).collection('trades')
            trade_dict = asdict(trade)
            trade_dict['symbol'] = trade_dict['symbol'].upper()
            
            doc_ref = trades_ref.add(trade_dict)
            trade_dict['id'] = doc_ref[1].id
            return trade_dict
        except Exception as e:
            print(f"Error adding trade: {e}")
            return None
    
    def update_trade(self, user_id: str, trade_id: str, updates: Dict) -> Optional[Dict]:
        """Update a trade"""
        if not self.db:
            return None
        
        try:
            trade_ref = self.db.collection('users').document(user_id).collection('trades').document(trade_id)
            trade_ref.update(updates)
            
            updated = trade_ref.get().to_dict()
            updated['id'] = trade_id
            return updated
        except Exception as e:
            print(f"Error updating trade: {e}")
            return None
    
    def close_trade(self, user_id: str, trade_id: str, exit_price: float, status: str = "closed") -> Optional[Dict]:
        """Close a trade with exit price"""
        if not self.db:
            return None
        
        try:
            trade_ref = self.db.collection('users').document(user_id).collection('trades').document(trade_id)
            trade = trade_ref.get().to_dict()
            
            if not trade:
                return None
            
            # Calculate PnL
            entry = trade.get('entry', 0)
            direction = trade.get('direction', 'LONG')
            
            if direction.upper() == 'LONG':
                pnl = exit_price - entry
            else:
                pnl = entry - exit_price
            
            updates = {
                'status': status,
                'exit_price': exit_price,
                'pnl': pnl,
                'closed_at': datetime.now().isoformat()
            }
            
            trade_ref.update(updates)
            trade.update(updates)
            trade['id'] = trade_id
            return trade
        except Exception as e:
            print(f"Error closing trade: {e}")
            return None
    
    def get_trade_stats(self, user_id: str) -> Dict:
        """Get trading statistics for a user"""
        trades = self.get_trades(user_id, status="closed")
        
        if not trades:
            return {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0
            }
        
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) < 0]
        
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        avg_win = sum(t.get('pnl', 0) for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.get('pnl', 0) for t in losses) / len(losses) if losses else 0
        
        return {
            "total": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(trades) * 100) if trades else 0,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss
        }

    # =========================================================================
    # REPORTS (Auto-generated analysis reports)
    # =========================================================================
    
    def save_report(self, symbol: str, date_str: str, content: str, report_type: str = "analysis") -> Optional[str]:
        """Save an auto-generated report to Firestore"""
        if not self.db:
            return None
        
        try:
            doc_id = f"{symbol}_{date_str}_{datetime.now().strftime('%H%M%S')}"
            self.db.collection('reports').document(doc_id).set({
                'symbol': symbol.upper(),
                'date': date_str,
                'content': content,
                'type': report_type,
                'created_at': datetime.now().isoformat()
            })
            print(f"✅ Report saved: {doc_id}")
            return doc_id
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            return None
    
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
            reports = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                reports.append(data)
            
            return reports
        except Exception as e:
            print(f"❌ Error getting reports: {e}")
            return []
    
    def get_report_content(self, doc_id: str) -> Optional[str]:
        """Get a specific report's content"""
        if not self.db:
            return None
        
        try:
            doc = self.db.collection('reports').document(doc_id).get()
            if doc.exists:
                return doc.to_dict().get('content')
            return None
        except Exception as e:
            print(f"❌ Error getting report content: {e}")
            return None


# Global instance
firestore_manager = FirestoreManager()

def get_firestore() -> FirestoreManager:
    """Get the Firestore manager instance"""
    return firestore_manager
