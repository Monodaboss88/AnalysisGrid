"""
Firestore Data Store
====================
Per-user persistent storage for alerts, trades, and watchlists.
Supports org-scoped paths: /orgs/{orgId}/{collection} for enterprise users,
or /users/{uid}/{collection} for personal users (default).
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
            # Use shared Firebase initializer (handles service account + ADC)
            from firebase_init import init_firebase_app
            app = init_firebase_app()
            if not app:
                print("⚠️ Firebase not initialized — Firestore unavailable")
                return
            
            self.db = firestore.client()
            print("✅ Firestore connected")
        except Exception as e:
            print(f"❌ Firestore init error: {e}")
    
    def is_available(self) -> bool:
        return self.db is not None

    # =========================================================================
    # ORG-SCOPED PATH RESOLUTION
    # =========================================================================

    def _col(self, user_id: str, collection: str, org_id: str = None):
        """Return the correct Firestore collection reference.
        
        org_id=None or "personal" → users/{uid}/{collection}
        org_id="acme"            → orgs/acme/{collection}
        
        For org paths, documents include a 'uid' field for per-user filtering.
        """
        if org_id and org_id != "personal":
            return self.db.collection('orgs').document(org_id).collection(collection)
        return self.db.collection('users').document(user_id).collection(collection)

    def _doc(self, user_id: str, collection: str, doc_id: str, org_id: str = None):
        """Return a specific document reference (org or personal)."""
        return self._col(user_id, collection, org_id).document(doc_id)
    
    # =========================================================================
    # ALERTS
    # =========================================================================
    
    def get_alerts(self, user_id: str, symbol: str = None, org_id: str = None) -> List[Dict]:
        """Get alerts for a user (or org-wide if org_id provided)"""
        if not self.db:
            return []
        
        try:
            alerts_ref = self._col(user_id, 'alerts', org_id)
            query = alerts_ref

            if symbol:
                query = query.where('symbol', '==', symbol.upper())
            # In org mode, optionally filter by uid for user's own alerts
            if org_id and org_id != 'personal':
                query = query.where('uid', '==', user_id)
            
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
    
    def add_alert(self, user_id: str, alert: UserAlert, org_id: str = None) -> Optional[Dict]:
        """Add an alert for a user"""
        if not self.db:
            return None
        
        try:
            alerts_ref = self._col(user_id, 'alerts', org_id)
            alert_dict = asdict(alert)
            alert_dict['symbol'] = alert_dict['symbol'].upper()
            if org_id and org_id != 'personal':
                alert_dict['uid'] = user_id
            
            doc_ref = alerts_ref.add(alert_dict)
            alert_dict['id'] = doc_ref[1].id
            return alert_dict
        except Exception as e:
            print(f"Error adding alert: {e}")
            return None
    
    def delete_alert(self, user_id: str, symbol: str, level: float, org_id: str = None) -> bool:
        """Delete an alert by symbol and level"""
        if not self.db:
            return False
        
        try:
            alerts_ref = self._col(user_id, 'alerts', org_id)
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
    
    def delete_alert_by_id(self, user_id: str, alert_id: str, org_id: str = None) -> bool:
        """Delete an alert by ID"""
        if not self.db:
            return False
        
        try:
            self._doc(user_id, 'alerts', alert_id, org_id).delete()
            return True
        except Exception as e:
            print(f"Error deleting alert: {e}")
            return False
    
    # =========================================================================
    # TRADES
    # =========================================================================
    
    def get_trades(self, user_id: str, symbol: str = None, status: str = None, org_id: str = None) -> List[Dict]:
        """Get trades for a user (or org-scoped)"""
        if not self.db:
            return []
        
        try:
            trades_ref = self._col(user_id, 'trades', org_id)
            query = trades_ref
            
            if symbol:
                query = query.where('symbol', '==', symbol.upper())
            if status:
                query = query.where('status', '==', status)
            # In org mode, filter by uid
            if org_id and org_id != 'personal':
                query = query.where('uid', '==', user_id)
            
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
    
    def add_trade(self, user_id: str, trade: UserTrade, org_id: str = None) -> Optional[Dict]:
        """Add a trade for a user"""
        if not self.db:
            return None
        
        try:
            trades_ref = self._col(user_id, 'trades', org_id)
            trade_dict = asdict(trade)
            trade_dict['symbol'] = trade_dict['symbol'].upper()
            if org_id and org_id != 'personal':
                trade_dict['uid'] = user_id
            
            doc_ref = trades_ref.add(trade_dict)
            trade_dict['id'] = doc_ref[1].id
            return trade_dict
        except Exception as e:
            print(f"Error adding trade: {e}")
            return None
    
    def update_trade(self, user_id: str, trade_id: str, updates: Dict, org_id: str = None) -> Optional[Dict]:
        """Update a trade"""
        if not self.db:
            return None
        
        try:
            trade_ref = self._doc(user_id, 'trades', trade_id, org_id)
            trade_ref.update(updates)
            
            updated = trade_ref.get().to_dict()
            updated['id'] = trade_id
            return updated
        except Exception as e:
            print(f"Error updating trade: {e}")
            return None
    
    def close_trade(self, user_id: str, trade_id: str, exit_price: float, status: str = "closed", org_id: str = None) -> Optional[Dict]:
        """Close a trade with exit price"""
        if not self.db:
            return None
        
        try:
            trade_ref = self._doc(user_id, 'trades', trade_id, org_id)
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

    # =========================================================================
    # AI SUGGESTIONS (Persisted Claude/GPT outputs for learning)
    # =========================================================================

    def save_ai_suggestion(self, symbol: str, suggestion_type: str, content: str,
                           metadata: Dict = None) -> Optional[str]:
        """
        Save an AI suggestion/commentary to Firestore.
        
        Args:
            symbol: Ticker symbol
            suggestion_type: 'mtf_plan', 'commentary', 'quick_commentary',
                             'full_analysis', 'trade_review', 'news_analysis',
                             'rule_explanation', 'rule_based_fallback'
            content: The AI-generated text
            metadata: Extra context (direction, confidence, model, etc.)
        """
        if not self.db:
            return None
        
        try:
            ts = datetime.now()
            doc_id = f"{symbol}_{suggestion_type}_{ts.strftime('%Y%m%d_%H%M%S')}"
            doc = {
                'symbol': symbol.upper(),
                'type': suggestion_type,
                'content': content,
                'metadata': metadata or {},
                'created_at': ts.isoformat(),
                'date': ts.strftime('%Y-%m-%d')
            }
            self.db.collection('ai_suggestions').document(doc_id).set(doc)
            return doc_id
        except Exception as e:
            print(f"⚠️ Error saving AI suggestion: {e}")
            return None

    def get_ai_suggestions(self, symbol: str = None, suggestion_type: str = None,
                           limit: int = 50) -> List[Dict]:
        """
        Retrieve past AI suggestions.
        
        Args:
            symbol: Filter by ticker (optional)
            suggestion_type: Filter by type (optional)
            limit: Max results (default 50)
        """
        if not self.db:
            return []
        
        try:
            query = self.db.collection('ai_suggestions')
            
            if symbol:
                query = query.where('symbol', '==', symbol.upper())
            if suggestion_type:
                query = query.where('type', '==', suggestion_type)
            
            query = query.order_by('created_at', direction=firestore.Query.DESCENDING).limit(limit)
            
            docs = query.stream()
            suggestions = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                suggestions.append(data)
            
            return suggestions
        except Exception as e:
            print(f"⚠️ Error getting AI suggestions: {e}")
            return []

    def get_ai_suggestion_content(self, doc_id: str) -> Optional[Dict]:
        """Get a specific AI suggestion by ID"""
        if not self.db:
            return None
        
        try:
            doc = self.db.collection('ai_suggestions').document(doc_id).get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
            return None
        except Exception as e:
            print(f"⚠️ Error getting AI suggestion: {e}")
            return None

    # =========================================================================
    # ORG-WIDE READS (admin/manager view — all users in the org)
    # =========================================================================

    def get_org_trades(self, org_id: str, symbol: str = None, status: str = None,
                       limit: int = 200) -> List[Dict]:
        """Get ALL trades across an org (not filtered by uid)."""
        if not self.db or not org_id or org_id == "personal":
            return []
        try:
            ref = self.db.collection('orgs').document(org_id).collection('trades')
            query = ref
            if symbol:
                query = query.where('symbol', '==', symbol.upper())
            if status:
                query = query.where('status', '==', status)
            query = query.order_by('created_at', direction=firestore.Query.DESCENDING).limit(limit)
            trades = []
            for doc in query.stream():
                d = doc.to_dict()
                d['id'] = doc.id
                trades.append(d)
            return trades
        except Exception as e:
            print(f"Error getting org trades: {e}")
            return []

    def get_org_alerts(self, org_id: str, symbol: str = None,
                       limit: int = 500) -> List[Dict]:
        """Get ALL alerts across an org."""
        if not self.db or not org_id or org_id == "personal":
            return []
        try:
            ref = self.db.collection('orgs').document(org_id).collection('alerts')
            query = ref
            if symbol:
                query = query.where('symbol', '==', symbol.upper())
            alerts = []
            for doc in query.limit(limit).stream():
                d = doc.to_dict()
                d['id'] = doc.id
                alerts.append(d)
            return alerts
        except Exception as e:
            print(f"Error getting org alerts: {e}")
            return []

    # =========================================================================
    # DATA MIGRATION (personal → org)
    # =========================================================================

    def migrate_user_to_org(self, user_id: str, org_id: str,
                            collections: List[str] = None) -> Dict:
        """Copy a user's personal data into org-scoped collections.
        
        Args:
            user_id: Firebase UID
            org_id: Target org ID
            collections: List of collections to migrate (default: alerts, trades)
        
        Returns:
            Dict with counts per collection: {"alerts": 12, "trades": 5}
        """
        if not self.db or not org_id or org_id == "personal":
            return {"error": "Invalid org_id"}

        if collections is None:
            collections = ["alerts", "trades"]

        result = {}
        for col_name in collections:
            try:
                # Read from personal path
                src = self.db.collection('users').document(user_id).collection(col_name)
                dst = self.db.collection('orgs').document(org_id).collection(col_name)
                
                count = 0
                for doc in src.stream():
                    data = doc.to_dict()
                    data['uid'] = user_id  # Tag with user ID
                    data['migrated_from'] = f"users/{user_id}/{col_name}/{doc.id}"
                    data['migrated_at'] = datetime.now().isoformat()
                    # Use same doc ID to avoid duplicates on re-run
                    dst.document(doc.id).set(data, merge=True)
                    count += 1
                
                result[col_name] = count
            except Exception as e:
                result[col_name] = f"error: {e}"
        
        return result

    def get_migration_status(self, user_id: str, org_id: str) -> Dict:
        """Check how much data a user has vs what's already migrated."""
        if not self.db:
            return {}
        
        status = {}
        for col_name in ["alerts", "trades"]:
            try:
                personal_count = len(list(
                    self.db.collection('users').document(user_id)
                    .collection(col_name).select([]).stream()
                ))
                org_count = len(list(
                    self.db.collection('orgs').document(org_id)
                    .collection(col_name)
                    .where('uid', '==', user_id).select([]).stream()
                ))
                status[col_name] = {
                    "personal": personal_count,
                    "org": org_count,
                    "migrated": org_count >= personal_count
                }
            except Exception as e:
                status[col_name] = {"error": str(e)}
        
        return status


# Global instance (lazy initialization)
_firestore_manager = None

def get_firestore() -> FirestoreManager:
    """Get the Firestore manager instance (lazy initialization)"""
    global _firestore_manager
    if _firestore_manager is None:
        _firestore_manager = FirestoreManager()
    # If db is None, try to reinitialize (env var might not have been loaded initially)
    if _firestore_manager.db is None:
        _firestore_manager._init_firestore()
    return _firestore_manager
