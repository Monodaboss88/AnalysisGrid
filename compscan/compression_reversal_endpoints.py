"""
Compression Reversal API Endpoints
==================================
Flask endpoints for the compression reversal scanner.
Add these routes to unified_server.py

Author: Rob's Trading Systems
Version: 1.0.0
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
from typing import List, Dict, Any

# Import the scanner
from compression_reversal import (
    CompressionReversalScanner,
    CompressionReversalSetup,
    SetupQuality,
    format_setup_alert,
    quick_scan
)

# Create Blueprint
compression_reversal_bp = Blueprint('compression_reversal', __name__)

# Global scanner instance
_scanner = None

def get_scanner() -> CompressionReversalScanner:
    """Get or create scanner instance"""
    global _scanner
    if _scanner is None:
        _scanner = CompressionReversalScanner()
    return _scanner


# =============================================================================
# ENDPOINTS
# =============================================================================

@compression_reversal_bp.route('/api/compression-reversal/scan', methods=['POST'])
def scan_symbol():
    """
    Scan a single symbol for compression reversal setup
    
    Request Body:
        {
            "symbol": "AAPL",
            "days": 30,        // optional, default 30
            "interval": "1h"   // optional, default "1h"
        }
    
    Response:
        {
            "success": true,
            "setup": { ... setup details ... }
        }
    """
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', '').upper()
        days = data.get('days', 30)
        interval = data.get('interval', '1h')
        
        if not symbol:
            return jsonify({
                'success': False,
                'error': 'Symbol required'
            }), 400
        
        # Import data fetcher (adjust based on your setup)
        try:
            from finnhub_scanner_v2 import MarketScanner
            fetcher = MarketScanner()
            df = fetcher.fetch_candles(symbol, days=days, interval=interval)
        except ImportError:
            # Fallback to yfinance
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", interval=interval)
            df.columns = df.columns.str.lower()
        
        if df is None or len(df) < 50:
            return jsonify({
                'success': False,
                'error': f'Insufficient data for {symbol}'
            }), 400
        
        # Run scan
        scanner = get_scanner()
        setup = scanner.scan(df, symbol=symbol)
        
        return jsonify({
            'success': True,
            'setup': setup.to_dict(),
            'alert': format_setup_alert(setup)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@compression_reversal_bp.route('/api/compression-reversal/scan-watchlist', methods=['POST'])
def scan_watchlist():
    """
    Scan multiple symbols for compression reversal setups
    
    Request Body:
        {
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "min_quality": "B",  // optional: "A+", "A", "B", "C"
            "days": 30,
            "interval": "1h"
        }
    
    Response:
        {
            "success": true,
            "setups": [ ... list of setups ... ],
            "scanned": 10,
            "found": 3
        }
    """
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols', [])
        min_quality_str = data.get('min_quality', 'B')
        days = data.get('days', 30)
        interval = data.get('interval', '1h')
        
        if not symbols:
            return jsonify({
                'success': False,
                'error': 'Symbols list required'
            }), 400
        
        # Parse min quality
        quality_map = {
            'A+': SetupQuality.A_PLUS,
            'A': SetupQuality.A,
            'B': SetupQuality.B,
            'C': SetupQuality.C
        }
        min_quality = quality_map.get(min_quality_str, SetupQuality.B)
        
        # Import data fetcher
        try:
            from finnhub_scanner_v2 import MarketScanner
            fetcher = MarketScanner()
        except ImportError:
            fetcher = None
        
        scanner = get_scanner()
        setups = []
        errors = []
        
        for symbol in symbols:
            try:
                symbol = symbol.upper()
                
                if fetcher:
                    df = fetcher.fetch_candles(symbol, days=days, interval=interval)
                else:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=f"{days}d", interval=interval)
                    df.columns = df.columns.str.lower()
                
                if df is None or len(df) < 50:
                    errors.append(f"{symbol}: Insufficient data")
                    continue
                
                setup = scanner.scan(df, symbol=symbol)
                
                # Filter by quality
                if setup.setup_quality.tradeable:
                    quality_order = [SetupQuality.NO_SETUP, SetupQuality.C, SetupQuality.B, SetupQuality.A, SetupQuality.A_PLUS]
                    if quality_order.index(setup.setup_quality) >= quality_order.index(min_quality):
                        setups.append(setup.to_dict())
            
            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
        
        # Sort by score
        setups.sort(key=lambda x: x['setup_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'setups': setups,
            'scanned': len(symbols),
            'found': len(setups),
            'errors': errors if errors else None
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@compression_reversal_bp.route('/api/compression-reversal/criteria', methods=['GET'])
def get_criteria():
    """
    Get the setup criteria and parameters
    
    Response:
        {
            "criteria": { ... },
            "options_params": { ... }
        }
    """
    scanner = get_scanner()
    
    return jsonify({
        'success': True,
        'criteria': {
            'profile_shape': 'NORMAL (football)',
            'compression': 'EXTREME, HIGH, or MODERATE',
            'val_proximity_pct': scanner.val_proximity_pct,
            'rsi_target': scanner.rsi_target,
            'rsi_tolerance': scanner.rsi_tolerance,
            'reversal_candle': 'Hammer, Bullish Engulfing, or Doji'
        },
        'options_params': {
            'direction': 'CALL',
            'delta': scanner.delta,
            'min_dte': scanner.min_dte,
            'stop_loss_pct': scanner.stop_loss_pct,
            'target_move_pct': scanner.target_move_pct,
            'target_rsi': 72
        },
        'quality_levels': {
            'A+': '90+ score - textbook setup',
            'A': '80-89 score - strong setup',
            'B': '70-79 score - good setup',
            'C': '60-69 score - marginal',
            'NO': '< 60 score - criteria not met'
        }
    })


@compression_reversal_bp.route('/api/compression-reversal/configure', methods=['POST'])
def configure_scanner():
    """
    Update scanner parameters
    
    Request Body:
        {
            "val_proximity_pct": 0.5,
            "rsi_target": 37,
            "rsi_tolerance": 5,
            "stop_loss_pct": 12.5,
            "target_move_pct": 1.5,
            "delta": 0.65,
            "min_dte": 21
        }
    
    Response:
        {
            "success": true,
            "config": { ... current config ... }
        }
    """
    global _scanner
    
    try:
        data = request.get_json() or {}
        
        # Create new scanner with updated params
        _scanner = CompressionReversalScanner(
            val_proximity_pct=data.get('val_proximity_pct', 0.5),
            rsi_target=data.get('rsi_target', 37),
            rsi_tolerance=data.get('rsi_tolerance', 5),
            stop_loss_pct=data.get('stop_loss_pct', 12.5),
            target_move_pct=data.get('target_move_pct', 1.5),
            delta=data.get('delta', 0.65),
            min_dte=data.get('min_dte', 21)
        )
        
        return jsonify({
            'success': True,
            'config': {
                'val_proximity_pct': _scanner.val_proximity_pct,
                'rsi_target': _scanner.rsi_target,
                'rsi_tolerance': _scanner.rsi_tolerance,
                'stop_loss_pct': _scanner.stop_loss_pct,
                'target_move_pct': _scanner.target_move_pct,
                'delta': _scanner.delta,
                'min_dte': _scanner.min_dte
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def register_compression_reversal_routes(app):
    """
    Register compression reversal routes with Flask app
    
    Usage in unified_server.py:
        from compression_reversal_endpoints import register_compression_reversal_routes
        register_compression_reversal_routes(app)
    """
    app.register_blueprint(compression_reversal_bp)
    print("✅ Compression Reversal endpoints registered")


# =============================================================================
# STANDALONE TEST SERVER
# =============================================================================

if __name__ == "__main__":
    from flask import Flask
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    register_compression_reversal_routes(app)
    
    @app.route('/')
    def index():
        return jsonify({
            'service': 'Compression Reversal Scanner',
            'version': '1.0.0',
            'endpoints': [
                'POST /api/compression-reversal/scan',
                'POST /api/compression-reversal/scan-watchlist',
                'GET /api/compression-reversal/criteria',
                'POST /api/compression-reversal/configure'
            ]
        })
    
    print("\n🚀 Starting Compression Reversal Scanner API")
    print("   http://localhost:5050")
    print("\nEndpoints:")
    print("   POST /api/compression-reversal/scan")
    print("   POST /api/compression-reversal/scan-watchlist")
    print("   GET  /api/compression-reversal/criteria")
    print("   POST /api/compression-reversal/configure")
    
    app.run(host='0.0.0.0', port=5050, debug=True)
