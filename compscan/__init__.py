"""
Compression Reversal Scanner Package
=====================================
Scans for compression reversal setups ideal for options trading.
"""

from .compression_reversal import (
    CompressionReversalScanner,
    CompressionReversalSetup,
    SetupQuality,
    CompressionLevel,
    ProfileShape,
    format_setup_alert,
    quick_scan
)

__all__ = [
    'CompressionReversalScanner',
    'CompressionReversalSetup', 
    'SetupQuality',
    'CompressionLevel',
    'ProfileShape',
    'format_setup_alert',
    'quick_scan'
]
