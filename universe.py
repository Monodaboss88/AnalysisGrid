"""
Universe — Centralized Symbol Watchlists
==========================================
Single source of truth for all stock symbol lists across the scanner system.
Import from here instead of hardcoding symbols in individual scanner files.

Usage:
    from universe import TECH, MEGA, ETFS, MEME, ALL_SYMBOLS
    from universe import get_universe, list_universes
"""

from typing import Dict, List


# =============================================================================
# CORE UNIVERSES (used by full_scan_discord, push_scan_discord, simple.html)
# =============================================================================

TECH = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    "ORCL", "ADBE", "NFLX", "PYPL", "SQ", "SHOP", "SNOW", "NET", "DDOG", "MDB",
    "AVGO", "MU", "PLTR", "APP",
]

MEGA = [
    "BRK.B", "UNH", "JNJ", "V", "JPM", "WMT", "PG", "MA", "HD", "DIS",
    "BAC", "XOM", "PFE", "LLY", "ABBV", "MRK",
]

ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "GLD", "SLV", "TLT",
    "SMH", "ARKK", "SOXX", "XBI", "VXX", "SQQQ", "TQQQ", "IBIT",
]

MEME = [
    "GME", "AMC", "SOFI", "RIVN", "LCID", "NIO", "HOOD", "COIN", "MARA",
    "RIOT", "DKNG", "SPCE", "TLRY", "MSTR",
]

# Deduplicated full universe
ALL_SYMBOLS = list(dict.fromkeys(TECH + MEGA + ETFS + MEME))


# =============================================================================
# ALPHA SCANNER UNIVERSES
# =============================================================================

ALPHA_TECH = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "CRM", "NFLX",
    "AMD", "QCOM", "ORCL", "ADBE", "INTC", "MU", "NOW", "PANW",
]

SEMIS = [
    "NVDA", "AMD", "AVGO", "QCOM", "INTC", "MU", "TSM", "MRVL", "LRCX", "KLAC",
    "AMAT", "ASML", "ON", "NXPI", "TXN",
]

MOMENTUM = [
    "PLTR", "SMCI", "MSTR", "COIN", "RKLB", "APP", "HOOD", "AFRM", "IONQ", "RDDT",
    "SOFI", "RIVN", "LCID", "ARM", "CRWD",
]

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

ALPHA_ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "SMH", "ARKK", "TQQQ"]

ALPHA_UNIVERSES = {
    "tech": ALPHA_TECH,
    "semis": SEMIS,
    "momentum": MOMENTUM,
    "etfs": ALPHA_ETFS,
    "mag7": MAG7,
    "all": list(set(ALPHA_TECH + SEMIS + MOMENTUM + ALPHA_ETFS)),
}


# =============================================================================
# THEMATIC PRESETS (options flow, buffett, etc.)
# =============================================================================

OPTIONS_PRESETS = {
    "mega_tech":  ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "sp500_top":  ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "JPM"],
    "meme_watch": ["GME", "AMC", "PLTR", "SOFI", "RIVN", "NIO", "BB", "MARA", "COIN", "HOOD"],
    "etfs":       ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "GLD", "SLV", "TLT"],
    "earnings":   ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "CRM", "AMD"],
    "finance":    ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "AXP", "V"],
}

BUFFETT_PRESETS = {
    "mega_cap":   ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"],
    "blue_chip":  ["JNJ", "PG", "KO", "PEP", "WMT", "MCD", "HD", "UNH", "ABBV", "MRK"],
    "growth":     ["CRM", "ADBE", "NOW", "PANW", "SNOW", "DDOG", "NET", "CRWD", "ZS", "SHOP"],
    "finance":    ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "AXP", "USB"],
    "energy":     ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "PSX", "VLO", "DVN"],
    "healthcare": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY"],
    "reits":      ["O", "AMT", "PLD", "CCI", "SPG", "EQIX", "DLR", "PSA", "WELL", "AVB"],
}


# =============================================================================
# AUTO SCANNER DEFAULTS
# =============================================================================

AUTO_SCANNER_DEFAULTS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "META", "TSLA",
    "AMZN", "GOOGL", "NFLX", "CRM", "AVGO", "MU", "COIN",
    "BA", "JPM", "GS", "XOM", "UNH", "V", "WMT", "HD", "DIS",
]


# =============================================================================
# HELPERS
# =============================================================================

def get_universe(name: str) -> List[str]:
    """Get a named universe. Falls back to ALL_SYMBOLS if not found."""
    all_universes = {
        "tech": TECH,
        "mega": MEGA,
        "etfs": ETFS,
        "meme": MEME,
        "all": ALL_SYMBOLS,
        "semis": SEMIS,
        "momentum": MOMENTUM,
        "mag7": MAG7,
        "auto_defaults": AUTO_SCANNER_DEFAULTS,
    }
    # Also check alpha and thematic presets
    all_universes.update(ALPHA_UNIVERSES)
    all_universes.update({f"options_{k}": v for k, v in OPTIONS_PRESETS.items()})
    all_universes.update({f"buffett_{k}": v for k, v in BUFFETT_PRESETS.items()})

    return all_universes.get(name.lower(), ALL_SYMBOLS)


def list_universes() -> Dict[str, int]:
    """Return dict of universe names and their symbol counts."""
    return {
        "tech": len(TECH),
        "mega": len(MEGA),
        "etfs": len(ETFS),
        "meme": len(MEME),
        "all": len(ALL_SYMBOLS),
        "semis": len(SEMIS),
        "momentum": len(MOMENTUM),
        "mag7": len(MAG7),
        "auto_defaults": len(AUTO_SCANNER_DEFAULTS),
    }
