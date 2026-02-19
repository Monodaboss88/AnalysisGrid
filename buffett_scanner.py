"""
Buffett Blood Scanner — Value Discovery Engine
===============================================
Finds fundamentally strong companies trading at distressed prices.

"Be fearful when others are greedy, be greedy when others are fearful."

Combines Buffett-style fundamental analysis with crisis-level price
detection to surface deep-value opportunities when quality meets capitulation.

Scoring Model:
  Fundamental Score (0-100): P/E, P/B, ROE, Margins, Debt, FCF Yield
  Blood Score (0-100):       52w Drawdown, Momentum, SMA position, Volume spike
  Composite Score:           50/50 blend + cross-bonus when both high

Author: Rob's Trading Systems
"""

import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from datetime import datetime, timezone


# ── Preset Watchlists ──
PRESETS = {
    "mega_cap":  ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"],
    "blue_chip": ["JNJ", "PG", "KO", "PEP", "WMT", "MCD", "HD", "UNH", "ABBV", "MRK"],
    "growth":    ["CRM", "ADBE", "NOW", "PANW", "SNOW", "DDOG", "NET", "CRWD", "ZS", "SHOP"],
    "finance":   ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "AXP", "USB"],
    "energy":    ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "PSX", "VLO", "DVN"],
    "healthcare":["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY"],
    "reits":     ["O", "AMT", "PLD", "CCI", "SPG", "EQIX", "DLR", "PSA", "WELL", "AVB"],
}


def scan_tickers(symbols: List[str], max_workers: int = 5) -> Dict:
    """Scan multiple tickers in parallel for value + blood metrics."""
    results = []
    errors = []

    clean = [s.strip().upper() for s in symbols if s.strip()]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_scan_single, sym): sym for sym in clean}
        for future in as_completed(future_map):
            sym = future_map[future]
            try:
                r = future.result(timeout=30)
                if r.get("error"):
                    errors.append({"ticker": sym, "error": r["error"]})
                else:
                    results.append(r)
            except Exception as e:
                errors.append({"ticker": sym, "error": str(e)})

    results.sort(key=lambda r: r.get("compositeScore", 0), reverse=True)

    return {
        "results": results,
        "errors": errors,
        "meta": {
            "scanned": len(clean),
            "returned": len(results),
            "failed": len(errors),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


# ── Single Ticker Scan ──

def _scan_single(symbol: str) -> Dict:
    import yfinance as yf

    try:
        t = yf.Ticker(symbol)
        info = t.info or {}

        price = _sf(info.get("currentPrice")) or _sf(info.get("regularMarketPrice"))
        if not price:
            return {"ticker": symbol, "error": "No price data", "compositeScore": 0}

        mcap = info.get("marketCap") or 0

        r: Dict = {
            "ticker": symbol,
            "name": info.get("longName") or info.get("shortName") or symbol,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "price": float(price),
            "marketCap": mcap,
            "marketCapFmt": _fmt_mcap(mcap),
        }

        # ── Fundamentals ──
        r["pe"]             = _sf(info.get("trailingPE"))
        r["forwardPe"]      = _sf(info.get("forwardPE"))
        r["pb"]             = _sf(info.get("priceToBook"))
        r["ps"]             = _sf(info.get("priceToSalesTrailing12Months"))
        r["roe"]            = _sf(info.get("returnOnEquity"))
        r["roa"]            = _sf(info.get("returnOnAssets"))
        r["profitMargin"]   = _sf(info.get("profitMargins"))
        r["operatingMargin"]= _sf(info.get("operatingMargins"))
        r["debtEquity"]     = _sf(info.get("debtToEquity"))       # percentage (150 = 1.5x)
        r["currentRatio"]   = _sf(info.get("currentRatio"))
        r["fcf"]            = info.get("freeCashflow")
        r["dividendYield"]  = _sf(info.get("dividendYield"))
        r["payoutRatio"]    = _sf(info.get("payoutRatio"))
        r["revenueGrowth"]  = _sf(info.get("revenueGrowth"))
        r["earningsGrowth"] = _sf(info.get("earningsGrowth"))
        r["fcfYield"]       = (r["fcf"] / mcap) if r["fcf"] and mcap > 0 else None

        # ── Price Metrics ──
        r["high52w"] = _sf(info.get("fiftyTwoWeekHigh")) or price
        r["low52w"]  = _sf(info.get("fiftyTwoWeekLow"))  or price
        r["sma200"]  = _sf(info.get("twoHundredDayAverage")) or price
        r["sma50"]   = _sf(info.get("fiftyDayAverage"))  or price

        h52 = r["high52w"]
        r["drawdownPct"]   = (price - h52) / h52 if h52 > 0 else 0
        r["priceVsSma200"] = (price - r["sma200"]) / r["sma200"] if r["sma200"] > 0 else 0
        r["priceVsSma50"]  = (price - r["sma50"]) / r["sma50"] if r["sma50"] > 0 else 0
        rng = r["high52w"] - r["low52w"]
        r["rangePosition"] = (price - r["low52w"]) / rng if rng > 0 else 0.5

        # ── Historical Returns + Volume ──
        try:
            hist = t.history(period="6mo")
            if not hist.empty and len(hist) > 1:
                closes = hist["Close"]
                cur = float(closes.iloc[-1])
                r["return1m"] = float(cur / closes.iloc[-21] - 1) if len(closes) >= 21 else float(cur / closes.iloc[0] - 1)
                r["return3m"] = float(cur / closes.iloc[-63] - 1) if len(closes) >= 63 else r["return1m"]
                r["return6m"] = float(cur / closes.iloc[0] - 1)
                avg_v = float(hist["Volume"].mean())
                rec_v = float(hist["Volume"].iloc[-5:].mean())
                r["avgVolume"]      = avg_v
                r["recentVolRatio"] = rec_v / avg_v if avg_v > 0 else 1.0
            else:
                _defaults(r)
        except Exception:
            _defaults(r)

        # ── Scoring ──
        r.update(_score_fundamentals(r))
        r.update(_score_blood(r))

        fs, bs = r["fundamentalScore"], r["bloodScore"]

        # FIX #1: Fundamental-weighted composite (60/40) — no inflating cross-bonus.
        # Blood alone should never carry a weak fundamental stock to a high score.
        composite = min(fs * 0.60 + bs * 0.40, 100)
        r["compositeScore"] = round(composite)
        r["grade"]  = _grade(composite, fs)
        r["signal"] = _signal(composite, fs, bs, r)
        r["error"]  = None
        return r

    except Exception as e:
        return {"ticker": symbol, "error": str(e), "compositeScore": 0}


# ── Fundamental Scoring ──

def _score_fundamentals(r: Dict) -> Dict:
    s: Dict = {}

    pe = r.get("pe")
    s["peScore"] = (
        100 if pe and 0 < pe < 10 else
        85  if pe and pe < 15 else
        70  if pe and pe < 20 else
        50  if pe and pe < 25 else
        30  if pe and pe < 35 else
        10  if pe and pe > 0 else 20
    )

    pb = r.get("pb")
    s["pbScore"] = (
        100 if pb and 0 < pb < 1.0 else
        85  if pb and pb < 1.5 else
        60  if pb and pb < 3.0 else
        40  if pb and pb < 5.0 else
        20  if pb and pb < 10 else
        5   if pb else 30
    )

    roe = r.get("roe")
    s["roeScore"] = (
        100 if roe and roe > 0.25 else
        85  if roe and roe > 0.20 else
        70  if roe and roe > 0.15 else
        50  if roe and roe > 0.10 else
        30  if roe and roe > 0.05 else
        10  if roe is not None else 30
    )

    mg = r.get("profitMargin")
    s["marginScore"] = (
        100 if mg and mg > 0.25 else
        85  if mg and mg > 0.20 else
        70  if mg and mg > 0.15 else
        55  if mg and mg > 0.10 else
        35  if mg and mg > 0.05 else
        10  if mg is not None else 30
    )

    de = r.get("debtEquity")
    s["debtScore"] = (
        100 if de is not None and de < 30  else
        85  if de is not None and de < 50  else
        65  if de is not None and de < 100 else
        45  if de is not None and de < 150 else
        25  if de is not None and de < 200 else
        10  if de is not None else 50
    )

    fy = r.get("fcfYield")
    s["fcfScore"] = (
        100 if fy and fy > 0.10 else
        85  if fy and fy > 0.08 else
        70  if fy and fy > 0.05 else
        50  if fy and fy > 0.03 else
        30  if fy and fy > 0.01 else
        10  if fy is not None else 30
    )

    # FIX #5: Debt-burden FCF haircut.
    # Backward-looking FCF is misleading when debt is extreme — debt service
    # eats free cash flow. Penalize FCF score proportional to leverage.
    de = r.get("debtEquity")
    if de is not None and de > 150 and fy and fy > 0:
        # Heavy leverage: haircut FCF score by 30-60%
        if de > 300:
            s["fcfScore"] = round(s["fcfScore"] * 0.40)  # 60% haircut
        elif de > 200:
            s["fcfScore"] = round(s["fcfScore"] * 0.55)  # 45% haircut
        else:
            s["fcfScore"] = round(s["fcfScore"] * 0.70)  # 30% haircut
        s["fcfDebtAdjusted"] = True
    else:
        s["fcfDebtAdjusted"] = False

    w = {"peScore": .20, "pbScore": .15, "roeScore": .20, "marginScore": .15, "debtScore": .15, "fcfScore": .15}
    s["fundamentalScore"] = round(sum(s[k] * w[k] for k in w))
    return s


# ── Blood Scoring ──

def _score_blood(r: Dict) -> Dict:
    s: Dict = {}

    dd = abs(r.get("drawdownPct", 0))
    s["drawdownScore"] = (
        100 if dd > 0.50 else 90 if dd > 0.40 else 75 if dd > 0.30 else
        60  if dd > 0.20 else 40 if dd > 0.10 else 20 if dd > 0.05 else 5
    )

    r1 = r.get("return1m", 0)
    r3 = r.get("return3m", 0)
    avg_r = r1 * 0.6 + r3 * 0.4
    s["momentumScore"] = (
        100 if avg_r < -0.30 else 85 if avg_r < -0.20 else 65 if avg_r < -0.10 else
        45  if avg_r < -0.05 else 25 if avg_r < 0 else 5
    )

    sm = r.get("priceVsSma200", 0)
    s["smaScore"] = (
        100 if sm < -0.30 else 85 if sm < -0.20 else 65 if sm < -0.10 else
        45  if sm < -0.05 else 25 if sm < 0 else 5
    )

    # FIX #2: Vol ratio scoring — context-dependent.
    # In a blood screen, LOW volume during selloff = institutions holding (bullish)
    # HIGH volume during drawdown = panic liquidation (more pain likely)
    vr = r.get("recentVolRatio", 1.0)
    dd = abs(r.get("drawdownPct", 0))
    if dd > 0.10:  # stock is in drawdown territory
        # Low volume on selloff = smart money not selling = bullish for value
        s["volScore"] = (
            90  if vr < 0.6 else   # volume dried up — sellers exhausted
            75  if vr < 0.8 else   # below average — low conviction selling
            55  if vr < 1.0 else   # avg volume — neutral
            35  if vr < 1.5 else   # above avg — some distribution
            15  if vr < 2.5 else   # high volume selling — capitulation
            5                      # extreme volume — panic liquidation
        )
    else:
        # Not in drawdown — standard interpretation
        s["volScore"] = 50 if vr > 1.5 else 40 if vr > 1.2 else 30

    w = {"drawdownScore": .35, "momentumScore": .30, "smaScore": .20, "volScore": .15}
    s["bloodScore"] = round(sum(s[k] * w[k] for k in w))
    return s


# ── Helpers ──

def _sf(val) -> Optional[float]:
    if val is None: return None
    try: return float(val)
    except (ValueError, TypeError): return None

def _fmt_mcap(m):
    if not m: return "N/A"
    if m >= 1e12: return f"${m/1e12:.1f}T"
    if m >= 1e9:  return f"${m/1e9:.1f}B"
    if m >= 1e6:  return f"${m/1e6:.0f}M"
    return f"${m:,.0f}"

def _defaults(r):
    r["return1m"] = 0; r["return3m"] = 0; r["return6m"] = 0
    r["avgVolume"] = 0; r["recentVolRatio"] = 1.0

def _grade(sc, fs=None):
    # FIX #4: Fundamental gate for top grades.
    # Blood score alone should NOT produce an A — both legs must be strong.
    if fs is not None:
        if sc >= 85 and fs >= 70: return "A+"
        if sc >= 75 and fs >= 65: return "A"
        if sc >= 85: return "B+"  # high composite but weak fundamentals → cap at B+
        if sc >= 75: return "B+"  # same — blood carrying, fundamentals lacking
    else:
        if sc >= 85: return "A+"
        if sc >= 75: return "A"
    if sc >= 65: return "B+"
    if sc >= 55: return "B"
    if sc >= 45: return "C"
    if sc >= 35: return "D"
    return "F"

def _signal(comp, fs, bs, r=None):
    # FIX #3: Strict fundamental floors for value signals.
    # Blood alone can NOT promote a stock to DEEP VALUE or VALUE.
    # P/E 30+ and P/B 10+ should NEVER be "deep value" regardless of drawdown.
    pe = r.get("pe") if r else None
    pb = r.get("pb") if r else None

    # Hard valuation caps — no amount of blood overcomes these
    valuation_ok = True
    if pe and pe > 28: valuation_ok = False   # P/E > 28 = not value territory
    if pb and pb > 8:  valuation_ok = False   # P/B > 8 = not value territory

    if comp >= 75 and fs >= 70 and valuation_ok:
        return "DEEP VALUE"
    if comp >= 60 and fs >= 55 and valuation_ok:
        return "VALUE"
    if comp >= 60 and not valuation_ok:
        return "BLOOD ONLY"   # beaten down but fundamentals don't support value label
    if comp >= 45:
        return "FAIR"
    if fs < 30:
        return "OVERVALUED"
    return "NEUTRAL"


# ── Async Wrapper ──

async def async_scan_tickers(symbols: List[str], max_workers: int = 5) -> Dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, scan_tickers, symbols, max_workers)
