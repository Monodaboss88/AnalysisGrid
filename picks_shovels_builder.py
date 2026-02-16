"""
Picks & Shovels Research Builder
================================
Reusable script to build investor-grade supply chain analysis for ANY industry.

Usage:
  1. Define your research in a config dict (tickers, layers, moats)
  2. Run: python picks_shovels_builder.py
  3. It pulls fundamentals (Finnhub) + performance (Polygon) + generates HTML

Modes:
  --stages     Interactive step-by-step (review each stage)
  --full       Pull everything and generate in one shot (default)
  --config     Path to a JSON config file (default: uses built-in example)

Examples:
  python picks_shovels_builder.py --full
  python picks_shovels_builder.py --stages
  python picks_shovels_builder.py --config my_biotech_research.json --full
"""

import os, sys, json, time, argparse
from datetime import datetime, timedelta
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests..."); os.system("pip install requests"); import requests

# ═══════════════════════════════════════════════════════════════
# CONFIG: Define your research here (or load from JSON)
# ═══════════════════════════════════════════════════════════════

EXAMPLE_CONFIG = {
    "title": "AI Infrastructure Picks & Shovels",
    "subtitle": "Mapping the supply chain behind the $10T AI buildout",
    "industry": "AI / Cloud Infrastructure",
    "date": "February 2026",

    # Layer 1: End customers (these are shown faded — the "buyers")
    "layer1": [
        {"ticker": "AAPL", "role": "Consumer Devices"},
        {"ticker": "MSFT", "role": "Cloud + AI"},
        {"ticker": "NVDA", "role": "AI GPUs"},
        {"ticker": "AMZN", "role": "AWS Cloud"},
        {"ticker": "GOOGL", "role": "Search + Cloud"},
        {"ticker": "META", "role": "AI + Social"},
    ],

    # Layer 2: Direct suppliers — the core picks & shovels
    "layer2": [
        {"ticker": "TSM",  "role": "Chip Fabrication",    "moat": "monopoly",   "moat_label": "Monopoly 90%",   "tier": 1, "founded": 1987},
        {"ticker": "ANET", "role": "DC Networking",       "moat": "tech",       "moat_label": "Tech Lead",      "tier": 1, "founded": 2004},
        {"ticker": "ASML", "role": "EUV Lithography",     "moat": "monopoly",   "moat_label": "Monopoly 100%",  "tier": 1, "founded": 1984},
        {"ticker": "MU",   "role": "Memory / HBM",        "moat": "oligopoly",  "moat_label": "Oligopoly",      "tier": 2, "founded": 1978},
        {"ticker": "MRVL", "role": "Custom Silicon",      "moat": "tech",       "moat_label": "ASIC Design",    "tier": 2, "founded": 1995},
        {"ticker": "GLW",  "role": "Fiber + Glass",        "moat": "scale",      "moat_label": "Materials",      "tier": 0, "founded": 1851},
        {"ticker": "VRT",  "role": "Power & Cooling",      "moat": "oligopoly",  "moat_label": "Bottleneck",     "tier": 0, "founded": 2016},
    ],

    # Layer 3: Suppliers to the suppliers
    "layer3": [
        {"ticker": "KLAC", "role": "Chip Inspection",     "moat": "monopoly",   "moat_label": "Near-Monopoly",  "tier": 3, "founded": 1975},
        {"ticker": "LRCX", "role": "Etch & Deposition",   "moat": "oligopoly",  "moat_label": "Oligopoly",      "tier": 3, "founded": 1980},
        {"ticker": "AMAT", "role": "Broadest Equipment",   "moat": "scale",      "moat_label": "Scale",          "tier": 3, "founded": 1967},
        {"ticker": "ENTG", "role": "Chemicals & Filters",  "moat": "consumable", "moat_label": "Consumable",     "tier": 3, "founded": 1966},
        {"ticker": "LIN",  "role": "Industrial Gases",     "moat": "oligopoly",  "moat_label": "Oligopoly",      "tier": 3, "founded": 1879},
    ],

    # Tier descriptions for the allocation framework
    "tiers": {
        1: {"name": "Core Holdings",     "pct": "60%", "desc": "Profitable, low debt, irreplaceable moats. Buy and hold through cycles."},
        2: {"name": "Growth Bets",        "pct": "20%", "desc": "High growth, improving balance sheets. Higher volatility, higher reward."},
        3: {"name": "Deep Infrastructure", "pct": "15%", "desc": "Layer 3 equipment suppliers. Benefit from every new fab built globally."},
        0: {"name": "Contrarian / Alt",    "pct": "5%",  "desc": "Non-obvious infrastructure plays with strong momentum."},
    },

    # Catalysts & risks per subsector
    "catalysts_risks": [
        {
            "name": "Chip Fabrication — TSM", "icon": "⚡", "color": "green",
            "catalysts": ["AI chip demand surge, Arizona fab expansion", "AAPL + NVDA contractually locked, 90% advanced node share"],
            "risks": ["China/Taiwan geopolitical risk", "CHIPS Act may dilute geographic moat"]
        },
        {
            "name": "Networking — ANET", "icon": "🔗", "color": "cyan",
            "catalysts": ["800G upgrade cycle, AI clusters need low-latency switching", "Zero debt, 65% gross margin"],
            "risks": ["Customer concentration — META ~30% of revenue", "Cisco pivot to AI networking"]
        },
        {
            "name": "Lithography — ASML", "icon": "🔬", "color": "red",
            "catalysts": ["High-NA EUV rollout — €350M per machine, no alternative", "100% monopoly on EUV"],
            "risks": ["China export ban reduces ~15% addressable market", "Lumpy quarterly revenue"]
        },
        {
            "name": "Memory — MU", "icon": "💾", "color": "amber",
            "catalysts": ["HBM3e demand from NVDA, supply constrained through 2026", "+85% revenue growth, 91% analyst buy rating"],
            "risks": ["Historically cyclical — oversupply crashes", "Samsung & SK Hynix competition"]
        },
        {
            "name": "Custom Silicon — MRVL", "icon": "🧮", "color": "purple",
            "catalysts": ["Cloud ASIC wins: AMZN Graviton, GOOGL TPU", "5G recovery + optical interconnect"],
            "risks": ["Design win lumpiness", "Market hasn't priced in the thesis"]
        },
        {
            "name": "Equipment — KLAC / LRCX / AMAT", "icon": "🏭", "color": "pink",
            "catalysts": ["Global fab buildout — CHIPS Act, EU subsidies", "Advanced packaging requires new equipment"],
            "risks": ["Capex cycle slowdown risk", "KLAC debt/equity at 1.95"]
        },
    ],
}


# ═══════════════════════════════════════════════════════════════
# API CLIENTS
# ═══════════════════════════════════════════════════════════════

POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "")


def get_all_tickers(config):
    """Extract all unique tickers from layers 2 and 3 (the investable ones)."""
    tickers = []
    for item in config.get("layer2", []) + config.get("layer3", []):
        tickers.append(item["ticker"])
    return list(dict.fromkeys(tickers))  # unique, preserving order


def fetch_finnhub_fundamentals(tickers):
    """Pull key fundamentals from Finnhub for each ticker."""
    if not FINNHUB_KEY:
        print("  ⚠  FINNHUB_API_KEY not set — skipping fundamentals")
        return {}

    results = {}
    base = "https://finnhub.io/api/v1"

    for i, t in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] Pulling {t}...", end=" ", flush=True)
        data = {}

        try:
            # Basic metrics
            r = requests.get(f"{base}/stock/metric", params={"symbol": t, "metric": "all", "token": FINNHUB_KEY}, timeout=10)
            if r.ok:
                m = r.json().get("metric", {})
                data["gross_margin"]   = m.get("grossMarginTTM") or m.get("grossMarginAnnual")
                data["net_margin"]     = m.get("netProfitMarginTTM") or m.get("netProfitMarginAnnual")
                data["roe"]            = m.get("roeTTM") or m.get("roeRly")
                data["debt_equity"]    = m.get("totalDebt/totalEquityQuarterly") or m.get("totalDebt/totalEquityAnnual")
                data["rev_growth"]     = m.get("revenueGrowthTTMYoy") or m.get("revenueGrowth3Y")
                data["pe"]             = m.get("peTTM") or m.get("peAnnual")
                data["beta"]           = m.get("beta")
                data["week52_high"]    = m.get("52WeekHigh")
                data["week52_low"]     = m.get("52WeekLow")
                data["market_cap"]     = m.get("marketCapitalization")

            time.sleep(0.15)  # rate limit

            # Analyst recommendations
            r2 = requests.get(f"{base}/stock/recommendation", params={"symbol": t, "token": FINNHUB_KEY}, timeout=10)
            if r2.ok and r2.json():
                rec = r2.json()[0]
                data["analyst_buy"]    = rec.get("buy", 0) + rec.get("strongBuy", 0)
                data["analyst_hold"]   = rec.get("hold", 0)
                data["analyst_sell"]   = rec.get("sell", 0) + rec.get("strongSell", 0)

            time.sleep(0.15)

            # Price target
            r3 = requests.get(f"{base}/stock/price-target", params={"symbol": t, "token": FINNHUB_KEY}, timeout=10)
            if r3.ok and r3.json():
                pt = r3.json()
                data["target_high"]    = pt.get("targetHigh")
                data["target_mean"]    = pt.get("targetMean")
                data["target_low"]     = pt.get("targetLow")

            time.sleep(0.15)

        except Exception as e:
            print(f"Error: {e}")

        results[t] = data
        status = "✓" if data.get("gross_margin") else "partial"
        print(status)

    return results


def fetch_company_profiles(tickers, config=None):
    """Pull company profiles (founding year, market cap, country) from Finnhub.
    Falls back to config-embedded 'founded' fields if API unavailable."""
    if not FINNHUB_KEY:
        print("  ⚠  FINNHUB_API_KEY not set — using config-embedded founding years only")
        return _profiles_from_config(tickers, config)

    results = {}
    base = "https://finnhub.io/api/v1"

    for i, t in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] Profile {t}...", end=" ", flush=True)
        data = {}

        try:
            r = requests.get(f"{base}/stock/profile2", params={"symbol": t, "token": FINNHUB_KEY}, timeout=10)
            if r.ok:
                p = r.json()
                data["name"] = p.get("name", t)
                data["ipo"] = p.get("ipo", "")           # IPO date YYYY-MM-DD
                data["country"] = p.get("country", "")
                data["market_cap"] = p.get("marketCapitalization")  # in millions
                data["industry"] = p.get("finnhubIndustry", "")
                # Try to derive founding year from IPO date
                if data["ipo"]:
                    try:
                        data["ipo_year"] = int(data["ipo"][:4])
                    except:
                        pass
            time.sleep(0.15)
        except Exception as e:
            print(f"Error: {e}")

        results[t] = data
        print(f"✓ {data.get('name', t)}")

    # Merge in config-embedded founding years (override IPO year if provided)
    if config:
        for item in config.get("layer2", []) + config.get("layer3", []):
            t = item.get("ticker", "")
            if t in results and item.get("founded"):
                results[t]["founded"] = item["founded"]
            elif t in results and not results[t].get("founded"):
                results[t]["founded"] = results[t].get("ipo_year")

    return results


def _profiles_from_config(tickers, config):
    """Extract founding years from config when API unavailable."""
    results = {}
    if not config:
        return results
    for item in config.get("layer2", []) + config.get("layer3", []):
        t = item.get("ticker", "")
        if t in tickers:
            results[t] = {
                "name": t,
                "founded": item.get("founded"),
                "market_cap": None
            }
    return results


def compute_trajectory(config, performance, profiles):
    """Compute trajectory analysis for each ticker.
    Returns a list of trajectory dicts sorted by velocity score."""
    current_year = datetime.now().year
    trajectories = []

    all_investable = config.get("layer2", []) + config.get("layer3", [])

    for item in all_investable:
        t = item["ticker"]
        perf = performance.get(t, {})
        prof = profiles.get(t, {})

        founded = prof.get("founded") or item.get("founded")
        if not founded:
            continue

        age = current_year - int(founded)
        if age <= 0:
            continue

        ret_1y = perf.get("1Y")
        ret_2y = perf.get("2Y")
        price = perf.get("price")
        mcap = prof.get("market_cap")  # in millions

        # Velocity score: 2Y return / age (how much growth per year existed)
        velocity = None
        if ret_2y is not None and age > 0:
            velocity = round(ret_2y / age, 2)
        elif ret_1y is not None and age > 0:
            velocity = round((ret_1y * 2) / age, 2)  # extrapolate

        # Trajectory label
        label = "Steady"
        if ret_1y is not None:
            if ret_2y is not None:
                if ret_2y > 200:
                    label = "Explosive" if ret_1y > 100 else "Monster"
                elif ret_2y > 100:
                    label = "Surging" if ret_1y > 50 else "Accelerating"
                elif ret_2y > 50:
                    label = "Strong" if ret_1y > 20 else "Compounding"
                elif ret_2y > 10:
                    label = "Steady" if ret_1y > 0 else "Decelerating"
                elif ret_2y > 0:
                    label = "Flat" if ret_1y > -5 else "Fading"
                elif ret_2y > -15:
                    label = "Stalled"
                else:
                    label = "Declining"
                # Special: old companies with huge returns = Reinvention
                if age > 100 and ret_2y > 100:
                    label = "Reinvention"
            else:
                if ret_1y > 100: label = "Surging"
                elif ret_1y > 30: label = "Strong"
                elif ret_1y > 5: label = "Steady"
                elif ret_1y > -10: label = "Flat"
                else: label = "Declining"

        # Grade
        grade = "C"
        if velocity is not None:
            if velocity > 15: grade = "A++"
            elif velocity > 8: grade = "A+"
            elif velocity > 4: grade = "A"
            elif velocity > 2: grade = "B+"
            elif velocity > 1: grade = "B"
            elif velocity > 0.3: grade = "C+"
            elif velocity > 0: grade = "C"
            else: grade = "D"

        # Format market cap
        mcap_str = "—"
        if mcap:
            if mcap >= 1_000_000:
                mcap_str = f"${mcap/1_000_000:.1f}T"
            elif mcap >= 1000:
                mcap_str = f"${mcap/1000:.0f}B"
            else:
                mcap_str = f"${mcap:.0f}M"

        trajectories.append({
            "ticker": t,
            "role": item.get("role", ""),
            "founded": int(founded),
            "age": age,
            "market_cap": mcap_str,
            "price": price,
            "return_1y": ret_1y,
            "return_2y": ret_2y,
            "velocity": velocity,
            "label": label,
            "grade": grade,
            "moat": item.get("moat", ""),
            "tier": item.get("tier", 0),
        })

    # Sort by velocity descending
    trajectories.sort(key=lambda x: x.get("velocity") or -999, reverse=True)
    return trajectories


def fetch_polygon_performance(tickers, periods=None):
    """Pull price performance over multiple timeframes from Polygon."""
    if not POLYGON_KEY:
        print("  ⚠  POLYGON_API_KEY not set — skipping performance")
        return {}

    if periods is None:
        periods = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}

    results = {}
    today = datetime.now()

    for i, t in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] Pulling {t}...", end=" ", flush=True)
        perf = {}

        try:
            # Get current price
            r = requests.get(f"https://api.polygon.io/v2/aggs/ticker/{t}/prev",
                             params={"apiKey": POLYGON_KEY}, timeout=10)
            if not r.ok or not r.json().get("results"):
                print("no data")
                results[t] = perf
                continue

            current = r.json()["results"][0]["c"]
            perf["price"] = current
            time.sleep(0.15)

            # Get historical prices for each period
            for label, days in periods.items():
                past_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
                r2 = requests.get(f"https://api.polygon.io/v2/aggs/ticker/{t}/range/1/day/{past_date}/{past_date}",
                                  params={"apiKey": POLYGON_KEY}, timeout=10)
                if r2.ok and r2.json().get("results"):
                    old_price = r2.json()["results"][0]["c"]
                    perf[label] = round(((current - old_price) / old_price) * 100, 1)
                else:
                    # Try nearby dates if exact date has no data
                    for offset in [1, 2, 3, -1, -2, -3]:
                        alt_date = (today - timedelta(days=days + offset)).strftime("%Y-%m-%d")
                        r3 = requests.get(f"https://api.polygon.io/v2/aggs/ticker/{t}/range/1/day/{alt_date}/{alt_date}",
                                          params={"apiKey": POLYGON_KEY}, timeout=10)
                        if r3.ok and r3.json().get("results"):
                            old_price = r3.json()["results"][0]["c"]
                            perf[label] = round(((current - old_price) / old_price) * 100, 1)
                            break
                        time.sleep(0.1)

                time.sleep(0.15)

        except Exception as e:
            print(f"Error: {e}")

        results[t] = perf
        print(f"${perf.get('price', '?')}")

    return results


# ═══════════════════════════════════════════════════════════════
# HTML GENERATOR
# ═══════════════════════════════════════════════════════════════

def color_class(val, thresholds=(40, 20)):
    """Return green/amber/red based on value vs thresholds."""
    if val is None: return "muted"
    if val >= thresholds[0]: return "green"
    if val >= thresholds[1]: return "amber"
    return "red"

def de_color(val):
    if val is None: return "muted"
    if val <= 0.4: return "green"
    if val <= 1.0: return "amber"
    return "red"

def growth_color(val):
    if val is None: return "muted"
    if val >= 15: return "green"
    if val >= 5: return "amber"
    return "red"

def fmt(val, suffix="", prefix=""):
    if val is None: return "—"
    if isinstance(val, float):
        return f"{prefix}{val:.1f}{suffix}"
    return f"{prefix}{val}{suffix}"

def fmt_pct(val):
    if val is None: return "—"
    return f"+{val:.0f}%" if val >= 0 else f"{val:.0f}%"


def generate_html(config, fundamentals, performance, profiles=None):
    """Generate the full investor presentation HTML."""
    if profiles is None:
        profiles = {}

    tickers_l2 = config.get("layer2", [])
    tickers_l3 = config.get("layer3", [])
    all_investable = tickers_l2 + tickers_l3
    all_tickers_list = [t["ticker"] for t in all_investable]

    # ── Compute KPIs ──
    total_companies = len(all_tickers_list)
    num_layers = sum(1 for k in ["layer1", "layer2", "layer3"] if config.get(k))
    monopoly_count = sum(1 for t in all_investable if t.get("moat") in ("monopoly", "oligopoly"))

    # Best 1Y performer
    best_1y_ticker = ""
    best_1y_val = -999
    for t in all_tickers_list:
        perf = performance.get(t, {})
        val = perf.get("1Y")
        if val is not None and val > best_1y_val:
            best_1y_val = val
            best_1y_ticker = t

    # ── Build tier groups ──
    tier_groups = {}  # tier_num -> list of items with fundamentals
    for item in all_investable:
        tier = item.get("tier", 0)
        if tier not in tier_groups:
            tier_groups[tier] = []
        ticker = item["ticker"]
        f = fundamentals.get(ticker, {})
        tier_groups[tier].append({**item, "fundamentals": f})

    # ── Sort performance for bar chart ──
    perf_sorted = []
    for item in all_investable:
        t = item["ticker"]
        p = performance.get(t, {})
        tier = item.get("tier", 0)
        perf_sorted.append({"ticker": t, "tier": tier, "return_1y": p.get("1Y"), "price": p.get("price")})
    perf_sorted.sort(key=lambda x: x.get("return_1y") or -999, reverse=True)
    max_return = max((abs(x.get("return_1y") or 0) for x in perf_sorted), default=100) or 100

    # ── Color map for CSS ──
    color_map = {
        "green": "#10b981", "red": "#ef4444", "amber": "#f59e0b",
        "cyan": "#00d4ff", "purple": "#a78bfa", "pink": "#f472b6", "blue": "#3b82f6"
    }
    tier_colors = {1: "green", 2: "amber", 3: "purple", 0: "cyan"}

    # ═══ Start building HTML ═══
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{config['title']}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
  *{{margin:0;padding:0;box-sizing:border-box}}
  :root{{--bg:#0a0e17;--bg2:#111827;--card:#151d2e;--border:#1e293b;--text:#e2e8f0;--muted:#94a3b8;--cyan:#00d4ff;--green:#10b981;--red:#ef4444;--amber:#f59e0b;--purple:#a78bfa;--pink:#f472b6;--blue:#3b82f6}}
  body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;overflow-x:hidden}}
  .hero{{text-align:center;padding:80px 40px 60px;background:linear-gradient(180deg,#0f1729,var(--bg));position:relative;overflow:hidden}}
  .hero::before{{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(circle,rgba(0,212,255,0.04),transparent 50%);animation:pulse 8s ease-in-out infinite}}
  @keyframes pulse{{0%,100%{{transform:scale(1);opacity:.5}}50%{{transform:scale(1.1);opacity:1}}}}
  .hero h1{{font-size:48px;font-weight:900;letter-spacing:-1.5px;background:linear-gradient(135deg,#fff,var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;position:relative;margin-bottom:12px}}
  .hero .subtitle{{font-size:20px;color:var(--muted);font-weight:400;margin-bottom:8px}}
  .hero .date{{font-size:14px;color:#475569;letter-spacing:2px;text-transform:uppercase}}
  .kpi-row{{display:flex;justify-content:center;gap:40px;margin-top:40px;flex-wrap:wrap}}
  .kpi{{text-align:center}}.kpi .num{{font-size:36px;font-weight:800;color:var(--cyan)}}.kpi .label{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-top:4px}}
  .container{{max-width:1280px;margin:0 auto;padding:0 24px}}
  section{{margin-bottom:60px}}
  .section-header{{display:flex;align-items:center;gap:12px;margin-bottom:28px;padding-bottom:12px;border-bottom:1px solid var(--border)}}
  .section-header .num{{width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,var(--cyan),var(--blue));display:flex;align-items:center;justify-content:center;font-weight:800;font-size:16px;color:#000;flex-shrink:0}}
  .section-header h2{{font-size:28px;font-weight:700;letter-spacing:-.5px}}
  .section-header .tag{{margin-left:auto;background:rgba(0,212,255,0.1);color:var(--cyan);padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;letter-spacing:.5px}}
  .chain-layer{{display:flex;align-items:stretch;gap:16px}}
  .layer-label{{width:140px;flex-shrink:0;display:flex;flex-direction:column;justify-content:center;align-items:flex-end;padding-right:20px;text-align:right}}
  .layer-label .layer-name{{font-size:13px;font-weight:700;color:var(--cyan);text-transform:uppercase;letter-spacing:1.5px}}
  .layer-label .layer-desc{{font-size:11px;color:var(--muted);margin-top:2px}}
  .layer-nodes{{display:flex;gap:10px;flex-wrap:wrap;flex:1;padding:16px 0}}
  .node{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 18px;min-width:130px;transition:all .2s;cursor:default}}
  .node:hover{{border-color:var(--cyan);transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,212,255,0.1)}}
  .node .ticker{{font-size:16px;font-weight:800;color:#fff}}.node .role{{font-size:11px;color:var(--muted);margin-top:2px}}
  .moat-badge{{display:inline-block;margin-top:6px;padding:2px 8px;border-radius:6px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}}
  .moat-monopoly{{background:rgba(239,68,68,0.15);color:var(--red)}}.moat-oligopoly{{background:rgba(245,158,11,0.15);color:var(--amber)}}.moat-tech{{background:rgba(0,212,255,0.15);color:var(--cyan)}}.moat-scale{{background:rgba(167,139,250,0.15);color:var(--purple)}}.moat-consumable{{background:rgba(16,185,129,0.15);color:var(--green)}}
  .chain-arrow{{display:flex;justify-content:center;padding:6px 0 6px 140px;position:relative}}
  .chain-arrow svg{{width:24px;height:32px}}
  .chain-arrow .arrow-label{{position:absolute;left:180px;top:50%;transform:translateY(-50%);font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase}}
  .tier-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:20px}}
  .tier-card{{background:var(--card);border:1px solid var(--border);border-radius:16px;overflow:hidden}}
  .tier-card-header{{padding:16px 20px;font-weight:700;font-size:14px;text-transform:uppercase;letter-spacing:1px;display:flex;align-items:center;gap:10px}}
  .tc-green .tier-card-header{{background:linear-gradient(90deg,rgba(16,185,129,0.15),transparent);color:var(--green);border-bottom:1px solid rgba(16,185,129,0.2)}}
  .tc-amber .tier-card-header{{background:linear-gradient(90deg,rgba(245,158,11,0.15),transparent);color:var(--amber);border-bottom:1px solid rgba(245,158,11,0.2)}}
  .tc-purple .tier-card-header{{background:linear-gradient(90deg,rgba(167,139,250,0.15),transparent);color:var(--purple);border-bottom:1px solid rgba(167,139,250,0.2)}}
  .tc-cyan .tier-card-header{{background:linear-gradient(90deg,rgba(0,212,255,0.15),transparent);color:var(--cyan);border-bottom:1px solid rgba(0,212,255,0.2)}}
  .stock-header{{display:grid;grid-template-columns:70px 1fr 80px 80px 80px;padding:8px 20px;font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid var(--border)}}
  .stock-header span:nth-child(n+3){{text-align:center}}
  .stock-row{{display:grid;grid-template-columns:70px 1fr 80px 80px 80px;align-items:center;padding:12px 20px;border-bottom:1px solid var(--border);font-size:13px;transition:background .15s}}
  .stock-row:last-child{{border-bottom:none}}.stock-row:hover{{background:rgba(255,255,255,0.02)}}
  .stock-row .sticker{{font-weight:800;color:#fff;font-size:15px}}.stock-row .desc{{color:var(--muted);font-size:12px}}
  .stock-row .metric{{text-align:center;font-weight:600}}
  .metric.green{{color:var(--green)}}.metric.amber{{color:var(--amber)}}.metric.red{{color:var(--red)}}.metric.muted{{color:var(--muted)}}
  .perf-container{{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:28px}}
  .perf-row{{display:grid;grid-template-columns:80px 80px 1fr 80px;align-items:center;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.03)}}
  .perf-row:last-child{{border-bottom:none}}
  .perf-row .pticker{{font-weight:800;font-size:15px}}.perf-row .ptier{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}}
  .perf-bar-wrap{{height:28px;background:rgba(255,255,255,0.03);border-radius:6px;overflow:hidden;position:relative}}
  .perf-bar{{height:100%;border-radius:6px;display:flex;align-items:center;justify-content:flex-end;padding-right:10px;font-size:12px;font-weight:700;color:#fff;min-width:48px}}
  .perf-bar.positive{{background:linear-gradient(90deg,rgba(16,185,129,0.3),var(--green))}}.perf-bar.negative{{background:linear-gradient(90deg,var(--red),rgba(239,68,68,0.3))}}
  .perf-row .page{{text-align:right;font-size:12px;color:var(--muted)}}
  .traj-table{{width:100%;border-collapse:separate;border-spacing:0;background:var(--card);border:1px solid var(--border);border-radius:16px;overflow:hidden}}
  .traj-table thead th{{background:rgba(0,212,255,0.06);padding:12px 16px;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);text-align:left;border-bottom:1px solid var(--border);font-weight:600}}
  .traj-table thead th:nth-child(n+4){{text-align:center}}
  .traj-table tbody td{{padding:12px 16px;border-bottom:1px solid rgba(255,255,255,0.03);font-size:13px}}
  .traj-table tbody tr:last-child td{{border-bottom:none}}
  .traj-table tbody tr:hover{{background:rgba(255,255,255,0.02)}}
  .traj-table .t-ticker{{font-weight:800;font-size:15px;color:#fff}}
  .traj-table .t-era{{font-size:11px;color:var(--muted)}}
  .traj-table .t-center{{text-align:center;font-weight:600}}
  .traj-label{{display:inline-block;padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700;letter-spacing:.3px}}
  .traj-label.explosive,.traj-label.monster{{background:rgba(16,185,129,0.2);color:var(--green)}}
  .traj-label.surging,.traj-label.accelerating{{background:rgba(0,212,255,0.15);color:var(--cyan)}}
  .traj-label.strong,.traj-label.compounding{{background:rgba(59,130,246,0.15);color:var(--blue)}}
  .traj-label.reinvention{{background:rgba(245,158,11,0.2);color:var(--amber)}}
  .traj-label.steady{{background:rgba(148,163,184,0.15);color:var(--muted)}}
  .traj-label.flat,.traj-label.fading,.traj-label.decelerating{{background:rgba(148,163,184,0.1);color:#64748b}}
  .traj-label.stalled,.traj-label.declining{{background:rgba(239,68,68,0.15);color:var(--red)}}
  .traj-grade{{display:inline-block;width:36px;height:24px;border-radius:6px;text-align:center;line-height:24px;font-size:12px;font-weight:800}}
  .traj-grade.a-plus-plus{{background:linear-gradient(135deg,var(--green),var(--cyan));color:#000}}
  .traj-grade.a-plus{{background:rgba(16,185,129,0.25);color:var(--green)}}
  .traj-grade.a{{background:rgba(0,212,255,0.2);color:var(--cyan)}}
  .traj-grade.b-plus,.traj-grade.b{{background:rgba(59,130,246,0.15);color:var(--blue)}}
  .traj-grade.c-plus,.traj-grade.c{{background:rgba(148,163,184,0.12);color:var(--muted)}}
  .traj-grade.d{{background:rgba(239,68,68,0.15);color:var(--red)}}
  .traj-leaders{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-top:28px}}
  .traj-leader-card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px;position:relative;overflow:hidden}}
  .traj-leader-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px}}
  .tlc-1::before{{background:linear-gradient(90deg,#ffd700,#f59e0b)}}
  .tlc-2::before{{background:linear-gradient(90deg,#c0c0c0,#94a3b8)}}
  .tlc-3::before{{background:linear-gradient(90deg,#cd7f32,#a78bfa)}}
  .traj-leader-card .tl-rank{{font-size:28px;margin-bottom:6px}}
  .traj-leader-card .tl-ticker{{font-size:20px;font-weight:900;color:#fff}}
  .traj-leader-card .tl-stats{{font-size:12px;color:var(--muted);margin-top:4px;line-height:1.6}}
  .traj-leader-card .tl-stats strong{{color:var(--cyan)}}
  .traj-timeline{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-top:28px}}
  .tl-era{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px;font-size:12px}}
  .tl-era .tl-year{{font-weight:800;color:var(--cyan);font-size:14px;margin-bottom:4px}}
  .tl-era .tl-name{{font-weight:700;color:#fff}}
  .tl-era .tl-what{{color:var(--muted);margin-top:2px;font-size:11px}}
  .cr-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:16px}}
  .cr-card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:24px}}
  .cr-card h3{{font-size:16px;font-weight:700;margin-bottom:14px;display:flex;align-items:center;gap:8px}}
  .cr-card h3 .icon{{width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}}
  .cr-item{{display:flex;gap:8px;margin-bottom:8px;font-size:13px}}
  .cr-item .bullet{{flex-shrink:0;margin-top:2px;font-size:10px}}.cr-item .bullet.cat{{color:var(--green)}}.cr-item .bullet.risk{{color:var(--red)}}
  .cr-item .text{{color:var(--muted)}}
  .verdict-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}
  .verdict-card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:24px;text-align:center;position:relative;overflow:hidden}}
  .verdict-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px}}
  .vc-green::before{{background:var(--green)}}.vc-amber::before{{background:var(--amber)}}.vc-purple::before{{background:var(--purple)}}.vc-cyan::before{{background:var(--cyan)}}
  .verdict-card .v-label{{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:12px}}
  .vl-green{{color:var(--green)}}.vl-amber{{color:var(--amber)}}.vl-purple{{color:var(--purple)}}.vl-cyan{{color:var(--cyan)}}
  .verdict-card .v-tickers{{font-size:22px;font-weight:800;color:#fff;letter-spacing:1px;margin-bottom:8px}}
  .verdict-card .v-desc{{font-size:12px;color:var(--muted)}}
  .thesis-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:20px}}
  .thesis-card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:28px}}
  .thesis-card .tc-label{{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}}
  .thesis-card p{{color:var(--muted);font-size:14px;line-height:1.7}}
  .thesis-card p strong{{color:#fff}}
  .moat-legend{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:24px;display:flex;flex-wrap:wrap;gap:20px;justify-content:center}}
  .moat-legend-item{{display:flex;align-items:center;gap:8px}}
  .moat-legend-item span:last-child{{color:var(--muted);font-size:12px}}
  footer{{text-align:center;padding:40px 20px 60px;color:#334155;font-size:12px}}
  footer a{{color:var(--cyan);text-decoration:none}}
  @media(max-width:768px){{.hero h1{{font-size:28px}}.kpi-row{{gap:20px}}.kpi .num{{font-size:24px}}.tier-grid{{grid-template-columns:1fr}}.stock-row,.stock-header{{grid-template-columns:60px 1fr 60px 60px 60px;font-size:11px;padding:10px 12px}}.layer-label{{width:80px;padding-right:10px}}.layer-label .layer-name{{font-size:10px}}.chain-arrow{{padding-left:80px}}.perf-row{{grid-template-columns:60px 60px 1fr 60px}}.section-header h2{{font-size:20px}}}}
  @media print{{body{{background:#fff;color:#000}}.node,.tier-card,.cr-card,.verdict-card,.perf-container{{border-color:#ddd;background:#fafafa}}.hero{{background:#fff}}.hero h1{{-webkit-text-fill-color:#000}}}}
</style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <p class="date">Investment Research — {config.get('date', datetime.now().strftime('%B %d, %Y'))}</p>
  <h1>{config['title']}</h1>
  <p class="subtitle">{config.get('subtitle', '')}</p>
  <div class="kpi-row">
    <div class="kpi"><div class="num">{total_companies}</div><div class="label">Companies Analyzed</div></div>
    <div class="kpi"><div class="num">{num_layers}</div><div class="label">Supply Chain Layers</div></div>
    <div class="kpi"><div class="num">{fmt_pct(best_1y_val) if best_1y_ticker else '—'}</div><div class="label">Best 1Y Return ({best_1y_ticker})</div></div>
    <div class="kpi"><div class="num">{monopoly_count}</div><div class="label">Monopoly / Oligopoly Moats</div></div>
  </div>
</div>

<div class="container">

<!-- THESIS -->
<section style="margin-top:50px">
  <div class="section-header"><div class="num">0</div><h2>Investment Thesis</h2></div>
  <div class="thesis-grid">
    <div class="thesis-card">
      <div class="tc-label" style="color:var(--cyan)">The Premise</div>
      <p>Every dollar of {config.get('industry', 'this sector')}\'s revenue flows through a small set of critical suppliers.
      These "picks &amp; shovels" companies have <strong>monopoly or oligopoly moats</strong>,
      sell to <strong>multiple end customers simultaneously</strong>,
      and benefit regardless of which end customer wins.</p>
    </div>
    <div class="thesis-card">
      <div class="tc-label" style="color:var(--green)">Why It Works</div>
      <p>During the Gold Rush, the most reliable profits went to those selling pickaxes — not to miners.
      The infrastructure layer has <strong>higher margins</strong>,
      <strong>stickier revenue</strong>, and <strong>less competitive risk</strong> than the application layer.</p>
    </div>
  </div>
</section>
"""

    # ═══ SUPPLY CHAIN MAP ═══
    html += """
<!-- SUPPLY CHAIN -->
<section>
  <div class="section-header"><div class="num">1</div><h2>Supply Chain Architecture</h2><span class="tag">""" + f"{num_layers} Layers Deep" + """</span></div>
  <div class="chain-wrapper">
"""

    # Layer 1
    if config.get("layer1"):
        html += '    <div class="chain-layer"><div class="layer-label"><div class="layer-name">Layer 1</div><div class="layer-desc">End Customers</div></div><div class="layer-nodes">\n'
        for item in config["layer1"]:
            html += f'      <div class="node" style="opacity:0.6"><div class="ticker">{item["ticker"]}</div><div class="role">{item["role"]}</div></div>\n'
        html += '    </div></div>\n'
        html += '    <div class="chain-arrow"><svg viewBox="0 0 24 32"><path d="M12 0 L12 24 M4 18 L12 28 L20 18" stroke="#334155" stroke-width="2" fill="none"/></svg><span class="arrow-label">depends on →</span></div>\n'

    # Layer 2
    if config.get("layer2"):
        html += '    <div class="chain-layer"><div class="layer-label"><div class="layer-name">Layer 2</div><div class="layer-desc">Direct Suppliers</div></div><div class="layer-nodes">\n'
        for item in config["layer2"]:
            moat_cls = f"moat-{item.get('moat', 'tech')}"
            html += f'      <div class="node"><div class="ticker">{item["ticker"]}</div><div class="role">{item["role"]}</div><span class="moat-badge {moat_cls}">{item.get("moat_label", item.get("moat", ""))}</span></div>\n'
        html += '    </div></div>\n'

    # Arrow between L2 and L3
    if config.get("layer3"):
        html += '    <div class="chain-arrow"><svg viewBox="0 0 24 32"><path d="M12 0 L12 24 M4 18 L12 28 L20 18" stroke="#334155" stroke-width="2" fill="none"/></svg><span class="arrow-label">depends on →</span></div>\n'
        html += '    <div class="chain-layer"><div class="layer-label"><div class="layer-name">Layer 3</div><div class="layer-desc">Deep Infrastructure</div></div><div class="layer-nodes">\n'
        for item in config["layer3"]:
            moat_cls = f"moat-{item.get('moat', 'tech')}"
            html += f'      <div class="node"><div class="ticker">{item["ticker"]}</div><div class="role">{item["role"]}</div><span class="moat-badge {moat_cls}">{item.get("moat_label", item.get("moat", ""))}</span></div>\n'
        html += '    </div></div>\n'

    html += '  </div>\n</section>\n'

    # ═══ TIER RANKINGS ═══
    html += '\n<!-- TIER RANKINGS -->\n<section>\n  <div class="section-header"><div class="num">2</div><h2>Tier Rankings</h2><span class="tag">Fundamentals</span></div>\n  <div class="tier-grid">\n'

    tier_order = [1, 2, 3, 0]
    for tier_num in tier_order:
        if tier_num not in tier_groups:
            continue
        items = tier_groups[tier_num]
        tier_info = config.get("tiers", {}).get(tier_num, {"name": f"Tier {tier_num}", "pct": "", "desc": ""})
        tc = tier_colors.get(tier_num, "cyan")

        html += f'    <div class="tier-card tc-{tc}">\n'
        html += f'      <div class="tier-card-header"><span>●</span> {tier_info["name"]}</div>\n'
        html += '      <div class="stock-header"><span>Ticker</span><span>Company</span><span>Margin</span><span>D/E</span><span>Growth</span></div>\n'
        html += '      <div class="tier-card-body">\n'

        for item in items:
            f = item.get("fundamentals", {})
            gm = f.get("gross_margin")
            de = f.get("debt_equity")
            rg = f.get("rev_growth")
            html += f'        <div class="stock-row">\n'
            html += f'          <span class="sticker">{item["ticker"]}</span>\n'
            html += f'          <span class="desc">{item["role"]}</span>\n'
            html += f'          <span class="metric {color_class(gm)}">{fmt(gm, "%")}</span>\n'
            html += f'          <span class="metric {de_color(de)}">{fmt(de)}</span>\n'
            html += f'          <span class="metric {growth_color(rg)}">{fmt_pct(rg) if rg else "—"}</span>\n'
            html += f'        </div>\n'

        html += '      </div>\n    </div>\n'

    html += '  </div>\n</section>\n'

    # ═══ PERFORMANCE BARS ═══
    html += '\n<!-- PERFORMANCE -->\n<section>\n  <div class="section-header"><div class="num">3</div><h2>1-Year Performance</h2><span class="tag">Polygon Data</span></div>\n  <div class="perf-container">\n'

    for item in perf_sorted:
        ret = item.get("return_1y")
        if ret is None:
            continue
        bar_width = min(100, max(8, abs(ret) / max_return * 100))
        bar_class = "positive" if ret >= 0 else "negative"
        tier = item["tier"]
        tc = tier_colors.get(tier, "cyan")
        tier_name = config.get("tiers", {}).get(tier, {}).get("name", "Alt")
        # Shorten tier name for display
        short_tier = {"Core Holdings": "Tier 1", "Growth Bets": "Tier 2", "Deep Infrastructure": "Tier 3", "Contrarian / Alt": "Alt"}.get(tier_name, tier_name)
        price_str = f"${item['price']:,.0f}" if item.get("price") else "—"
        ticker_color = "var(--green)" if ret >= 0 else "var(--red)"

        html += f'    <div class="perf-row">\n'
        html += f'      <span class="pticker" style="color:{ticker_color}">{item["ticker"]}</span>\n'
        html += f'      <span class="ptier" style="color:var(--{tc})">{short_tier}</span>\n'
        html += f'      <div class="perf-bar-wrap"><div class="perf-bar {bar_class}" style="width:{bar_width:.0f}%">{fmt_pct(ret)}</div></div>\n'
        html += f'      <span class="page">{price_str}</span>\n'
        html += f'    </div>\n'

    html += '  </div>\n</section>\n'

    # ═══ TRAJECTORY ANALYSIS ═══
    trajectories = compute_trajectory(config, performance, profiles)
    if trajectories:
        html += '\n<!-- TRAJECTORY ANALYSIS -->\n<section>\n  <div class="section-header"><div class="num">4</div><h2>Trajectory Analysis</h2><span class="tag">Performance &times; Company Age</span></div>\n'

        # Table
        html += '  <div style="overflow-x:auto">\n  <table class="traj-table">\n'
        html += '    <thead><tr><th>Ticker</th><th>Role</th><th>Founded</th><th>Age</th><th>Mkt Cap</th><th>1Y</th><th>2Y</th><th>Velocity</th><th>Trajectory</th><th>Grade</th></tr></thead>\n'
        html += '    <tbody>\n'

        for tr in trajectories:
            ret1 = fmt_pct(tr["return_1y"]) if tr["return_1y"] is not None else "—"
            ret2 = fmt_pct(tr["return_2y"]) if tr["return_2y"] is not None else "—"
            vel_str = "{:.1f}".format(tr["velocity"]) if tr["velocity"] is not None else "—"
            ret1_color = "green" if tr.get("return_1y") and tr["return_1y"] >= 0 else ("red" if tr.get("return_1y") else "muted")
            ret2_color = "green" if tr.get("return_2y") and tr["return_2y"] >= 0 else ("red" if tr.get("return_2y") else "muted")
            label_cls = tr["label"].lower().replace(" ", "")
            grade_cls = tr["grade"].lower().replace("+", "-plus")

            html += '    <tr>'
            html += '<td class="t-ticker">{}</td>'.format(tr["ticker"])
            html += '<td class="t-era">{}</td>'.format(tr["role"])
            html += '<td class="t-center">{}</td>'.format(tr["founded"])
            html += '<td class="t-center">{}y</td>'.format(tr["age"])
            html += '<td class="t-center">{}</td>'.format(tr["market_cap"])
            html += '<td class="t-center" style="color:var(--{})">{}</td>'.format(ret1_color, ret1)
            html += '<td class="t-center" style="color:var(--{})">{}</td>'.format(ret2_color, ret2)
            html += '<td class="t-center" style="color:var(--cyan);font-weight:700">{}</td>'.format(vel_str)
            html += '<td class="t-center"><span class="traj-label {}">{}</span></td>'.format(label_cls, tr["label"])
            html += '<td class="t-center"><span class="traj-grade {}">{}</span></td>'.format(grade_cls, tr["grade"])
            html += '</tr>\n'

        html += '    </tbody>\n  </table>\n  </div>\n'

        # Top 3 Leaders cards
        top3 = [t for t in trajectories if t["velocity"] is not None][:3]
        if top3:
            medals = ["\U0001f947", "\U0001f948", "\U0001f949"]
            html += '  <div class="traj-leaders">\n'
            for i, ldr in enumerate(top3):
                html += '    <div class="traj-leader-card tlc-{}">\n'.format(i + 1)
                html += '      <div class="tl-rank">{}</div>\n'.format(medals[i])
                html += '      <div class="tl-ticker">{} &mdash; {}</div>\n'.format(ldr["ticker"], ldr["label"])
                html += '      <div class="tl-stats">\n'
                html += '        Founded <strong>{}</strong> ({} years old)<br>\n'.format(ldr["founded"], ldr["age"])
                html += '        Market Cap: <strong>{}</strong><br>\n'.format(ldr["market_cap"])
                if ldr["return_2y"] is not None:
                    html += '        2Y Return: <strong>{}</strong><br>\n'.format(fmt_pct(ldr["return_2y"]))
                html += '        Velocity Score: <strong>{:.1f}</strong> (return/age)\n'.format(ldr["velocity"])
                html += '      </div>\n    </div>\n'
            html += '  </div>\n'

        # Generational Timeline
        sorted_by_age = sorted(trajectories, key=lambda x: x["founded"])
        html += '  <h3 style="margin-top:36px;margin-bottom:16px;font-size:18px;font-weight:700;color:var(--muted)">Generational Timeline</h3>\n'
        html += '  <div class="traj-timeline">\n'
        for tr in sorted_by_age:
            html += '    <div class="tl-era"><div class="tl-year">{}</div><div class="tl-name">{}</div><div class="tl-what">{}</div></div>\n'.format(tr["founded"], tr["ticker"], tr["role"])
        html += '  </div>\n'

        html += '</section>\n'

    # ═══ CATALYSTS & RISKS ═══
    next_section = 5 if trajectories else 4
    if config.get("catalysts_risks"):
        html += '\n<!-- CATALYSTS & RISKS -->\n<section>\n  <div class="section-header"><div class="num">{}</div><h2>Catalysts &amp; Risks by Subsector</h2></div>\n  <div class="cr-grid">\n'.format(next_section)

        for cr in config["catalysts_risks"]:
            c = cr.get("color", "green")
            html += f'    <div class="cr-card">\n'
            html += f'      <h3><div class="icon" style="background:rgba({",".join(str(int(color_map.get(c, "#10b981").lstrip("#")[i:i+2], 16)) for i in (0,2,4))},0.15);color:var(--{c})">{cr["icon"]}</div> {cr["name"]}</h3>\n'
            for cat in cr.get("catalysts", []):
                html += f'      <div class="cr-item"><span class="bullet cat">▲</span><span class="text">{cat}</span></div>\n'
            for risk in cr.get("risks", []):
                html += f'      <div class="cr-item"><span class="bullet risk">▼</span><span class="text">{risk}</span></div>\n'
            html += f'    </div>\n'

        html += '  </div>\n</section>\n'

    # ═══ VERDICT / ALLOCATION ═══
    alloc_section = (6 if trajectories else 5) if config.get("catalysts_risks") else (5 if trajectories else 4)
    html += f'\n<!-- ALLOCATION -->\n<section>\n  <div class="section-header"><div class="num">{alloc_section}</div><h2>Portfolio Allocation Framework</h2></div>\n  <div class="verdict-grid">\n'

    for tier_num in tier_order:
        if tier_num not in tier_groups:
            continue
        tier_info = config.get("tiers", {}).get(tier_num, {"name": f"Tier {tier_num}", "pct": "—", "desc": ""})
        tc = tier_colors.get(tier_num, "cyan")
        tickers_str = " · ".join(item["ticker"] for item in tier_groups[tier_num])

        html += f'    <div class="verdict-card vc-{tc}">\n'
        html += f'      <div class="v-label vl-{tc}">{tier_info["name"]} ({tier_info["pct"]})</div>\n'
        html += f'      <div class="v-tickers">{tickers_str}</div>\n'
        html += f'      <div class="v-desc">{tier_info["desc"]}</div>\n'
        html += f'    </div>\n'

    html += '  </div>\n</section>\n'

    # ═══ MOAT LEGEND ═══
    html += """
<!-- MOAT LEGEND -->
<section>
  <div class="moat-legend">
    <div class="moat-legend-item"><span class="moat-badge moat-monopoly" style="font-size:11px">Monopoly</span><span>No competitor exists</span></div>
    <div class="moat-legend-item"><span class="moat-badge moat-oligopoly" style="font-size:11px">Oligopoly</span><span>2-3 players only</span></div>
    <div class="moat-legend-item"><span class="moat-badge moat-tech" style="font-size:11px">Tech Lead</span><span>Multi-year technology gap</span></div>
    <div class="moat-legend-item"><span class="moat-badge moat-scale" style="font-size:11px">Scale</span><span>Size makes replication hard</span></div>
    <div class="moat-legend-item"><span class="moat-badge moat-consumable" style="font-size:11px">Consumable</span><span>Recurring revenue model</span></div>
  </div>
</section>
"""

    # ═══ GENERATED TIMESTAMP ═══
    html += f"""
</div>
<footer>
  Research compiled from Finnhub &amp; Polygon APIs · Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} · Not financial advice<br>
  Built with <a href="#">Decision Point</a> · Picks &amp; Shovels Builder v1.0
</footer>
</body>
</html>"""

    return html


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

def run_stages(config, output_path):
    """Interactive stage-by-stage execution."""
    tickers = get_all_tickers(config)

    print(f"\n{'='*60}")
    print(f"  Picks & Shovels Builder — {config['title']}")
    print(f"  {len(tickers)} tickers across {sum(1 for k in ['layer1','layer2','layer3'] if config.get(k))} layers")
    print(f"{'='*60}\n")

    # Stage 1: Fundamentals
    input(f"📊 STAGE 1: Pull fundamentals for {len(tickers)} tickers from Finnhub?\n   Press Enter to continue (or Ctrl+C to skip)...")
    print()
    fundamentals = fetch_finnhub_fundamentals(tickers)
    print(f"\n   ✅ Got fundamentals for {sum(1 for v in fundamentals.values() if v.get('gross_margin'))} / {len(tickers)} tickers\n")

    # Stage 2: Performance
    input(f"📈 STAGE 2: Pull performance data for {len(tickers)} tickers from Polygon?\n   Press Enter to continue...")
    print()
    performance = fetch_polygon_performance(tickers)
    print(f"\n   ✅ Got prices for {sum(1 for v in performance.values() if v.get('price'))} / {len(tickers)} tickers\n")

    # Stage 3: Generate
    input(f"🖥️  STAGE 3: Generate investor HTML report?\n   Press Enter to continue...")
    print()
    html = generate_html(config, fundamentals, performance)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   ✅ Report saved to: {output_path}")
    print(f"   📄 Open in browser to view\n")

    # Save data cache
    cache_path = output_path.replace(".html", "_data.json")
    with open(cache_path, "w") as f:
        json.dump({"fundamentals": fundamentals, "performance": performance, "config": config}, f, indent=2, default=str)
    print(f"   💾 Data cached to: {cache_path}")


def run_full(config, output_path):
    """Pull everything and generate in one shot."""
    tickers = get_all_tickers(config)

    print(f"\n{'='*60}")
    print(f"  Picks & Shovels Builder — FULL RUN")
    print(f"  {config['title']}")
    print(f"  {len(tickers)} tickers · {sum(1 for k in ['layer1','layer2','layer3'] if config.get(k))} layers")
    print(f"{'='*60}\n")

    print("📊 Pulling fundamentals from Finnhub...")
    fundamentals = fetch_finnhub_fundamentals(tickers)
    print(f"   ✅ {sum(1 for v in fundamentals.values() if v.get('gross_margin'))}/{len(tickers)} complete\n")

    print("📈 Pulling performance from Polygon...")
    performance = fetch_polygon_performance(tickers)
    print(f"   ✅ {sum(1 for v in performance.values() if v.get('price'))}/{len(tickers)} complete\n")

    print("� Pulling company profiles for trajectory analysis...")
    profiles = fetch_company_profiles(tickers, config)
    print(f"   ✅ {sum(1 for v in profiles.values() if v.get('founded') or v.get('ipo_year'))}/{len(tickers)} profiles\n")

    print("🖥️  Generating HTML report...")
    html = generate_html(config, fundamentals, performance, profiles)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   ✅ Report saved to: {output_path}\n")

    # Save data cache
    cache_path = output_path.replace(".html", "_data.json")
    with open(cache_path, "w") as f:
        json.dump({"fundamentals": fundamentals, "performance": performance, "profiles": profiles}, f, indent=2, default=str)
    print(f"   💾 Data cached to: {cache_path}")
    print(f"\n{'='*60}")
    print(f"  ✅ DONE — Open {output_path} in a browser to view")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Picks & Shovels Research Builder")
    parser.add_argument("--stages", action="store_true", help="Interactive step-by-step mode")
    parser.add_argument("--full", action="store_true", help="Pull everything in one shot (default)")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    parser.add_argument("--html-only", action="store_true", help="Skip API calls, use cached data or config defaults")
    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        print(f"📄 Loaded config from {args.config}")
    else:
        config = EXAMPLE_CONFIG
        print("📄 Using built-in AI Infrastructure config (pass --config for custom)")

    # Output path
    if args.output:
        output_path = args.output
    else:
        safe_name = config.get("title", "research").lower().replace(" ", "_").replace("&", "and")
        output_path = os.path.join("reports", f"{safe_name}.html")

    # HTML-only mode (no API calls)
    if args.html_only:
        cache_path = output_path.replace(".html", "_data.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            fundamentals = cached.get("fundamentals", {})
            performance = cached.get("performance", {})
            profiles = cached.get("profiles", {})
        else:
            fundamentals = {}
            performance = {}
            profiles = _profiles_from_config(get_all_tickers(config), config)
        html = generate_html(config, fundamentals, performance, profiles)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"✅ HTML generated (no API calls): {output_path}")
        return

    # Run
    if args.stages:
        run_stages(config, output_path)
    else:
        run_full(config, output_path)


if __name__ == "__main__":
    main()
