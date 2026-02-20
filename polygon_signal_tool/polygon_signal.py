#!/usr/bin/env python3
"""
Polygon Signal Tool — Automated Trading Signal Dashboard
Fetches OHLCV data from Polygon.io, runs probability analysis, generates HTML dashboard.

Usage:
    python polygon_signal.py SMH
    python polygon_signal.py AAPL --days 180
    python polygon_signal.py SMH --api-key YOUR_KEY
"""
import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime, timedelta

import requests
from jinja2 import Template

from signal_config import POLYGON_API_KEY, POLYGON_BASE_URL, CACHE_DIR, DEFAULT_LOOKBACK_DAYS
from signal_analyzer import classify_days, run_full_analysis


def fetch_polygon_data(ticker, days_back, api_key):
    """Fetch daily OHLCV from Polygon.io with smart caching."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{ticker.upper()}.json")

    cached_data = []
    last_cached_date = None

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
        if cached_data:
            last_cached_date = cached_data[-1]["date"]
            print(f"  Cache has {len(cached_data)} days through {last_cached_date}")

    # Determine date range to fetch
    end_date = datetime.now()
    if last_cached_date:
        # Delta sync: fetch only from day after last cached
        start_date = datetime.strptime(last_cached_date, "%Y-%m-%d") + timedelta(days=1)
        if start_date.date() > end_date.date():
            print("  Cache is up to date!")
            # Still enforce lookback window
            cutoff = (end_date - timedelta(days=days_back)).strftime("%Y-%m-%d")
            filtered = [d for d in cached_data if d["date"] >= cutoff]
            return filtered
    else:
        start_date = end_date - timedelta(days=days_back + 30)  # extra buffer for weekends

    from_str = start_date.strftime("%Y-%m-%d")
    to_str = end_date.strftime("%Y-%m-%d")

    print(f"  Fetching {ticker} from Polygon.io: {from_str} to {to_str}...")

    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker.upper()}/range/1/day/{from_str}/{to_str}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": api_key,
    }

    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        print(f"  ERROR: Polygon API returned {resp.status_code}")
        try:
            err = resp.json()
            print(f"  Message: {err.get('message', err.get('error', 'Unknown'))}")
        except Exception:
            print(f"  Response: {resp.text[:200]}")
        if cached_data:
            print("  Using cached data instead.")
            cutoff = (end_date - timedelta(days=days_back)).strftime("%Y-%m-%d")
            return [d for d in cached_data if d["date"] >= cutoff]
        sys.exit(1)

    body = resp.json()
    results = body.get("results", [])
    if not results:
        print(f"  WARNING: No data returned from Polygon for {ticker}")
        if cached_data:
            print("  Using cached data.")
            cutoff = (end_date - timedelta(days=days_back)).strftime("%Y-%m-%d")
            return [d for d in cached_data if d["date"] >= cutoff]
        sys.exit(1)

    print(f"  Got {len(results)} new bars from API")

    # Parse API response
    new_data = []
    for bar in results:
        ts = bar["t"] / 1000  # Polygon timestamps are in ms
        dt = datetime.utcfromtimestamp(ts)
        date_str = dt.strftime("%Y-%m-%d")
        new_data.append({
            "date": date_str,
            "open": bar["o"],
            "high": bar["h"],
            "low": bar["l"],
            "close": bar["c"],
            "volume": bar.get("v", 0),
        })

    # Merge with cache (avoid duplicates)
    if cached_data:
        existing_dates = {d["date"] for d in cached_data}
        for nd in new_data:
            if nd["date"] not in existing_dates:
                cached_data.append(nd)
        cached_data.sort(key=lambda x: x["date"])
        merged = cached_data
    else:
        merged = new_data

    # Save cache
    with open(cache_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  Cache updated: {len(merged)} total days saved")

    # Apply lookback filter
    cutoff = (end_date - timedelta(days=days_back)).strftime("%Y-%m-%d")
    filtered = [d for d in merged if d["date"] >= cutoff]
    print(f"  Using {len(filtered)} days for analysis (last {days_back} days)")
    return filtered


def render_dashboard(ticker, analysis, output_path):
    """Render HTML dashboard from analysis results."""
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "dashboard.html")
    with open(template_path, "r") as f:
        tmpl = Template(f.read())

    # Prepare data for template
    signal = analysis["signal"]
    stats = analysis["all_stats"]
    straddle = analysis["straddle"]
    rg = analysis["range_groups"]
    days = analysis["days"]

    # Recent 30 days for history table
    recent = days[-30:] if len(days) >= 30 else days

    # Scenario comparison data
    scenario_rows = []
    for label, direction, key in [
        ("Every Day", "CALL", "call_all"),
        ("Green Days", "CALL", "call_green"),
        ("Red Days", "CALL", "call_red"),
        ("2+ Green", "CALL", "call_green2"),
        ("2+ Red", "CALL", "call_red2"),
        ("3+ Green", "CALL", "call_green3"),
        ("3+ Red", "CALL", "call_red3"),
        ("Every Day", "PUT", "put_all"),
        ("Green Days", "PUT", "put_green"),
        ("Red Days", "PUT", "put_red"),
        ("2+ Green", "PUT", "put_green2"),
        ("2+ Red", "PUT", "put_red2"),
        ("3+ Green", "PUT", "put_green3"),
        ("3+ Red", "PUT", "put_red3"),
    ]:
        s = stats.get(key)
        if s:
            scenario_rows.append({
                "label": label, "direction": direction,
                "count": s["count"],
                "rate_1d": s["rate_1d"], "rate_2d": s["rate_2d"],
                "rate_3d": s["rate_3d"], "rate_5d": s["rate_5d"],
                "avg_best_3d": s["avg_best_3d"],
                "avg_worst_3d": s["avg_worst_3d"],
                "close_win_1d": s["close_win_1d"],
            })

    # Range comparison data
    range_rows = []
    for label in ["All Days", "Green Days", "Red Days", "2+ Green Streak", "2+ Red Streak", "3+ Green Streak", "3+ Red Streak"]:
        r = rg.get(label)
        if r:
            range_rows.append({
                "label": label, "count": r["count"],
                "avg_up_1d": r["avg_up_1d"], "avg_dn_1d": r["avg_dn_1d"],
                "up_wins_1d": r["up_bigger_rate_1d"],
                "avg_up_3d": r["avg_up_3d"], "avg_dn_3d": r["avg_dn_3d"],
                "up_wins_3d": r["up_bigger_rate_3d"],
            })

    # Period performance
    if len(days) >= 2:
        period_change = days[-1]["close"] - days[0]["close"]
        period_pct = period_change / days[0]["close"]
    else:
        period_change = 0
        period_pct = 0

    html = tmpl.render(
        ticker=ticker.upper(),
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        signal=signal,
        scenario_rows=scenario_rows,
        range_rows=range_rows,
        straddle=straddle,
        recent=recent,
        n_days=analysis["n"],
        green_count=analysis["green_count"],
        red_count=analysis["red_count"],
        period_change=period_change,
        period_pct=period_pct,
        severity_stats=analysis["severity_stats"],
    )

    with open(output_path, "w") as f:
        f.write(html)
    print(f"\nDashboard saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Polygon Signal Tool — Trading Signal Dashboard")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., SMH, AAPL, TSLA)")
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS, help=f"Lookback period in days (default: {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--api-key", help="Polygon.io API key (or set POLYGON_API_KEY env var)")
    parser.add_argument("--output", help="Output HTML path (default: {ticker}_dashboard.html)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open in browser")
    args = parser.parse_args()

    api_key = args.api_key or POLYGON_API_KEY
    if not api_key:
        print("ERROR: No Polygon.io API key provided.")
        print("Set POLYGON_API_KEY environment variable or use --api-key flag.")
        sys.exit(1)

    ticker = args.ticker.upper()
    print(f"\n{'='*60}")
    print(f"  POLYGON SIGNAL TOOL — {ticker}")
    print(f"{'='*60}")

    # Step 1: Fetch data
    print(f"\n[1/3] Fetching data...")
    raw_data = fetch_polygon_data(ticker, args.days, api_key)
    if not raw_data or len(raw_data) < 10:
        print(f"ERROR: Not enough data ({len(raw_data) if raw_data else 0} days). Need at least 10.")
        sys.exit(1)

    # Step 2: Run analysis
    print(f"\n[2/3] Running analysis on {len(raw_data)} trading days...")
    days = classify_days(raw_data)
    analysis = run_full_analysis(days)

    sig = analysis["signal"]
    print(f"\n  Signal: {sig['recommendation']}")
    print(f"  Confidence: {sig['confidence']}%")
    print(f"  3D Scalp Hit Rate: {sig['hit_3d']*100:.1f}%")
    print(f"  Avg Best Scalp 3D: ${sig['avg_best_3d']:.2f}")
    print(f"  Today: {sig['today']['color']} day (streak: R{sig['today']['rstreak']} G{sig['today']['gstreak']})")

    # Step 3: Generate dashboard
    print(f"\n[3/3] Generating dashboard...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "mnt", "outputs")
    if not os.path.isdir(output_dir):
        output_dir = script_dir
    output_path = args.output or os.path.join(output_dir, f"{ticker}_dashboard.html")
    render_dashboard(ticker, analysis, output_path)

    if not args.no_open:
        try:
            webbrowser.open(f"file://{os.path.abspath(output_path)}")
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"  Done! Dashboard: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
