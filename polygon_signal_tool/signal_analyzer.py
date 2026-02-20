"""
Core analysis engine for the Polygon Signal Tool.
Replicates all probability/scalp analysis from the SMH Excel workbooks.
"""
from datetime import datetime
from signal_config import HORIZONS, SEVERITY_BUCKETS, WEIGHT_WIN_RATE, WEIGHT_HIT_RATE, WEIGHT_AVG_SCALP, WEIGHT_SEVERITY


def classify_days(data):
    """
    Classify each day as green/red and compute streaks.
    data: list of dicts with keys: date, open, high, low, close, volume
    Returns enriched list with green, red, gstreak, rstreak fields.
    """
    days = []
    for d in data:
        green = d["close"] >= d["open"]
        days.append({
            **d,
            "green": green,
            "red": not green,
        })

    n = len(days)
    for i in range(n):
        gs = rs = 0
        if days[i]["green"]:
            gs = 1
            j = i - 1
            while j >= 0 and days[j]["green"]:
                gs += 1
                j -= 1
        if days[i]["red"]:
            rs = 1
            j = i - 1
            while j >= 0 and days[j]["red"]:
                rs += 1
                j -= 1
        days[i]["gstreak"] = gs
        days[i]["rstreak"] = rs

    return days


def analyze_trades(days, indices, direction):
    """
    Analyze trades for given indices and direction.
    direction: 'call' or 'put'
    For calls: profit when price goes UP (high > entry)
    For puts: profit when price goes DOWN (low < entry)
    """
    n = len(days)
    trades = []
    for i in indices:
        if i >= n - 1:
            continue
        entry = days[i]["close"]
        t = {
            "date": days[i]["date"],
            "entry": entry,
            "day_type": "GREEN" if days[i]["green"] else "RED",
            "gstreak": days[i]["gstreak"],
            "rstreak": days[i]["rstreak"],
            "oc_move": days[i]["close"] - days[i]["open"],
            "oc_pct": (days[i]["close"] - days[i]["open"]) / days[i]["open"] if days[i]["open"] != 0 else 0,
        }

        for horizon in HORIZONS:
            hit = False
            best = 0
            worst = 0
            for look in range(1, horizon + 1):
                if i + look >= n:
                    break
                d = days[i + look]
                if direction == "call":
                    if d["high"] > entry:
                        hit = True
                        best = max(best, d["high"] - entry)
                    worst = min(worst, d["low"] - entry)
                else:
                    if d["low"] < entry:
                        hit = True
                        best = max(best, entry - d["low"])
                    worst = min(worst, entry - d["high"])
            t[f"hit_{horizon}d"] = hit
            t[f"best_{horizon}d"] = best
            t[f"worst_{horizon}d"] = worst

        # Close-based P/L
        for look in [1, 3, 5]:
            if i + look < n:
                if direction == "call":
                    t[f"close_pl_{look}d"] = days[i + look]["close"] - entry
                else:
                    t[f"close_pl_{look}d"] = entry - days[i + look]["close"]
            else:
                t[f"close_pl_{look}d"] = None
        trades.append(t)
    return trades


def compute_stats(trades, label=""):
    """Compute aggregate stats for a set of trades."""
    ct = len(trades)
    if ct == 0:
        return None
    s = {"count": ct, "label": label}
    for h in HORIZONS:
        hits = sum(1 for t in trades if t.get(f"hit_{h}d", False))
        s[f"hit_{h}d"] = hits
        s[f"rate_{h}d"] = hits / ct
        bests = [t[f"best_{h}d"] for t in trades if t.get(f"best_{h}d") is not None]
        s[f"avg_best_{h}d"] = sum(bests) / len(bests) if bests else 0
        # Percentage version: best move / entry price
        best_pcts = [t[f"best_{h}d"] / t["entry"] * 100 for t in trades 
                     if t.get(f"best_{h}d") is not None and t.get("entry", 0) > 0]
        s[f"avg_best_pct_{h}d"] = sum(best_pcts) / len(best_pcts) if best_pcts else 0
        worsts = [t[f"worst_{h}d"] for t in trades if t.get(f"worst_{h}d") is not None]
        s[f"avg_worst_{h}d"] = sum(worsts) / len(worsts) if worsts else 0
        worst_pcts = [t[f"worst_{h}d"] / t["entry"] * 100 for t in trades
                      if t.get(f"worst_{h}d") is not None and t.get("entry", 0) > 0]
        s[f"avg_worst_pct_{h}d"] = sum(worst_pcts) / len(worst_pcts) if worst_pcts else 0
    for look in [1, 3, 5]:
        pls = [t[f"close_pl_{look}d"] for t in trades if t.get(f"close_pl_{look}d") is not None]
        s[f"avg_close_pl_{look}d"] = sum(pls) / len(pls) if pls else 0
        s[f"close_win_{look}d"] = sum(1 for p in pls if p > 0) / len(pls) if pls else 0
    return s


def compute_scalp_ranges(days):
    """
    For every day, compute upside and downside range from close.
    Returns trades list and grouped stats.
    """
    n = len(days)
    trades = []
    for i in range(n - 1):
        entry = days[i]["close"]
        t = {
            "idx": i, "date": days[i]["date"], "entry": entry,
            "green": days[i]["green"], "gs": days[i]["gstreak"], "rs": days[i]["rstreak"],
            "oc_move": days[i]["close"] - days[i]["open"],
            "oc_pct": (days[i]["close"] - days[i]["open"]) / days[i]["open"] if days[i]["open"] else 0,
        }
        for h in HORIZONS:
            max_up = 0
            max_dn = 0
            for look in range(1, h + 1):
                if i + look >= n:
                    break
                d = days[i + look]
                up = d["high"] - entry
                dn = entry - d["low"]
                if up > max_up:
                    max_up = up
                if dn > max_dn:
                    max_dn = dn
            t[f"up_{h}d"] = max_up
            t[f"dn_{h}d"] = max_dn
            t[f"up_pct_{h}d"] = max_up / entry if entry else 0
            t[f"dn_pct_{h}d"] = max_dn / entry if entry else 0
            t[f"ratio_{h}d"] = max_up / max_dn if max_dn > 0 else 99
        trades.append(t)

    def grp_stats(subset, label):
        ct = len(subset)
        if ct == 0:
            return None
        s = {"label": label, "count": ct}
        for h in HORIZONS:
            ups = [t[f"up_{h}d"] for t in subset]
            dns = [t[f"dn_{h}d"] for t in subset]
            up_pcts = [t[f"up_pct_{h}d"] for t in subset]
            dn_pcts = [t[f"dn_pct_{h}d"] for t in subset]
            ratios = [t[f"ratio_{h}d"] for t in subset if t[f"ratio_{h}d"] < 99]
            s[f"avg_up_{h}d"] = sum(ups) / ct
            s[f"avg_dn_{h}d"] = sum(dns) / ct
            s[f"avg_up_pct_{h}d"] = sum(up_pcts) / ct
            s[f"avg_dn_pct_{h}d"] = sum(dn_pcts) / ct
            s[f"avg_ratio_{h}d"] = sum(ratios) / len(ratios) if ratios else 0
            s[f"up_bigger_{h}d"] = sum(1 for t in subset if t[f"up_{h}d"] > t[f"dn_{h}d"])
            s[f"up_bigger_rate_{h}d"] = s[f"up_bigger_{h}d"] / ct
        return s

    all_t = trades
    green_t = [t for t in trades if t["green"]]
    red_t = [t for t in trades if not t["green"]]
    g2 = [t for t in trades if t["gs"] >= 2]
    r2 = [t for t in trades if t["rs"] >= 2]
    g3 = [t for t in trades if t["gs"] >= 3]
    r3 = [t for t in trades if t["rs"] >= 3]

    groups = [
        ("All Days", all_t), ("Green Days", green_t), ("Red Days", red_t),
        ("2+ Green Streak", g2), ("2+ Red Streak", r2),
        ("3+ Green Streak", g3), ("3+ Red Streak", r3),
    ]
    gstats = {lab: grp_stats(sub, lab) for lab, sub in groups}
    return trades, gstats


def compute_straddle(days):
    """Compute daily straddle stats (both call and put at every close).
    Uses ATR-based threshold: a 'hit' requires the move to exceed 25% of ATR."""
    n = len(days)
    
    # Compute 14-day ATR for threshold
    atrs = []
    for i in range(1, n):
        tr = max(
            days[i]["high"] - days[i]["low"],
            abs(days[i]["high"] - days[i-1]["close"]),
            abs(days[i]["low"] - days[i-1]["close"])
        )
        atrs.append(tr)
    
    both_count = 0
    call_only = 0
    put_only = 0
    neither = 0
    total_call_scalp = 0
    total_put_scalp = 0
    total_best = 0
    daily_data = []

    for i in range(n - 1):
        entry = days[i]["close"]
        nd = days[i + 1]
        
        # ATR threshold: use trailing 14-day ATR, minimum 25% of ATR for a 'hit'
        atr_window = atrs[max(0, i-13):i+1] if i < len(atrs) else atrs[-14:]
        atr = sum(atr_window) / len(atr_window) if atr_window else entry * 0.02
        threshold = atr * 0.25  # 25% of ATR minimum move
        
        cs = max(0, nd["high"] - entry)
        ps = max(0, entry - nd["low"])
        call_hit = cs >= threshold
        put_hit = ps >= threshold

        if call_hit and put_hit:
            both_count += 1
        elif call_hit:
            call_only += 1
        elif put_hit:
            put_only += 1
        else:
            neither += 1

        if call_hit:
            total_call_scalp += cs
        if put_hit:
            total_put_scalp += ps
        total_best += max(cs, ps)

        daily_data.append({
            "date": days[i]["date"],
            "entry": entry,
            "call_scalp": cs if call_hit else 0,
            "put_scalp": ps if put_hit else 0,
            "call_hit": call_hit,
            "put_hit": put_hit,
            "both_hit": call_hit and put_hit,
            "best_side": max(cs, ps),
        })

    total_days = n - 1
    return {
        "total_days": total_days,
        "both_count": both_count,
        "both_rate": both_count / total_days if total_days else 0,
        "call_only": call_only,
        "put_only": put_only,
        "neither": neither,
        "at_least_one": (both_count + call_only + put_only),
        "at_least_one_rate": (both_count + call_only + put_only) / total_days if total_days else 0,
        "total_call_scalp": total_call_scalp,
        "total_put_scalp": total_put_scalp,
        "total_best": total_best,
        "avg_daily_best": total_best / total_days if total_days else 0,
        "avg_call_scalp": total_call_scalp / (both_count + call_only) if (both_count + call_only) else 0,
        "avg_put_scalp": total_put_scalp / (both_count + put_only) if (both_count + put_only) else 0,
        "daily_data": daily_data,
    }


# ── NEW: Close Location Value (Relative Close) ──────────────────────────

def compute_close_location(days):
    """
    For each day compute Close Location Value = (Close - Low) / (High - Low).
    0% = closed at the bottom, 100% = closed at the top.
    Then measure next-day gap behaviour based on close position buckets.
    """
    n = len(days)
    records = []
    for i in range(n):
        rng = days[i]["high"] - days[i]["low"]
        clv = (days[i]["close"] - days[i]["low"]) / rng if rng > 0 else 0.5
        records.append({
            "date": days[i]["date"],
            "clv": clv,
            "close": days[i]["close"],
        })

    # Bucket analysis: what happens the NEXT day based on today's CLV
    strong_close = []   # CLV > 0.80
    weak_close = []     # CLV < 0.20
    mid_close = []      # 0.40 - 0.60

    for i in range(n - 1):
        clv = records[i]["clv"]
        next_open = days[i + 1]["open"]
        gap_pct = (next_open - days[i]["close"]) / days[i]["close"] * 100 if days[i]["close"] else 0
        higher_open = next_open > days[i]["close"]

        entry = {"clv": clv, "gap_pct": gap_pct, "higher_open": higher_open}

        if clv >= 0.80:
            strong_close.append(entry)
        elif clv <= 0.20:
            weak_close.append(entry)
        if 0.40 <= clv <= 0.60:
            mid_close.append(entry)

    def _bucket(entries, label):
        ct = len(entries)
        if ct == 0:
            return {"label": label, "count": 0}
        higher = sum(1 for e in entries if e["higher_open"])
        gaps = [e["gap_pct"] for e in entries]
        return {
            "label": label,
            "count": ct,
            "higher_open_rate": round(higher / ct * 100, 1),
            "lower_open_rate": round((ct - higher) / ct * 100, 1),
            "avg_gap_pct": round(sum(gaps) / ct, 3),
        }

    # Trend cluster: count of last 5 days with CLV > 0.80
    recent_strong = sum(1 for r in records[-5:] if r["clv"] >= 0.80) if n >= 5 else 0
    recent_weak = sum(1 for r in records[-5:] if r["clv"] <= 0.20) if n >= 5 else 0

    return {
        "today_clv": round(records[-1]["clv"] * 100, 1) if records else 0,
        "strong_closers": _bucket(strong_close, "CLV > 80%"),
        "weak_closers": _bucket(weak_close, "CLV < 20%"),
        "mid_closers": _bucket(mid_close, "CLV 40-60%"),
        "recent_5d_strong": recent_strong,
        "recent_5d_weak": recent_weak,
        "trend_cluster": "STRONG TREND" if recent_strong >= 3 else ("WEAK TREND" if recent_weak >= 3 else "NEUTRAL"),
    }


# ── NEW: Gap Analysis / Gap Reversion ────────────────────────────────────

def compute_gap_analysis(days):
    """
    Measure gaps (today open vs yesterday close) and gap-fill rates.
    A gap is 'filled' if price trades back to or beyond the previous close
    during the session.
    """
    n = len(days)
    gap_ups = []
    gap_downs = []

    for i in range(1, n):
        prev_close = days[i - 1]["close"]
        today_open = days[i]["open"]
        gap = today_open - prev_close
        gap_pct = gap / prev_close * 100 if prev_close else 0

        # Determine if gap was filled during the session
        if gap > 0:
            # Gap up: filled if today's low <= prev_close
            filled = days[i]["low"] <= prev_close
            gap_ups.append({
                "date": days[i]["date"], "gap_pct": gap_pct, "filled": filled,
                "gap_dollars": gap,
            })
        elif gap < 0:
            # Gap down: filled if today's high >= prev_close
            filled = days[i]["high"] >= prev_close
            gap_downs.append({
                "date": days[i]["date"], "gap_pct": abs(gap_pct), "filled": filled,
                "gap_dollars": abs(gap),
            })

    def _gap_stats(entries, label):
        ct = len(entries)
        if ct == 0:
            return {"label": label, "count": 0}
        fills = sum(1 for e in entries if e["filled"])
        pcts = [e["gap_pct"] for e in entries]
        # Only significant gaps (>0.3%)
        sig = [e for e in entries if e["gap_pct"] >= 0.3]
        sig_ct = len(sig)
        sig_fills = sum(1 for e in sig if e["filled"])
        return {
            "label": label,
            "count": ct,
            "fill_rate": round(fills / ct * 100, 1),
            "avg_gap_pct": round(sum(pcts) / ct, 3),
            "significant_count": sig_ct,
            "significant_fill_rate": round(sig_fills / sig_ct * 100, 1) if sig_ct else 0,
        }

    # Current state: was there a gap today?
    today_gap = 0
    today_gap_pct = 0
    today_gap_filled = False
    if n >= 2:
        prev_c = days[-2]["close"]
        t_open = days[-1]["open"]
        today_gap = t_open - prev_c
        today_gap_pct = today_gap / prev_c * 100 if prev_c else 0
        if today_gap > 0:
            today_gap_filled = days[-1]["low"] <= prev_c
        elif today_gap < 0:
            today_gap_filled = days[-1]["high"] >= prev_c

    return {
        "gap_ups": _gap_stats(gap_ups, "Gap Up"),
        "gap_downs": _gap_stats(gap_downs, "Gap Down"),
        "today_gap_pct": round(today_gap_pct, 3),
        "today_gap_direction": "UP" if today_gap > 0 else ("DOWN" if today_gap < 0 else "FLAT"),
        "today_gap_filled": today_gap_filled,
    }


# ── NEW: Volatility Regime Detection ─────────────────────────────────────

def compute_volatility_regime(days):
    """
    Compute 10-day ATR / 30-day ATR ratio to classify the current regime.
    Ratios:  <0.80 = Squeeze (avoid), 0.80-1.10 = Stable (trend-follow),
             1.20-1.50 = Expanding, >1.50 = Extreme (mean-revert).
    """
    n = len(days)
    if n < 32:
        return {"regime": "INSUFFICIENT DATA", "atr_ratio": 0}

    # Compute True Ranges
    trs = []
    for i in range(1, n):
        tr = max(
            days[i]["high"] - days[i]["low"],
            abs(days[i]["high"] - days[i - 1]["close"]),
            abs(days[i]["low"] - days[i - 1]["close"])
        )
        trs.append(tr)

    # ATR windows
    atr_10 = sum(trs[-10:]) / 10
    atr_30 = sum(trs[-30:]) / 30
    ratio = atr_10 / atr_30 if atr_30 > 0 else 1.0

    # Full ATR for reference
    atr_14 = sum(trs[-14:]) / min(14, len(trs))

    # Classify
    if ratio < 0.80:
        regime = "SQUEEZE"
        action = "Avoid — chop kills accounts. Wait for breakout."
    elif ratio <= 1.10:
        regime = "STABLE"
        action = "Trend-following. Wider stops, let winners run."
    elif ratio <= 1.50:
        regime = "EXPANDING"
        action = "Volatility breakouts. Momentum is high."
    else:
        regime = "EXTREME"
        action = "Mean-reversion fades are highest probability."

    # Historical regime distribution
    regime_counts = {"SQUEEZE": 0, "STABLE": 0, "EXPANDING": 0, "EXTREME": 0}
    for i in range(30, len(trs)):
        a10 = sum(trs[i - 9:i + 1]) / 10
        a30 = sum(trs[i - 29:i + 1]) / 30
        r = a10 / a30 if a30 > 0 else 1.0
        if r < 0.80:
            regime_counts["SQUEEZE"] += 1
        elif r <= 1.10:
            regime_counts["STABLE"] += 1
        elif r <= 1.50:
            regime_counts["EXPANDING"] += 1
        else:
            regime_counts["EXTREME"] += 1
    total_periods = sum(regime_counts.values())
    regime_pcts = {k: round(v / total_periods * 100, 1) if total_periods else 0 for k, v in regime_counts.items()}

    return {
        "regime": regime,
        "action": action,
        "atr_ratio": round(ratio, 3),
        "atr_10": round(atr_10, 4),
        "atr_30": round(atr_30, 4),
        "atr_14": round(atr_14, 4),
        "regime_distribution": regime_pcts,
    }


# ── NEW: Extension Z-Score (Overextension Detection) ─────────────────────

def compute_extension_zscore(days):
    """
    Z-Score of today's range vs 30-day average range.
    Z > 2.0 = 'blow-off top' or 'panic bottom' — historically followed by
    sideways/reversal 90% of the time in next 3 days.
    """
    n = len(days)
    if n < 31:
        return {"zscore": 0, "status": "INSUFFICIENT DATA"}

    # Daily ranges (High - Low)
    ranges = [days[i]["high"] - days[i]["low"] for i in range(n)]

    # Today's range
    today_range = ranges[-1]

    # 30-day stats (excluding today)
    window = ranges[-31:-1]
    avg_range = sum(window) / len(window)
    variance = sum((r - avg_range) ** 2 for r in window) / len(window)
    std_range = variance ** 0.5

    zscore = (today_range - avg_range) / std_range if std_range > 0 else 0

    # Today's move from open as percentage
    today_move_pct = abs(days[-1]["close"] - days[-1]["open"]) / days[-1]["open"] * 100 if days[-1]["open"] else 0

    # Extension limit: what % of avg daily range has today consumed
    extension_pct = today_range / avg_range * 100 if avg_range > 0 else 0

    # Status
    if zscore >= 2.0:
        status = "EXTREME — high probability of sideways/reversal next 1-3 days"
    elif zscore >= 1.5:
        status = "STRETCHED — momentum fading, caution on continuation"
    elif zscore <= -1.0:
        status = "COMPRESSED — coiling for potential breakout"
    else:
        status = "NORMAL"

    # Historical proof: when Z > 2, what happens next 3 days?
    z_extreme_events = 0
    z_extreme_reversals = 0
    for i in range(31, n - 3):
        w = ranges[i - 30:i]
        mu = sum(w) / len(w)
        var = sum((r - mu) ** 2 for r in w) / len(w)
        sd = var ** 0.5
        z = (ranges[i] - mu) / sd if sd > 0 else 0
        if z >= 2.0:
            z_extreme_events += 1
            # Check next 3 days: did range contract or reverse?
            next_3_ranges = ranges[i + 1:i + 4]
            if next_3_ranges:
                avg_next_3 = sum(next_3_ranges) / len(next_3_ranges)
                if avg_next_3 < mu * 1.1:  # range contracted back toward normal
                    z_extreme_reversals += 1

    revert_rate = round(z_extreme_reversals / z_extreme_events * 100, 1) if z_extreme_events else 0

    return {
        "zscore": round(zscore, 2),
        "status": status,
        "today_range": round(today_range, 2),
        "avg_30d_range": round(avg_range, 2),
        "extension_pct": round(extension_pct, 1),
        "today_move_pct": round(today_move_pct, 2),
        "extreme_events": z_extreme_events,
        "revert_after_extreme_rate": revert_rate,
    }


# ── NEW: OpEx Behavior Analysis ──────────────────────────────────────────

def compute_opex_behavior(days):
    """
    Detect Options Expiration days (3rd Friday of each month) and measure
    how price behaviour differs on those days — pinning, reduced range, etc.
    """
    from datetime import datetime as _dt

    def _is_third_friday(d):
        """Check if a date string (YYYY-MM-DD) is the 3rd Friday of its month."""
        try:
            dt = _dt.strptime(d, "%Y-%m-%d")
        except (ValueError, TypeError):
            return False
        if dt.weekday() != 4:  # Friday = 4
            return False
        day = dt.day
        return 15 <= day <= 21

    n = len(days)
    opex_days = []
    non_opex_days = []

    for i in range(n):
        rng = days[i]["high"] - days[i]["low"]
        rng_pct = rng / days[i]["open"] * 100 if days[i]["open"] else 0
        clv = (days[i]["close"] - days[i]["low"]) / rng if rng > 0 else 0.5
        mid_range = abs(clv - 0.5) < 0.15  # closed within 35-65% of range = "pinned"

        entry = {
            "date": days[i]["date"],
            "range_pct": rng_pct,
            "clv": clv,
            "pinned": mid_range,
            "green": days[i]["green"],
        }

        if _is_third_friday(days[i]["date"]):
            opex_days.append(entry)
        else:
            non_opex_days.append(entry)

    def _stats(entries, label):
        ct = len(entries)
        if ct == 0:
            return {"label": label, "count": 0}
        ranges = [e["range_pct"] for e in entries]
        pins = sum(1 for e in entries if e["pinned"])
        greens = sum(1 for e in entries if e["green"])
        return {
            "label": label,
            "count": ct,
            "avg_range_pct": round(sum(ranges) / ct, 3),
            "pin_rate": round(pins / ct * 100, 1),
            "green_rate": round(greens / ct * 100, 1),
        }

    # Is today OpEx?
    today_opex = _is_third_friday(days[-1]["date"]) if days else False

    return {
        "opex": _stats(opex_days, "OpEx Days"),
        "non_opex": _stats(non_opex_days, "Non-OpEx Days"),
        "today_is_opex": today_opex,
        "range_reduction": round(
            (1 - _stats(opex_days, "")["avg_range_pct"] / _stats(non_opex_days, "")["avg_range_pct"]) * 100, 1
        ) if _stats(non_opex_days, "").get("avg_range_pct", 0) > 0 and _stats(opex_days, "").get("avg_range_pct", 0) > 0 else 0,
    }


def run_full_analysis(days):
    """
    Run the complete analysis pipeline.
    Returns dict with all scenario stats, scalp ranges, straddle, and signal.
    """
    n = len(days)

    # Build index sets
    green_idx = [i for i in range(n) if days[i]["green"]]
    red_idx = [i for i in range(n) if days[i]["red"]]
    all_idx = list(range(n))
    green2_idx = [i for i in range(n) if days[i]["gstreak"] >= 2]
    red2_idx = [i for i in range(n) if days[i]["rstreak"] >= 2]
    green3_idx = [i for i in range(n) if days[i]["gstreak"] >= 3]
    red3_idx = [i for i in range(n) if days[i]["rstreak"] >= 3]

    # Run all call/put scenarios
    scenarios = {}
    for prefix, idx_set in [
        ("call_all", all_idx), ("call_green", green_idx), ("call_red", red_idx),
        ("call_green2", green2_idx), ("call_red2", red2_idx),
        ("call_green3", green3_idx), ("call_red3", red3_idx),
    ]:
        scenarios[prefix] = analyze_trades(days, idx_set, "call")

    for prefix, idx_set in [
        ("put_all", all_idx), ("put_green", green_idx), ("put_red", red_idx),
        ("put_green2", green2_idx), ("put_red2", red2_idx),
        ("put_green3", green3_idx), ("put_red3", red3_idx),
    ]:
        scenarios[prefix] = analyze_trades(days, idx_set, "put")

    # Compute stats for each scenario
    all_stats = {}
    for k, v in scenarios.items():
        all_stats[k] = compute_stats(v, k)

    # Scalp ranges
    range_trades, range_groups = compute_scalp_ranges(days)

    # Straddle
    straddle = compute_straddle(days)

    # Severity analysis for red days
    severity_stats = {}
    red_trades = scenarios.get("call_red", [])
    for label, lo, hi in SEVERITY_BUCKETS:
        bucket = [t for t in red_trades if lo <= abs(t["oc_pct"]) < hi]
        severity_stats[label] = compute_stats(bucket, label)

    # ── New predictability modules ──
    close_location = compute_close_location(days)
    gap_analysis = compute_gap_analysis(days)
    vol_regime = compute_volatility_regime(days)
    extension = compute_extension_zscore(days)
    opex = compute_opex_behavior(days)

    # Generate signal
    signal = generate_signal(days, all_stats, range_groups, straddle, severity_stats)

    return {
        "days": days,
        "n": n,
        "scenarios": scenarios,
        "all_stats": all_stats,
        "range_trades": range_trades,
        "range_groups": range_groups,
        "straddle": straddle,
        "severity_stats": severity_stats,
        "close_location": close_location,
        "gap_analysis": gap_analysis,
        "vol_regime": vol_regime,
        "extension": extension,
        "opex": opex,
        "signal": signal,
        "green_count": len(green_idx),
        "red_count": len(red_idx),
    }


def generate_signal(days, all_stats, range_groups, straddle, severity_stats):
    """
    Generate a trading signal based on current market conditions.
    Uses the last day's data to determine current state, then looks up
    historical probabilities for that condition.
    """
    if len(days) < 2:
        return {"recommendation": "INSUFFICIENT DATA", "confidence": 0}

    today = days[-1]
    today_color = "GREEN" if today["green"] else "RED"
    rstreak = today["rstreak"]
    gstreak = today["gstreak"]

    # Determine current severity (if red)
    oc_pct = abs((today["close"] - today["open"]) / today["open"]) if today["open"] else 0
    severity_label = "N/A"
    for label, lo, hi in SEVERITY_BUCKETS:
        if lo <= oc_pct < hi:
            severity_label = label
            break

    # Evaluate each possible action
    candidates = []

    # 1. Calls based on current condition
    if today["red"]:
        if rstreak >= 3:
            key = "call_red3"
        elif rstreak >= 2:
            key = "call_red2"
        else:
            key = "call_red"
        label = f"CALL (after {rstreak} red day{'s' if rstreak > 1 else ''})"
    else:
        if gstreak >= 3:
            key = "call_green3"
        elif gstreak >= 2:
            key = "call_green2"
        else:
            key = "call_green"
        label = f"CALL (after {gstreak} green day{'s' if gstreak > 1 else ''})"

    cs = all_stats.get(key)
    if cs:
        # Score: weighted combination
        win_score = cs.get("close_win_1d", 0)
        hit_score = cs.get("rate_3d", 0)
        # Normalize avg scalp to 0-1 range (assume $20 max)
        scalp_norm = min(cs.get("avg_best_3d", 0) / 20, 1.0)
        # Severity bonus: bigger drops = better call odds
        sev_bonus = 0
        if today["red"]:
            sev_bonus = min(oc_pct / 0.03, 1.0)  # max at 3%

        call_score = (
            WEIGHT_WIN_RATE * win_score +
            WEIGHT_HIT_RATE * hit_score +
            WEIGHT_AVG_SCALP * scalp_norm +
            WEIGHT_SEVERITY * sev_bonus
        )
        candidates.append({
            "direction": "CALL",
            "label": label,
            "key": key,
            "score": call_score,
            "stats": cs,
            "hit_3d": cs.get("rate_3d", 0),
            "avg_best_3d": cs.get("avg_best_3d", 0),
            "close_win_1d": cs.get("close_win_1d", 0),
        })

    # 2. Puts based on current condition
    if today["red"]:
        if rstreak >= 3:
            key = "put_red3"
        elif rstreak >= 2:
            key = "put_red2"
        else:
            key = "put_red"
        label = f"PUT (after {rstreak} red day{'s' if rstreak > 1 else ''})"
    else:
        if gstreak >= 3:
            key = "put_green3"
        elif gstreak >= 2:
            key = "put_green2"
        else:
            key = "put_green"
        label = f"PUT (after {gstreak} green day{'s' if gstreak > 1 else ''})"

    ps = all_stats.get(key)
    if ps:
        win_score = ps.get("close_win_1d", 0)
        hit_score = ps.get("rate_3d", 0)
        scalp_norm = min(ps.get("avg_best_3d", 0) / 20, 1.0)
        sev_bonus = 0
        if today["green"]:
            sev_bonus = min(oc_pct / 0.03, 1.0)

        put_score = (
            WEIGHT_WIN_RATE * win_score +
            WEIGHT_HIT_RATE * hit_score +
            WEIGHT_AVG_SCALP * scalp_norm +
            WEIGHT_SEVERITY * sev_bonus
        )
        candidates.append({
            "direction": "PUT",
            "label": label,
            "key": key,
            "score": put_score,
            "stats": ps,
            "hit_3d": ps.get("rate_3d", 0),
            "avg_best_3d": ps.get("avg_best_3d", 0),
            "close_win_1d": ps.get("close_win_1d", 0),
        })

    # 3. Straddle is always an option
    straddle_score = straddle["at_least_one_rate"] * 0.7 + straddle["both_rate"] * 0.3
    candidates.append({
        "direction": "STRADDLE",
        "label": "STRADDLE (both sides)",
        "key": "straddle",
        "score": straddle_score,
        "stats": straddle,
        "hit_3d": straddle["at_least_one_rate"],
        "avg_best_3d": straddle["avg_daily_best"],
        "close_win_1d": straddle["both_rate"],
    })

    # Pick best
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    # Confidence: normalize score to 0-100
    confidence = min(round(best["score"] * 100), 99)

    # Get scalp ranges for the current condition
    rg = range_groups
    if today["red"]:
        if rstreak >= 2:
            range_key = "2+ Red Streak"
        else:
            range_key = "Red Days"
    else:
        if gstreak >= 2:
            range_key = "2+ Green Streak"
        else:
            range_key = "Green Days"

    range_data = rg.get(range_key, rg.get("All Days"))

    return {
        "recommendation": best["direction"],
        "label": best["label"],
        "confidence": confidence,
        "score": best["score"],
        "hit_3d": best["hit_3d"],
        "avg_best_3d": best["avg_best_3d"],
        "close_win_1d": best["close_win_1d"],
        "candidates": candidates,
        "today": {
            "date": today["date"],
            "open": today["open"],
            "high": today["high"],
            "low": today["low"],
            "close": today["close"],
            "color": today_color,
            "rstreak": rstreak,
            "gstreak": gstreak,
            "oc_pct": oc_pct,
            "severity": severity_label,
        },
        "range_condition": range_key,
        "expected_upside": range_data[f"avg_up_pct_1d"] * 100 if range_data else 0,
        "expected_downside": range_data[f"avg_dn_pct_1d"] * 100 if range_data else 0,
        "expected_upside_3d": range_data[f"avg_up_pct_3d"] * 100 if range_data else 0,
        "expected_downside_3d": range_data[f"avg_dn_pct_3d"] * 100 if range_data else 0,
        # Keep dollar amounts for reference
        "expected_upside_dollars": range_data[f"avg_up_1d"] if range_data else 0,
        "expected_downside_dollars": range_data[f"avg_dn_1d"] if range_data else 0,
    }
