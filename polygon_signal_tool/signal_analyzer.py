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
        worsts = [t[f"worst_{h}d"] for t in trades if t.get(f"worst_{h}d") is not None]
        s[f"avg_worst_{h}d"] = sum(worsts) / len(worsts) if worsts else 0
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
    """Compute daily straddle stats (both call and put at every close)."""
    n = len(days)
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
        call_hit = nd["high"] > entry
        put_hit = nd["low"] < entry
        cs = max(0, nd["high"] - entry)
        ps = max(0, entry - nd["low"])

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
        "expected_upside": range_data[f"avg_up_1d"] if range_data else 0,
        "expected_downside": range_data[f"avg_dn_1d"] if range_data else 0,
        "expected_upside_3d": range_data[f"avg_up_3d"] if range_data else 0,
        "expected_downside_3d": range_data[f"avg_dn_3d"] if range_data else 0,
    }
