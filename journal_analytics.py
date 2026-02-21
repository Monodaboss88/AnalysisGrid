"""
Journal Analytics Engine
=========================
Computes rich trading performance analytics from trade data
(Firestore or SQLite). Answers the questions every trader needs:

  - Win rate by setup type, symbol, direction, timeframe, signal
  - Avg R:R by scanner source
  - Performance over time (equity curve, monthly, weekly)
  - Strategy-level P&L attribution
  - Streak tracking (current, best, worst)
  - Risk metrics (max drawdown, profit factor, expectancy)
  - Day-of-week and time patterns
  - Best/worst symbols

Works on a list of trade dicts — agnostic to storage backend.

Author: Rob's Trading Systems
"""

from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional
import math


def _safe_div(a: float, b: float, default: float = 0) -> float:
    return a / b if b else default


def _parse_dt(ts: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts:
        return None
    try:
        # Handle various ISO formats
        ts = ts.replace("Z", "+00:00")
        if "T" in ts:
            return datetime.fromisoformat(ts.split("+")[0].split("Z")[0])
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _compute_r_multiple(trade: dict) -> Optional[float]:
    """Compute R-multiple from trade fields."""
    # Use stored value if available
    if trade.get("pnl_r") is not None and trade["pnl_r"] != 0:
        return trade["pnl_r"]
    if trade.get("r_multiple") is not None:
        return trade["r_multiple"]

    entry = trade.get("entry") or trade.get("entry_price") or trade.get("actual_entry") or 0
    stop = trade.get("stop") or trade.get("stop_loss") or 0
    exit_p = trade.get("exit_price") or trade.get("actual_exit") or 0
    direction = (trade.get("direction") or "LONG").upper()

    if not entry or not stop or not exit_p:
        return None

    risk = abs(entry - stop)
    if risk < 0.001:
        return None

    if direction == "LONG":
        return round((exit_p - entry) / risk, 2)
    else:
        return round((entry - exit_p) / risk, 2)


def _get_pnl(trade: dict) -> Optional[float]:
    """Extract P&L from trade."""
    pnl = trade.get("pnl") or trade.get("pnl_dollars")
    if pnl is not None:
        return float(pnl)
    # Compute from entry/exit
    entry = trade.get("entry") or trade.get("entry_price") or 0
    exit_p = trade.get("exit_price") or trade.get("actual_exit") or 0
    direction = (trade.get("direction") or "LONG").upper()
    if entry and exit_p:
        return (exit_p - entry) if direction == "LONG" else (entry - exit_p)
    return None


def _is_win(trade: dict) -> bool:
    status = (trade.get("status") or "").upper()
    if status in ("WIN", "CLOSED_WIN"):
        return True
    if status in ("LOSS", "CLOSED_LOSS"):
        return False
    pnl = _get_pnl(trade)
    if pnl is not None:
        return pnl > 0
    return False


def _is_loss(trade: dict) -> bool:
    status = (trade.get("status") or "").upper()
    if status in ("LOSS", "CLOSED_LOSS"):
        return True
    if status in ("WIN", "CLOSED_WIN"):
        return False
    pnl = _get_pnl(trade)
    if pnl is not None:
        return pnl < 0
    return False


def _is_closed(trade: dict) -> bool:
    status = (trade.get("status") or "").upper()
    return status in ("WIN", "LOSS", "BREAKEVEN", "CLOSED_WIN", "CLOSED_LOSS",
                       "CLOSED_BE", "CLOSED", "closed")


# ── Main Analytics Function ──

def compute_journal_analytics(trades: List[Dict], days: int = 90) -> Dict:
    """
    Compute comprehensive trading analytics from a list of trade dicts.

    Returns a single analytics object with every breakdown a trader needs.
    """
    if not trades:
        return {"total_trades": 0, "message": "No trades to analyze"}

    # Separate closed vs all
    closed = [t for t in trades if _is_closed(t)]
    wins = [t for t in closed if _is_win(t)]
    losses = [t for t in closed if _is_loss(t)]
    open_trades = [t for t in trades if (t.get("status") or "").upper() in
                   ("OPEN", "ACTIVE", "PENDING", "pending", "active")]

    # ── Core Metrics ──
    win_rate = _safe_div(len(wins), len(closed)) * 100

    r_values = [_compute_r_multiple(t) for t in closed]
    r_values = [r for r in r_values if r is not None]
    avg_r = _safe_div(sum(r_values), len(r_values))

    win_r = [r for r in [_compute_r_multiple(t) for t in wins] if r is not None]
    loss_r = [r for r in [_compute_r_multiple(t) for t in losses] if r is not None]
    avg_win_r = _safe_div(sum(win_r), len(win_r))
    avg_loss_r = _safe_div(sum(loss_r), len(loss_r))

    # Expectancy per trade in R
    expectancy = (win_rate / 100 * avg_win_r) + ((1 - win_rate / 100) * avg_loss_r)

    # Profit factor
    gross_profit = sum(r for r in r_values if r > 0)
    gross_loss = abs(sum(r for r in r_values if r < 0))
    profit_factor = _safe_div(gross_profit, gross_loss)

    # P&L
    pnl_values = [_get_pnl(t) for t in closed]
    pnl_values = [p for p in pnl_values if p is not None]
    total_pnl = sum(pnl_values)

    # ── Streaks ──
    streaks = _compute_streaks(closed)

    # ── By Symbol ──
    by_symbol = _group_analytics(closed, lambda t: (t.get("symbol") or "?").upper())

    # ── By Direction ──
    by_direction = _group_analytics(closed, lambda t: (t.get("direction") or "LONG").upper())

    # ── By Timeframe ──
    by_timeframe = _group_analytics(closed, lambda t: t.get("timeframe") or "unknown")

    # ── By Signal (GREEN/YELLOW/RED) ──
    by_signal = _group_analytics(closed, lambda t: (t.get("signal") or "UNKNOWN").upper())

    # ── By Setup Grade ──
    by_grade = _group_analytics(
        [t for t in closed if t.get("setup_grade")],
        lambda t: t.get("setup_grade", "?")
    )

    # ── By Tags ──
    by_tag = _compute_tag_analytics(closed)

    # ── By Day of Week ──
    by_dow = _compute_dow_analytics(closed)

    # ── Monthly Performance ──
    monthly = _compute_monthly(closed)

    # ── Weekly Performance ──
    weekly = _compute_weekly(closed)

    # ── Equity Curve ──
    equity_curve = _compute_equity_curve(closed)

    # ── Max Drawdown ──
    max_dd = _compute_max_drawdown(equity_curve)

    # ── Best / Worst Trades ──
    sorted_by_r = sorted(
        [(t, _compute_r_multiple(t)) for t in closed if _compute_r_multiple(t) is not None],
        key=lambda x: x[1], reverse=True
    )
    best_trades = [
        {"symbol": t.get("symbol"), "direction": t.get("direction"),
         "r": r, "date": t.get("closed_at") or t.get("created_at")}
        for t, r in sorted_by_r[:5]
    ]
    worst_trades = [
        {"symbol": t.get("symbol"), "direction": t.get("direction"),
         "r": r, "date": t.get("closed_at") or t.get("created_at")}
        for t, r in sorted_by_r[-5:]
    ]

    # ── Holding Time Analysis ──
    holding = _compute_holding_times(closed)

    return {
        "period_days": days,
        "total_trades": len(trades),
        "open_trades": len(open_trades),
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(closed) - len(wins) - len(losses),
        "win_rate": round(win_rate, 1),
        "avg_r": round(avg_r, 2),
        "avg_win_r": round(avg_win_r, 2),
        "avg_loss_r": round(avg_loss_r, 2),
        "expectancy_r": round(expectancy, 3),
        "profit_factor": round(profit_factor, 2),
        "total_pnl": round(total_pnl, 2),
        "streaks": streaks,
        "by_symbol": by_symbol,
        "by_direction": by_direction,
        "by_timeframe": by_timeframe,
        "by_signal": by_signal,
        "by_setup_grade": by_grade,
        "by_tag": by_tag,
        "by_day_of_week": by_dow,
        "monthly": monthly,
        "weekly": weekly,
        "equity_curve": equity_curve,
        "max_drawdown_r": round(max_dd, 2),
        "best_trades": best_trades,
        "worst_trades": worst_trades,
        "holding_time": holding,
    }


# ── Group Analytics Helper ──

def _group_analytics(trades: List[Dict], key_fn) -> Dict:
    """Compute win rate + avg R for each group."""
    groups = defaultdict(list)
    for t in trades:
        k = key_fn(t)
        if k:
            groups[k].append(t)

    result = {}
    for key, group in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
        w = [t for t in group if _is_win(t)]
        l = [t for t in group if _is_loss(t)]
        r_vals = [_compute_r_multiple(t) for t in group]
        r_vals = [r for r in r_vals if r is not None]
        pnl_list = [_get_pnl(t) for t in group]
        pnl_list = [p for p in pnl_list if p is not None]

        result[key] = {
            "trades": len(group),
            "wins": len(w),
            "losses": len(l),
            "win_rate": round(_safe_div(len(w), len(w) + len(l)) * 100, 1),
            "avg_r": round(_safe_div(sum(r_vals), len(r_vals)), 2),
            "total_pnl": round(sum(pnl_list), 2),
        }
    return result


# ── Tag Analytics ──

def _compute_tag_analytics(trades: List[Dict]) -> Dict:
    """Break down performance by comma-separated tags."""
    tag_groups = defaultdict(list)
    for t in trades:
        tags_str = t.get("tags") or t.get("notes") or ""
        # Extract hashtag-style tags from notes if no explicit tags
        if not t.get("tags") and tags_str:
            # Pull #tag or known keywords
            for word in tags_str.lower().split():
                if word.startswith("#"):
                    tag_groups[word[1:]].append(t)
        else:
            for tag in tags_str.split(","):
                tag = tag.strip().lower()
                if tag:
                    tag_groups[tag].append(t)

    result = {}
    for tag, group in sorted(tag_groups.items(), key=lambda x: len(x[1]), reverse=True):
        w = [t for t in group if _is_win(t)]
        r_vals = [_compute_r_multiple(t) for t in group]
        r_vals = [r for r in r_vals if r is not None]
        result[tag] = {
            "trades": len(group),
            "wins": len(w),
            "win_rate": round(_safe_div(len(w), len(group)) * 100, 1),
            "avg_r": round(_safe_div(sum(r_vals), len(r_vals)), 2),
        }
    return result


# ── Day of Week ──

def _compute_dow_analytics(trades: List[Dict]) -> Dict:
    """Win rate by day of week."""
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    groups = defaultdict(list)
    for t in trades:
        dt = _parse_dt(t.get("closed_at") or t.get("created_at") or "")
        if dt:
            groups[dow_names[dt.weekday()]].append(t)

    result = {}
    for day in dow_names[:5]:  # Mon-Fri only
        group = groups.get(day, [])
        if not group:
            continue
        w = [t for t in group if _is_win(t)]
        r_vals = [_compute_r_multiple(t) for t in group]
        r_vals = [r for r in r_vals if r is not None]
        result[day] = {
            "trades": len(group),
            "wins": len(w),
            "win_rate": round(_safe_div(len(w), len(group)) * 100, 1),
            "avg_r": round(_safe_div(sum(r_vals), len(r_vals)), 2),
        }
    return result


# ── Monthly ──

def _compute_monthly(trades: List[Dict]) -> List[Dict]:
    """Monthly P&L summary."""
    months = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0, "r_sum": 0, "r_count": 0})
    for t in trades:
        dt = _parse_dt(t.get("closed_at") or t.get("created_at") or "")
        if not dt:
            continue
        key = dt.strftime("%Y-%m")
        if _is_win(t):
            months[key]["wins"] += 1
        elif _is_loss(t):
            months[key]["losses"] += 1
        pnl = _get_pnl(t)
        if pnl is not None:
            months[key]["pnl"] += pnl
        r = _compute_r_multiple(t)
        if r is not None:
            months[key]["r_sum"] += r
            months[key]["r_count"] += 1

    result = []
    for month in sorted(months.keys()):
        d = months[month]
        total = d["wins"] + d["losses"]
        result.append({
            "month": month,
            "trades": total,
            "wins": d["wins"],
            "losses": d["losses"],
            "win_rate": round(_safe_div(d["wins"], total) * 100, 1),
            "pnl": round(d["pnl"], 2),
            "avg_r": round(_safe_div(d["r_sum"], d["r_count"]), 2),
        })
    return result


# ── Weekly ──

def _compute_weekly(trades: List[Dict]) -> List[Dict]:
    """Weekly P&L summary (last 12 weeks)."""
    weeks = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0, "r_sum": 0, "r_count": 0})
    for t in trades:
        dt = _parse_dt(t.get("closed_at") or t.get("created_at") or "")
        if not dt:
            continue
        key = dt.strftime("%Y-W%W")
        if _is_win(t):
            weeks[key]["wins"] += 1
        elif _is_loss(t):
            weeks[key]["losses"] += 1
        pnl = _get_pnl(t)
        if pnl is not None:
            weeks[key]["pnl"] += pnl
        r = _compute_r_multiple(t)
        if r is not None:
            weeks[key]["r_sum"] += r
            weeks[key]["r_count"] += 1

    result = []
    for week in sorted(weeks.keys())[-12:]:
        d = weeks[week]
        total = d["wins"] + d["losses"]
        result.append({
            "week": week,
            "trades": total,
            "wins": d["wins"],
            "losses": d["losses"],
            "win_rate": round(_safe_div(d["wins"], total) * 100, 1),
            "pnl": round(d["pnl"], 2),
            "avg_r": round(_safe_div(d["r_sum"], d["r_count"]), 2),
        })
    return result


# ── Equity Curve ──

def _compute_equity_curve(trades: List[Dict]) -> List[Dict]:
    """
    Cumulative R-multiple curve (one point per closed trade, sorted by date).
    """
    dated = []
    for t in trades:
        dt = _parse_dt(t.get("closed_at") or t.get("created_at") or "")
        r = _compute_r_multiple(t)
        pnl = _get_pnl(t)
        if dt and (r is not None or pnl is not None):
            dated.append((dt, r or 0, pnl or 0, t.get("symbol", "?")))

    dated.sort(key=lambda x: x[0])

    curve = []
    cum_r = 0
    cum_pnl = 0
    for dt, r, pnl, sym in dated:
        cum_r += r
        cum_pnl += pnl
        curve.append({
            "date": dt.strftime("%Y-%m-%d"),
            "symbol": sym,
            "r": round(r, 2),
            "cum_r": round(cum_r, 2),
            "pnl": round(pnl, 2),
            "cum_pnl": round(cum_pnl, 2),
        })
    return curve


# ── Max Drawdown ──

def _compute_max_drawdown(equity_curve: List[Dict]) -> float:
    """Max drawdown in R-multiples from the equity curve."""
    if not equity_curve:
        return 0
    peak = 0
    max_dd = 0
    for point in equity_curve:
        cum = point["cum_r"]
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ── Streaks ──

def _compute_streaks(trades: List[Dict]) -> Dict:
    """Track win/loss streaks."""
    # Sort by close date
    dated = []
    for t in trades:
        dt = _parse_dt(t.get("closed_at") or t.get("created_at") or "")
        dated.append((dt or datetime.min, t))
    dated.sort(key=lambda x: x[0])

    current_streak = 0
    current_type = None
    best_win_streak = 0
    worst_loss_streak = 0
    temp_streak = 0
    temp_type = None

    for _, t in dated:
        if _is_win(t):
            outcome = "W"
        elif _is_loss(t):
            outcome = "L"
        else:
            continue

        if outcome == temp_type:
            temp_streak += 1
        else:
            temp_type = outcome
            temp_streak = 1

        if outcome == "W" and temp_streak > best_win_streak:
            best_win_streak = temp_streak
        if outcome == "L" and temp_streak > worst_loss_streak:
            worst_loss_streak = temp_streak

        current_type = temp_type
        current_streak = temp_streak

    return {
        "current": current_streak,
        "current_type": "WIN" if current_type == "W" else "LOSS" if current_type == "L" else "NONE",
        "best_win_streak": best_win_streak,
        "worst_loss_streak": worst_loss_streak,
    }


# ── Holding Time ──

def _compute_holding_times(trades: List[Dict]) -> Dict:
    """Average holding time for wins vs losses."""
    win_hours = []
    loss_hours = []
    for t in trades:
        entered = _parse_dt(t.get("entered_at") or t.get("created_at") or "")
        exited = _parse_dt(t.get("closed_at") or t.get("exited_at") or "")
        if not entered or not exited:
            continue
        hours = (exited - entered).total_seconds() / 3600
        if hours < 0 or hours > 10000:
            continue
        if _is_win(t):
            win_hours.append(hours)
        elif _is_loss(t):
            loss_hours.append(hours)

    return {
        "avg_win_hours": round(_safe_div(sum(win_hours), len(win_hours)), 1),
        "avg_loss_hours": round(_safe_div(sum(loss_hours), len(loss_hours)), 1),
        "fastest_win_hours": round(min(win_hours), 1) if win_hours else 0,
        "slowest_win_hours": round(max(win_hours), 1) if win_hours else 0,
    }


# ── Generate Text Report ──

def generate_analytics_report(analytics: Dict) -> str:
    """Human-readable performance report text."""
    a = analytics
    if a.get("total_trades", 0) == 0:
        return "No trades to analyze."

    lines = []
    lines.append("=" * 65)
    lines.append("TRADING PERFORMANCE REPORT")
    lines.append(f"  Period: {a['period_days']}d | Trades: {a['total_trades']} "
                 f"({a['closed_trades']} closed, {a['open_trades']} open)")
    lines.append("=" * 65)

    # Core
    lines.append(f"\n{'─'*40}")
    lines.append(f"  Win Rate:        {a['win_rate']}%  ({a['wins']}W / {a['losses']}L)")
    lines.append(f"  Avg R:           {a['avg_r']:+.2f}R")
    lines.append(f"  Avg Win:         {a['avg_win_r']:+.2f}R")
    lines.append(f"  Avg Loss:        {a['avg_loss_r']:+.2f}R")
    lines.append(f"  Expectancy:      {a['expectancy_r']:+.3f}R per trade")
    lines.append(f"  Profit Factor:   {a['profit_factor']:.2f}")
    lines.append(f"  Max Drawdown:    {a['max_drawdown_r']:.2f}R")
    lines.append(f"  Total P&L:       ${a['total_pnl']:+,.2f}")

    # Streaks
    s = a.get("streaks", {})
    if s:
        lines.append(f"\n  Current Streak:  {s['current']} {s['current_type']}")
        lines.append(f"  Best Win Streak: {s['best_win_streak']}")
        lines.append(f"  Worst Loss Run:  {s['worst_loss_streak']}")

    # By Direction
    bd = a.get("by_direction", {})
    if bd:
        lines.append(f"\n{'─'*40}")
        lines.append("  BY DIRECTION:")
        for d, stats in bd.items():
            lines.append(f"    {d:6s}: {stats['win_rate']}% WR | {stats['avg_r']:+.2f}R | "
                        f"{stats['trades']} trades | ${stats['total_pnl']:+,.2f}")

    # By Symbol (top 8)
    bs = a.get("by_symbol", {})
    if bs:
        lines.append(f"\n{'─'*40}")
        lines.append("  BY SYMBOL (top 8):")
        for sym, stats in list(bs.items())[:8]:
            lines.append(f"    {sym:6s}: {stats['win_rate']}% WR | {stats['avg_r']:+.2f}R | "
                        f"{stats['trades']} trades | ${stats['total_pnl']:+,.2f}")

    # By Timeframe
    bt = a.get("by_timeframe", {})
    if bt:
        lines.append(f"\n{'─'*40}")
        lines.append("  BY TIMEFRAME:")
        for tf, stats in bt.items():
            lines.append(f"    {tf:8s}: {stats['win_rate']}% WR | {stats['avg_r']:+.2f}R | "
                        f"{stats['trades']} trades")

    # By Signal
    bsig = a.get("by_signal", {})
    if bsig:
        lines.append(f"\n{'─'*40}")
        lines.append("  BY SIGNAL:")
        for sig, stats in bsig.items():
            lines.append(f"    {sig:8s}: {stats['win_rate']}% WR | {stats['avg_r']:+.2f}R | "
                        f"{stats['trades']} trades")

    # Day of Week
    dow = a.get("by_day_of_week", {})
    if dow:
        lines.append(f"\n{'─'*40}")
        lines.append("  BY DAY OF WEEK:")
        for day, stats in dow.items():
            lines.append(f"    {day:10s}: {stats['win_rate']}% WR | {stats['avg_r']:+.2f}R | "
                        f"{stats['trades']} trades")

    # Monthly
    ml = a.get("monthly", [])
    if ml:
        lines.append(f"\n{'─'*40}")
        lines.append("  MONTHLY P&L:")
        for m in ml[-6:]:
            lines.append(f"    {m['month']}: {m['win_rate']}% WR | {m['avg_r']:+.2f}R | "
                        f"${m['pnl']:+,.2f} ({m['trades']} trades)")

    # Holding time
    ht = a.get("holding_time", {})
    if ht and (ht.get("avg_win_hours") or ht.get("avg_loss_hours")):
        lines.append(f"\n{'─'*40}")
        lines.append("  HOLDING TIME:")
        lines.append(f"    Avg Win:   {ht['avg_win_hours']:.1f} hours")
        lines.append(f"    Avg Loss:  {ht['avg_loss_hours']:.1f} hours")

    lines.append("\n" + "=" * 65)
    return "\n".join(lines)
