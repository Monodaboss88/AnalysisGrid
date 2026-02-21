"""
card_renderer.py — HTML renderer for Execution and Thesis trading cards.

Renders structured card data (from card_data_builder) into self-contained
HTML fragments suitable for embedding or returning as full pages.
"""

from html import escape
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v, fmt_type="price"):
    """Format a value for display."""
    if v is None:
        return "—"
    if fmt_type == "price":
        v = float(v)
        return f"${v:,.2f}" if v > 0 else "—"
    if fmt_type == "pct":
        v = float(v)
        return f"{v:.1f}%" if v != 0 else "—"
    if fmt_type == "int":
        return str(int(v)) if v else "—"
    return str(v) if v else "—"


def _dir_class(direction: str) -> str:
    """CSS class for direction."""
    return "long" if direction.upper() == "LONG" else "short"


def _dir_icon(direction: str) -> str:
    return "▲" if direction.upper() == "LONG" else "▼"


def _grade_color(grade: str) -> str:
    g = (grade or "").upper().rstrip("+-")
    return {"A": "#00e676", "B": "#66bb6a", "C": "#ffc107", "D": "#ff9800", "F": "#f44336"}.get(g, "#888")


def _conviction_bar(conv: int, max_val: int = 10) -> str:
    pct = min(100, max(0, conv * 100 // max_val))
    color = "#00e676" if pct >= 70 else "#ffc107" if pct >= 40 else "#f44336"
    return f"""<div class="conv-bar"><div class="conv-fill" style="width:{pct}%;background:{color}"></div><span>{conv}/{max_val}</span></div>"""


def _odds_chip(label: str, val: float, is_pct: bool = True) -> str:
    if val == 0:
        return ""
    color = "#00e676" if val >= 60 else "#ffc107" if val >= 40 else "#f44336"
    display = f"{val:.1f}%" if is_pct else str(val)
    return f'<span class="odds-chip" style="border-color:{color}"><b>{label}</b> {display}</span>'


def _scanner_dot(name: str, vote: str) -> str:
    """Render a scanner evidence dot."""
    color = "#00e676" if vote == "LONG" else "#f44336" if vote == "SHORT" else "#555"
    icon = "▲" if vote == "LONG" else "▼" if vote == "SHORT" else "●"
    return f'<span class="scanner-dot" style="color:{color}" title="{name}: {vote}">{icon} {name}</span>'


# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------

SHARED_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

.card {
    width: 460px;
    border-radius: 16px;
    font-family: 'Inter', -apple-system, sans-serif;
    color: #e0e0e0;
    overflow: hidden;
    position: relative;
}

.card-header {
    padding: 16px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.card-header .symbol {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.card-header .price {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 600;
}
.card-header .ts {
    font-size: 10px;
    opacity: 0.5;
}

.dir-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.5px;
}
.dir-badge.long { background: rgba(0,230,118,0.15); color: #00e676; }
.dir-badge.short { background: rgba(244,67,54,0.15); color: #f44336; }

.card-body {
    padding: 0 20px 20px;
}

.section {
    margin-bottom: 14px;
}
.section-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
    opacity: 0.6;
}

.kv-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0;
    font-size: 12px;
}
.kv-row .label { opacity: 0.6; }
.kv-row .value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}

.level-bar {
    position: relative;
    width: 100%;
    height: 180px;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    margin: 8px 0;
    overflow: visible;
}
.level-mark {
    position: absolute;
    left: 0; right: 0;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 12px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
}
.level-line {
    flex: 1;
    height: 1px;
    opacity: 0.4;
}
.level-label { opacity: 0.5; font-size: 9px; min-width: 36px; }
.level-price { font-weight: 600; min-width: 70px; text-align: right; }

.price-marker {
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #fff;
    box-shadow: 0 0 8px rgba(255,255,255,0.5);
    z-index: 5;
}

.odds-strip {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin: 6px 0;
}
.odds-chip {
    padding: 3px 8px;
    border: 1px solid;
    border-radius: 6px;
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
}
.odds-chip b { margin-right: 3px; }

.conv-bar {
    position: relative;
    height: 16px;
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    overflow: hidden;
    margin: 4px 0;
}
.conv-bar .conv-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}
.conv-bar span {
    position: absolute;
    right: 6px;
    top: 1px;
    font-size: 10px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.consensus-row {
    display: flex;
    gap: 12px;
    margin: 6px 0;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
}
.consensus-row .long-ct { color: #00e676; }
.consensus-row .short-ct { color: #f44336; }
.consensus-row .neut-ct { color: #888; }

.scanner-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px 12px;
}
.scanner-dot {
    font-size: 11px;
    font-weight: 500;
    white-space: nowrap;
}

.options-box {
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 10px 12px;
    margin: 6px 0;
}
.options-box .opt-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
    opacity: 0.5;
}
.opt-row {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    padding: 2px 0;
}
.opt-row .label { opacity: 0.6; }
.opt-row .value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}

.kill-zone {
    background: rgba(244,67,54,0.08);
    border: 1px solid rgba(244,67,54,0.2);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 6px 0;
    font-size: 11px;
}
.kill-zone .kz-title {
    font-size: 10px;
    font-weight: 700;
    color: #f44336;
    margin-bottom: 4px;
}

.thesis-evidence {
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 10px 12px;
    margin: 6px 0;
}
.thesis-evidence .ev-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
    opacity: 0.5;
}

.conflict-box {
    background: rgba(255,193,7,0.08);
    border: 1px solid rgba(255,193,7,0.2);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 6px 0;
    font-size: 11px;
}
.conflict-box .cb-title {
    font-size: 10px;
    font-weight: 700;
    color: #ffc107;
    margin-bottom: 4px;
}

.range-vis {
    position: relative;
    height: 32px;
    background: rgba(255,255,255,0.05);
    border-radius: 6px;
    margin: 8px 0;
    overflow: hidden;
}
.range-fill {
    position: absolute;
    top: 0; bottom: 0;
    background: rgba(0,230,118,0.15);
    border-radius: 6px;
}
.range-marker {
    position: absolute;
    top: 2px; bottom: 2px;
    width: 3px;
    border-radius: 2px;
    z-index: 2;
}

.insider-row {
    display: flex;
    gap: 16px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    margin: 4px 0;
}
.insider-buy { color: #00e676; }
.insider-sell { color: #f44336; }

.footer {
    padding: 10px 20px;
    font-size: 9px;
    opacity: 0.3;
    text-align: center;
    border-top: 1px solid rgba(255,255,255,0.05);
}
"""


# ---------------------------------------------------------------------------
# EXECUTION CARD (green-accented)
# ---------------------------------------------------------------------------

def render_execution_card(d: dict) -> str:
    """Render the Execution Card — entry zones, price ladder, risk parameters."""

    direction = d.get("direction", "LONG")
    is_long = direction.upper() == "LONG"
    prefix = "mtf_long" if is_long else "mtf_short"

    # Gather key prices for the level bar
    entry_low = d.get(f"{prefix}_entry_low", 0)
    entry_high = d.get(f"{prefix}_entry_high", 0)
    stop = d.get("working_stop", 0) or d.get("hard_stop", 0)
    t1 = d.get(f"{prefix}_t1", 0)
    t2 = d.get(f"{prefix}_t2", 0)
    price = d.get("price", 0)
    vah = d.get("vah", 0)
    poc = d.get("poc", 0)
    val = d.get("val", 0)

    # Collect all valid prices for scale
    all_prices = [p for p in [vah, poc, val, price, entry_low, entry_high, stop, t1, t2] if p > 0]
    if not all_prices:
        all_prices = [100]
    p_max = max(all_prices) * 1.005
    p_min = min(all_prices) * 0.995
    p_range = max(p_max - p_min, 0.01)

    def _y_pct(price_val):
        """Convert price to percentage from TOP (high=0%, low=100%)."""
        if price_val <= 0:
            return -10  # hide off screen
        return max(0, min(100, (p_max - price_val) / p_range * 100))

    # Build level marks
    levels = []
    if t2 > 0:
        levels.append(("T2", t2, "#7c4dff"))
    if t1 > 0:
        levels.append(("T1", t1, "#448aff"))
    if vah > 0:
        levels.append(("VAH", vah, "#ff9800"))
    if poc > 0:
        levels.append(("POC", poc, "#ffc107"))
    if entry_high > 0:
        levels.append(("ENTRY↑", entry_high, "#00e676" if is_long else "#f44336"))
    if entry_low > 0:
        levels.append(("ENTRY↓", entry_low, "#00e676" if is_long else "#f44336"))
    if val > 0:
        levels.append(("VAL", val, "#ff9800"))
    if stop > 0:
        levels.append(("STOP", stop, "#f44336"))

    level_html = ""
    for label, px, color in levels:
        y = _y_pct(px)
        level_html += f"""<div class="level-mark" style="top:{y}%">
            <span class="level-label">{label}</span>
            <div class="level-line" style="background:{color}"></div>
            <span class="level-price" style="color:{color}">{_fmt(px)}</span>
        </div>\n"""

    # Price marker
    price_y = _y_pct(price)
    level_html += f'<div class="price-marker" style="top:calc({price_y}% - 5px)" title="Current: {_fmt(price)}"></div>'

    # Entry zone highlight
    entry_zone_html = ""
    if entry_low > 0 and entry_high > 0:
        ez_top = _y_pct(entry_high)
        ez_bot = _y_pct(entry_low)
        ez_h = ez_bot - ez_top
        ez_color = "rgba(0,230,118,0.08)" if is_long else "rgba(244,67,54,0.08)"
        entry_zone_html = f'<div style="position:absolute;left:0;right:0;top:{ez_top}%;height:{ez_h}%;background:{ez_color};border-radius:4px"></div>'

    # Odds strip
    odds_html = ""
    odds_html += _odds_chip("Call 1D", d.get("call_hit_1d", 0))
    odds_html += _odds_chip("Call 3D", d.get("call_hit_3d", 0))
    odds_html += _odds_chip("Put 1D", d.get("put_hit_1d", 0))
    odds_html += _odds_chip("Straddle", d.get("straddle_rate", 0))
    odds_html += _odds_chip("VWAP Rev", d.get("vwap_revert_rate", 0))

    # Options strategy
    opt_html = ""
    if d.get("opt_call_strike", 0) > 0 or d.get("opt_put_strike", 0) > 0:
        opt_html = f"""<div class="options-box">
            <div class="opt-title">Options Idea</div>
            {"<div class='opt-row'><span class='label'>Call Strike</span><span class='value'>" + _fmt(d.get('opt_call_strike')) + "</span></div>" if d.get('opt_call_strike',0)>0 else ""}
            {"<div class='opt-row'><span class='label'>Put Strike</span><span class='value'>" + _fmt(d.get('opt_put_strike')) + "</span></div>" if d.get('opt_put_strike',0)>0 else ""}
            {"<div class='opt-row'><span class='label'>DTE</span><span class='value'>" + _fmt(d.get('opt_call_dte',0),'int') + "d</span></div>" if d.get('opt_call_dte',0)>0 else ""}
        </div>"""

    # Kill zone / invalidation
    invalid = d.get(f"{prefix}_invalid", "")
    kill_html = ""
    if invalid:
        kill_html = f"""<div class="kill-zone">
            <div class="kz-title">⚠ INVALIDATION</div>
            {escape(invalid)}
        </div>"""

    # Trigger
    trigger = d.get(f"{prefix}_trigger", "")
    trigger_html = ""
    if trigger:
        trigger_html = f"""<div class="section">
            <div class="section-title">Entry Trigger</div>
            <div style="font-size:12px;opacity:0.8">{escape(trigger)}</div>
        </div>"""

    grade = d.get(f"{prefix}_grade", "")
    conv = d.get(f"{prefix}_conviction", 0)
    rr = d.get(f"{prefix}_rr", "")
    ev = d.get(f"{prefix}_ev", "")

    return f"""<div class="card" style="background:linear-gradient(135deg,#0d1117 0%,#0f1a12 100%);border:1px solid rgba(0,230,118,0.12)">
    <style>{SHARED_CSS}</style>

    <!-- HEADER -->
    <div class="card-header">
        <div>
            <div class="symbol">{escape(d.get('symbol',''))}</div>
            <div class="price">{_fmt(d.get('price'))}</div>
            <div class="ts">{d.get('timestamp','')[:19]}</div>
        </div>
        <div style="text-align:right">
            <div class="dir-badge {_dir_class(direction)}">{_dir_icon(direction)} {direction}</div>
            <div style="margin-top:6px;font-size:11px;opacity:0.5">EXECUTION CARD</div>
        </div>
    </div>

    <div class="card-body">

        <!-- GRADE + CONVICTION -->
        <div class="section" style="display:flex;gap:16px;align-items:center">
            <div style="text-align:center">
                <div style="font-size:32px;font-weight:800;color:{_grade_color(grade)}">{grade or '—'}</div>
                <div style="font-size:9px;opacity:0.5">GRADE</div>
            </div>
            <div style="flex:1">
                <div style="font-size:9px;opacity:0.5;margin-bottom:2px">CONVICTION</div>
                {_conviction_bar(conv)}
                <div style="display:flex;gap:12px;margin-top:4px;font-size:10px;font-family:'JetBrains Mono',monospace">
                    <span style="opacity:0.6">R:R <b style="color:#448aff">{rr or '—'}</b></span>
                    <span style="opacity:0.6">EV <b style="color:#7c4dff">{ev or '—'}</b></span>
                    <span style="opacity:0.6">Size <b style="color:#ffc107">{d.get('position_size','—')}</b></span>
                </div>
            </div>
        </div>

        <!-- PRICE LADDER -->
        <div class="section">
            <div class="section-title">Price Ladder</div>
            <div class="level-bar">
                {entry_zone_html}
                {level_html}
            </div>
        </div>

        <!-- KEY PARAMS -->
        <div class="section">
            <div class="section-title">Risk Parameters</div>
            <div class="kv-row"><span class="label">Working Stop</span><span class="value" style="color:#f44336">{_fmt(d.get('working_stop'))}</span></div>
            <div class="kv-row"><span class="label">Hard Stop</span><span class="value" style="color:#f44336">{_fmt(d.get('hard_stop'))}</span></div>
            <div class="kv-row"><span class="label">T1 Exit Weight</span><span class="value">{d.get('t1_exit_weight','—')}</span></div>
            <div class="kv-row"><span class="label">Hold Period</span><span class="value">{d.get('hold_period','—')}</span></div>
            <div class="kv-row"><span class="label">ATR</span><span class="value">{_fmt(d.get('atr'),'pct')}</span></div>
            <div class="kv-row"><span class="label">rVOL</span><span class="value">{d.get('rvol',1.0):.2f}x</span></div>
        </div>

        {trigger_html}

        <!-- HISTORICAL ODDS -->
        <div class="section">
            <div class="section-title">Historical Odds</div>
            <div class="odds-strip">{odds_html}</div>
            <div style="display:flex;gap:12px;font-size:10px;font-family:'JetBrains Mono',monospace;margin-top:4px">
                <span style="opacity:0.6">E[↑1D] <b style="color:#00e676">{d.get('expected_up_1d',0):+.2f}%</b></span>
                <span style="opacity:0.6">E[↓1D] <b style="color:#f44336">{d.get('expected_dn_1d',0):+.2f}%</b></span>
                <span style="opacity:0.6">Cond: <b>{escape(d.get('condition','—'))}</b></span>
            </div>
        </div>

        {opt_html}
        {kill_html}

    </div>

    <div class="footer">AnalysisGrid Execution Card &bull; Not financial advice</div>
</div>"""


# ---------------------------------------------------------------------------
# THESIS CARD (purple-accented)
# ---------------------------------------------------------------------------

def render_thesis_card(d: dict) -> str:
    """Render the Thesis Card — scanner evidence, conflicts, key levels, fundamentals."""

    direction = d.get("direction", "LONG")
    is_long = direction.upper() == "LONG"
    prefix = "mtf_long" if is_long else "mtf_short"

    # ── Scanner votes ──
    def _vote(name: str, condition) -> str:
        """Determine a scanner's directional vote."""
        if condition == "LONG" or condition == "BULLISH" or condition is True:
            return "LONG"
        elif condition == "SHORT" or condition == "BEARISH" or condition is False:
            return "SHORT"
        return "NEUTRAL"

    # Build scanner evidence
    scanners = [
        ("Simple", _vote("Simple", "LONG" if "LONG" in d.get("simple_signal","").upper() else ("SHORT" if "SHORT" in d.get("simple_signal","").upper() else "NEUTRAL"))),
        ("MTF AI", _vote("MTF", d.get("mtf_preferred",""))),
        ("Opt Flow", _vote("Flow", d.get("flow_sentiment",""))),
        ("Order Flow", _vote("OF", d.get("flow_bias",""))),
        ("War Room", _vote("WR", "SHORT" if d.get("fade_conviction",0) >= 60 else ("LONG" if d.get("fade_conviction",0) <= 20 else "NEUTRAL"))),
        ("Buffett", _vote("Buff", "LONG" if d.get("buffett_signal","") in ("VALUE","BLOOD") else ("SHORT" if d.get("buffett_signal","")=="OVERVALUED" else "NEUTRAL"))),
        ("Sustain", _vote("Sust", "LONG" if d.get("cycle_phase","") in ("EARLY","MID") else ("SHORT" if d.get("cycle_phase","") in ("LATE","EXTENDED") else "NEUTRAL"))),
    ]

    scanner_html = ""
    for name, vote in scanners:
        scanner_html += _scanner_dot(name, vote)

    # ── Conflicts ──
    conflicts = []
    # Direction vs options flow
    if d.get("flow_sentiment",""):
        flow_dir = "LONG" if d.get("flow_sentiment","") == "BULLISH" else "SHORT"
        if flow_dir != direction:
            conflicts.append(f"Options flow ({d.get('flow_sentiment','')}) disagrees with {direction}")
    # Extension warning
    if d.get("ext_snap_prob", 0) >= 70:
        conflicts.append(f"High snap-back probability: {d.get('ext_snap_prob',0):.0f}% ({d.get('ext_direction','')})")
    # War room fade vs direction
    if d.get("fade_conviction", 0) >= 60 and is_long:
        conflicts.append(f"War Room fade conviction {d.get('fade_conviction',0)}% suggests reversal")
    elif d.get("fade_conviction", 0) <= 20 and not is_long:
        conflicts.append(f"War Room low fade {d.get('fade_conviction',0)}% supports continuation")
    # RSI extreme
    if d.get("rsi", 50) > 75 and is_long:
        conflicts.append(f"RSI overbought at {d.get('rsi',50):.1f}")
    elif d.get("rsi", 50) < 25 and not is_long:
        conflicts.append(f"RSI oversold at {d.get('rsi',50):.1f}")

    conflict_html = ""
    if conflicts:
        conflict_items = "".join(f"<div style='padding:2px 0'>⚡ {escape(c)}</div>" for c in conflicts)
        conflict_html = f"""<div class="conflict-box">
            <div class="cb-title">⚠ CONFLICTS ({len(conflicts)})</div>
            {conflict_items}
        </div>"""

    # ── Range structure ──
    price = d.get("price", 0)
    vah = d.get("vah", 0)
    val = d.get("val", 0)
    poc = d.get("poc", 0)
    range_total = max(vah - val, 0.01) if vah > 0 and val > 0 else 1

    range_html = ""
    if vah > 0 and val > 0:
        # Value area fill
        va_left = 10
        va_width = 80
        # Price position within range
        price_pct = max(0, min(100, (price - val) / range_total * 100))
        poc_pct = max(0, min(100, (poc - val) / range_total * 100)) if poc > 0 else 50

        range_html = f"""<div class="section">
            <div class="section-title">Range Structure</div>
            <div style="display:flex;justify-content:space-between;font-size:10px;opacity:0.5;margin-bottom:2px">
                <span>VAL {_fmt(val)}</span>
                <span>VAH {_fmt(vah)}</span>
            </div>
            <div class="range-vis">
                <div class="range-fill" style="left:{va_left}%;width:{va_width}%"></div>
                <div class="range-marker" style="left:{va_left + poc_pct * va_width / 100}%;background:#ffc107" title="POC {_fmt(poc)}"></div>
                <div class="range-marker" style="left:{va_left + price_pct * va_width / 100}%;background:#fff" title="Price {_fmt(price)}"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:10px;margin-top:2px">
                <span style="opacity:0.5">VWAP {_fmt(d.get('vwap'))}</span>
                <span style="opacity:0.5">RSI {d.get('rsi',50):.1f}</span>
                <span style="opacity:0.5">Pos: {escape(d.get('position',''))}</span>
            </div>
        </div>"""

    # ── Key levels ──
    key_levels_html = f"""<div class="section">
        <div class="section-title">Key Levels</div>
        <div class="kv-row"><span class="label">POC</span><span class="value" style="color:#ffc107">{_fmt(poc)}</span></div>
        <div class="kv-row"><span class="label">VWAP</span><span class="value">{_fmt(d.get('vwap'))}</span></div>
        <div class="kv-row"><span class="label">Fib .618</span><span class="value" style="color:#7c4dff">{_fmt(d.get('fib_618'))}</span></div>
        <div class="kv-row"><span class="label">Fib .786</span><span class="value" style="color:#7c4dff">{_fmt(d.get('fib_786'))}</span></div>
        <div class="kv-row"><span class="label">Call Wall</span><span class="value" style="color:#00e676">{_fmt(d.get('call_wall'))}</span></div>
        <div class="kv-row"><span class="label">Put Wall</span><span class="value" style="color:#f44336">{_fmt(d.get('put_wall'))}</span></div>
        <div class="kv-row"><span class="label">Max Pain</span><span class="value">{_fmt(d.get('max_pain'))}</span></div>
        <div class="kv-row"><span class="label">Key Level (AI)</span><span class="value" style="color:#448aff">{_fmt(d.get('mtf_key_level'))}</span></div>
    </div>"""

    # ── MTF timeframe alignment ──
    def _tf_chip(label, signal):
        if not signal:
            return ""
        color = "#00e676" if "LONG" in signal.upper() else "#f44336" if "SHORT" in signal.upper() else "#888"
        return f'<span style="display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-family:\'JetBrains Mono\',monospace;background:rgba(255,255,255,0.05);color:{color};margin:2px">{label}: {signal}</span>'

    mtf_html = f"""<div class="section">
        <div class="section-title">Timeframe Alignment</div>
        <div style="display:flex;flex-wrap:wrap;gap:4px">
            {_tf_chip("30m", d.get("mtf_30min_signal",""))}
            {_tf_chip("1H", d.get("mtf_1hr_signal",""))}
            {_tf_chip("2H", d.get("mtf_2hr_signal",""))}
            {_tf_chip("4H", d.get("mtf_4hr_signal",""))}
        </div>
        <div class="kv-row" style="margin-top:4px"><span class="label">Confluence</span><span class="value">{d.get('mtf_confluence',0):.0f}%</span></div>
        <div class="kv-row"><span class="label">Dominant</span><span class="value">{escape(d.get('mtf_dominant','—'))}</span></div>
    </div>"""

    # ── Order Flow section ──
    of_html = f"""<div class="section">
        <div class="section-title">Order Flow</div>
        <div class="kv-row"><span class="label">Bias</span><span class="value">{escape(d.get('flow_bias','—'))}</span></div>
        <div class="kv-row"><span class="label">Buy Pressure</span><span class="value" style="color:#00e676">{d.get('buy_pressure',0):.1f}%</span></div>
        <div class="kv-row"><span class="label">Sell Pressure</span><span class="value" style="color:#f44336">{d.get('sell_pressure',0):.1f}%</span></div>
        <div class="kv-row"><span class="label">Momentum</span><span class="value">{escape(d.get('flow_momentum','—'))}</span></div>
    </div>"""

    # ── Options Flow section ──
    oflow_html = ""
    if d.get("flow_sentiment",""):
        oflow_html = f"""<div class="section">
            <div class="section-title">Options Flow</div>
            <div class="kv-row"><span class="label">Sentiment</span><span class="value">{escape(d.get('flow_sentiment',''))}</span></div>
            <div class="kv-row"><span class="label">P/C Ratio</span><span class="value">{d.get('pc_ratio',0):.2f}</span></div>
            <div class="kv-row"><span class="label">IV Rank</span><span class="value">{d.get('iv_pct',0):.0f}% ({escape(d.get('iv_level',''))})</span></div>
            <div class="kv-row"><span class="label">Exp Move</span><span class="value">±{d.get('expected_move_pct',0):.1f}% ({_fmt(d.get('expected_move_usd'))})</span></div>
            <div class="kv-row"><span class="label">Flow Score</span><span class="value">{d.get('flow_score',0)}/100</span></div>
            <div class="kv-row"><span class="label">Unusual</span><span class="value">{d.get('unusual_count',0)} contracts</span></div>
        </div>"""

    # ── Fundamentals strip ──
    fund_html = ""
    if d.get("buffett_grade","") or d.get("rs_grade",""):
        fund_html = f"""<div class="section">
            <div class="section-title">Fundamentals</div>
            <div style="display:flex;gap:16px">
                <div style="flex:1">
                    <div style="font-size:9px;opacity:0.5">BUFFETT</div>
                    <div style="font-size:11px">{escape(d.get('buffett_grade','—'))} &bull; Score {d.get('buffett_score',0)}</div>
                    <div style="font-size:10px;opacity:0.6">DD: {d.get('drawdown_pct',0):.1f}% &bull; Rev: {d.get('revenue_growth',0):+.1f}%</div>
                </div>
                <div style="flex:1">
                    <div style="font-size:9px;opacity:0.5">SUSTAINABILITY</div>
                    <div style="font-size:11px">{escape(d.get('rs_grade','—'))} &bull; Score {d.get('rs_score',0)}</div>
                    <div style="font-size:10px;opacity:0.6">Cycle: {escape(d.get('cycle_phase','—'))}</div>
                </div>
            </div>
        </div>"""

    # ── Insider activity ──
    insider_html = ""
    if d.get("insider_buys", 0) > 0 or d.get("insider_sells", 0) > 0:
        insider_html = f"""<div class="section">
            <div class="section-title">Insider Activity (6mo)</div>
            <div class="insider-row">
                <span class="insider-buy">+{d.get('insider_buys',0)} Buys</span>
                <span class="insider-sell">-{d.get('insider_sells',0)} Sells</span>
                <span style="opacity:0.6">Signal: {escape(d.get('insider_signal','—'))}</span>
            </div>
        </div>"""

    # ── War Room signals ──
    war_html = ""
    war_signals = d.get("war_signals", [])
    if war_signals or d.get("regime",""):
        sig_items = ""
        for s in war_signals[:5]:
            if isinstance(s, str):
                sig_items += f"<div style='padding:1px 0;font-size:10px;opacity:0.7'>• {escape(s)}</div>"
            elif isinstance(s, dict):
                sig_items += f"<div style='padding:1px 0;font-size:10px;opacity:0.7'>• {escape(s.get('signal',''))}</div>"
        war_html = f"""<div class="section">
            <div class="section-title">War Room</div>
            <div class="kv-row"><span class="label">Regime</span><span class="value">{escape(d.get('regime','—'))}</span></div>
            <div class="kv-row"><span class="label">Fade Conviction</span><span class="value">{d.get('fade_conviction',0)}%</span></div>
            <div class="kv-row"><span class="label">Exhaustion</span><span class="value">{d.get('exhaustion',0):.1f}</span></div>
            {sig_items}
        </div>"""

    # ── AI Why (reasoning) ──
    why = d.get(f"{prefix}_why", "")
    why_html = ""
    if why:
        why_html = f"""<div class="section">
            <div class="section-title">AI Reasoning</div>
            <div style="font-size:11px;opacity:0.8;line-height:1.5">{escape(why)}</div>
        </div>"""

    return f"""<div class="card" style="background:linear-gradient(135deg,#0d1117 0%,#14101f 100%);border:1px solid rgba(124,77,255,0.12)">
    <style>{SHARED_CSS}</style>

    <!-- HEADER -->
    <div class="card-header">
        <div>
            <div class="symbol">{escape(d.get('symbol',''))}</div>
            <div class="price">{_fmt(d.get('price'))}</div>
            <div class="ts">{d.get('timestamp','')[:19]}</div>
        </div>
        <div style="text-align:right">
            <div class="dir-badge {_dir_class(direction)}">{_dir_icon(direction)} {direction}</div>
            <div style="margin-top:6px;font-size:11px;opacity:0.5">THESIS CARD</div>
        </div>
    </div>

    <div class="card-body">

        <!-- SCANNER CONSENSUS -->
        <div class="section">
            <div class="section-title">Scanner Consensus</div>
            <div class="consensus-row">
                <span class="long-ct">▲ {d.get('scanner_long',0)} LONG</span>
                <span class="short-ct">▼ {d.get('scanner_short',0)} SHORT</span>
                <span class="neut-ct">● {d.get('scanner_neutral',0)} NEUTRAL</span>
            </div>
            <div class="scanner-grid" style="margin-top:6px">
                {scanner_html}
            </div>
        </div>

        {conflict_html}

        {range_html}
        {key_levels_html}
        {mtf_html}
        {of_html}
        {oflow_html}
        {war_html}
        {fund_html}
        {insider_html}
        {why_html}

    </div>

    <div class="footer">AnalysisGrid Thesis Card &bull; Not financial advice</div>
</div>"""
