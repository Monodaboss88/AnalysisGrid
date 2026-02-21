"""
card_renderer.py — HTML renderer for Execution and Thesis trading cards.

Renders structured card data (from card_data_builder) into self-contained
HTML fragments suitable for embedding or returning as full pages.

Capital Ladder replaces the old Execution card — shows entry/stop/target
price levels plus interactive capital deployment, contract allocation,
and real-time outcome matrix.
"""

from html import escape
from datetime import datetime
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
# CAPITAL LADDER CARD  (replaces old Execution Card)
# ---------------------------------------------------------------------------

LADDER_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
:root {
  --bg-deep:#06080d;--bg-card:#0d1117;--bg-surface:#161b22;--bg-elevated:#1c2333;
  --border:#21262d;--border-light:#30363d;
  --text-primary:#f0f6fc;--text-secondary:#8b949e;--text-muted:#484f58;
  --green:#3fb950;--green-bright:#56d364;--green-dim:rgba(63,185,80,0.1);
  --red:#f85149;--red-bright:#ff7b72;--red-dim:rgba(248,81,73,0.1);
  --amber:#d29922;--amber-bright:#e3b341;--amber-dim:rgba(210,153,34,0.1);
  --blue:#58a6ff;--cyan:#39d2c0;--cyan-dim:rgba(57,210,192,0.08);--purple:#bc8cff;
}
*{margin:0;padding:0;box-sizing:border-box}
.ladder-card{width:560px;background:var(--bg-card);border-radius:16px;border:1px solid var(--border);overflow:hidden;box-shadow:0 0 80px rgba(0,0,0,0.5);font-family:'Outfit',sans-serif;color:var(--text-primary)}
.lc-header{padding:20px 24px 16px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
.lc-header-left{display:flex;align-items:center;gap:14px}
.ticker-icon{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:18px;color:#fff;box-shadow:0 2px 16px rgba(24,119,242,0.3)}
.lc-ticker{font-size:24px;font-weight:800;letter-spacing:-0.5px}
.lc-sub{font-size:9px;color:var(--text-muted);letter-spacing:2px;text-transform:uppercase;font-weight:600;margin-top:2px}
.capital-input-wrap{display:flex;align-items:baseline;gap:2px}
.capital-dollar{font-family:'JetBrains Mono';font-size:26px;font-weight:800;color:var(--cyan)}
.capital-input{font-family:'JetBrains Mono';font-size:26px;font-weight:800;color:var(--cyan);background:transparent;border:none;outline:none;width:120px;text-align:right;border-bottom:2px solid var(--border-light);padding-bottom:2px;transition:border-color 0.2s}
.capital-input:focus{border-bottom-color:var(--cyan)}
.capital-label{font-size:9px;color:var(--text-muted);letter-spacing:1.5px;text-transform:uppercase;font-weight:600;text-align:right;margin-top:2px}
.quick-amounts{display:flex;gap:4px;margin-top:6px;justify-content:flex-end}
.quick-btn{font-family:'JetBrains Mono';font-size:9px;font-weight:700;padding:3px 8px;border-radius:4px;background:var(--bg-elevated);color:var(--text-muted);border:1px solid var(--border);cursor:pointer;transition:all 0.15s}
.quick-btn:hover{background:var(--bg-surface);color:var(--cyan);border-color:var(--cyan)}
.quick-btn.active{background:var(--cyan-dim);color:var(--cyan);border-color:rgba(57,210,192,0.3)}
.structure-strip{display:flex;border-bottom:1px solid var(--border)}
.struct-cell{flex:1;padding:10px 6px;text-align:center;border-right:1px solid var(--border)}
.struct-cell:last-child{border-right:none}
.struct-val{font-family:'JetBrains Mono';font-size:12px;font-weight:700;color:var(--text-primary)}
.struct-lbl{font-size:7.5px;color:var(--text-muted);letter-spacing:1.2px;text-transform:uppercase;font-weight:600;margin-top:2px}
.struct-val.green{color:var(--green-bright)}
.struct-input{font-family:'JetBrains Mono';font-size:12px;font-weight:700;color:var(--cyan);background:transparent;border:none;outline:none;width:58px;text-align:center;border-bottom:1px solid var(--border-light);transition:border-color 0.2s}
.struct-input:focus{border-bottom-color:var(--cyan)}
.struct-input.amber{color:var(--amber-bright)}
.struct-input.amber:focus{border-bottom-color:var(--amber-bright)}
.alloc-section{padding:16px 24px;border-bottom:1px solid var(--border)}
.alloc-title{font-size:9px;color:var(--text-muted);letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center}
.alloc-ratio-btns{display:flex;gap:4px}
.ratio-btn{font-family:'JetBrains Mono';font-size:9px;font-weight:700;padding:3px 8px;border-radius:4px;background:var(--bg-elevated);color:var(--text-muted);border:1px solid var(--border);cursor:pointer;transition:all 0.15s}
.ratio-btn:hover{background:var(--bg-surface);color:var(--text-primary);border-color:var(--text-muted)}
.ratio-btn.active{background:rgba(57,210,192,0.08);color:var(--cyan);border-color:rgba(57,210,192,0.3)}
.alloc-row{margin-bottom:10px}
.alloc-bar-label{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}
.alloc-name{font-size:12px;font-weight:600;color:var(--text-primary)}
.alloc-name .strike{font-family:'JetBrains Mono';color:var(--cyan);font-weight:700}
.alloc-name .hedge{font-family:'JetBrains Mono';color:var(--amber);font-weight:700}
.alloc-amount{font-family:'JetBrains Mono';font-size:13px;font-weight:700;color:var(--text-primary)}
.alloc-pct{font-family:'JetBrains Mono';font-size:10px;color:var(--text-secondary);margin-left:4px}
.alloc-track{height:6px;border-radius:3px;background:var(--bg-elevated);overflow:hidden}
.alloc-fill{height:100%;border-radius:3px;transition:width 0.3s ease}
.alloc-fill.call{background:linear-gradient(90deg,#1a6e3a,var(--green))}
.alloc-fill.put{background:linear-gradient(90deg,#8a6d00,var(--amber))}
.alloc-detail{font-size:10px;color:var(--text-muted);margin-top:3px;font-family:'JetBrains Mono';font-weight:500}
.alloc-detail span{color:var(--text-secondary)}
.ladder{padding:6px 0;position:relative}
.level{display:grid;grid-template-columns:36px 1fr 90px 68px;align-items:center;padding:12px 20px;position:relative;transition:background 0.15s}
.level:hover{background:rgba(255,255,255,0.015)}
.level::before{content:'';position:absolute;left:36px;top:0;bottom:0;width:2px;background:var(--border-light)}
.level:first-child::before{top:50%}.level:last-child::before{bottom:50%}
.node{width:14px;height:14px;border-radius:50%;position:relative;z-index:2;display:flex;align-items:center;justify-content:center}
.node::after{content:'';position:absolute;width:6px;height:6px;border-radius:50%;background:var(--bg-card)}
.node.t2{background:var(--green-bright);box-shadow:0 0 12px rgba(86,211,100,0.4)}
.node.t1{background:var(--green);box-shadow:0 0 12px rgba(63,185,80,0.35)}
.node.entry{background:var(--cyan);box-shadow:0 0 14px rgba(57,210,192,0.45);animation:pulse 2.5s infinite}
.node.stop{background:var(--red);box-shadow:0 0 12px rgba(248,81,73,0.4)}
@keyframes pulse{0%,100%{box-shadow:0 0 14px rgba(57,210,192,0.45)}50%{box-shadow:0 0 24px rgba(57,210,192,0.75)}}
.level-info{padding-left:16px}
.level-name{font-size:13px;font-weight:700;color:var(--text-primary)}
.level-desc{font-size:10px;color:var(--text-muted);margin-top:1px;font-weight:500}
.level-price{font-family:'JetBrains Mono';font-size:16px;font-weight:700;text-align:right}
.level-price.green{color:var(--green-bright)}.level-price.cyan{color:var(--cyan)}.level-price.red{color:var(--red-bright)}
.level-dist{font-family:'JetBrains Mono';font-size:11px;font-weight:600;text-align:right}
.level-dist.up{color:var(--green)}.level-dist.dn{color:var(--red)}.level-dist.flat{color:var(--text-muted)}
.level.current{background:var(--cyan-dim);border-left:3px solid var(--cyan);padding-left:17px}
.zone-sep{padding:0 20px;margin:2px 0}
.zone-sep-line{border-top:1px dashed var(--border-light);position:relative}
.zone-sep-label{position:absolute;top:-7px;left:50%;transform:translateX(-50%);background:var(--bg-card);padding:0 10px;font-size:8px;color:var(--text-muted);letter-spacing:1.8px;text-transform:uppercase;font-weight:700}
.capital-ladder-section{border-top:1px solid var(--border);padding:16px 24px}
.cap-title{font-size:9px;color:var(--text-muted);letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:14px}
.cap-level{display:grid;grid-template-columns:72px 1fr 110px;align-items:start;margin-bottom:16px;padding-bottom:16px;border-bottom:1px solid rgba(255,255,255,0.03)}
.cap-level:last-child{border-bottom:none;margin-bottom:0;padding-bottom:0}
.cap-price-col{text-align:center}
.cap-price{font-family:'JetBrains Mono';font-size:13px;font-weight:700}
.cap-price.green{color:var(--green-bright)}.cap-price.cyan{color:var(--cyan)}.cap-price.red{color:var(--red-bright)}
.cap-tag{display:inline-block;font-size:7px;font-weight:800;letter-spacing:1px;padding:2px 6px;border-radius:3px;margin-top:3px}
.cap-tag.entry{background:var(--cyan-dim);color:var(--cyan)}
.cap-tag.t1{background:var(--green-dim);color:var(--green-bright)}
.cap-tag.t2{background:var(--green-dim);color:var(--green-bright)}
.cap-tag.stop{background:var(--red-dim);color:var(--red-bright)}
.cap-detail{padding:0 14px}
.cap-action{font-size:12px;font-weight:700;color:var(--text-primary);margin-bottom:4px}
.cap-action .hl-green{color:var(--green-bright)}.cap-action .hl-red{color:var(--red-bright)}.cap-action .hl-cyan{color:var(--cyan)}
.cap-steps{list-style:none}
.cap-steps li{font-size:10px;color:var(--text-secondary);padding:2px 0;padding-left:12px;position:relative;line-height:1.5}
.cap-steps li::before{content:'›';position:absolute;left:0;color:var(--text-muted);font-weight:700}
.cap-steps .mono{font-family:'JetBrains Mono';font-weight:600;color:var(--text-primary);font-size:10px}
.cap-result{text-align:right;padding-top:2px}
.cap-result-val{font-family:'JetBrains Mono';font-size:15px;font-weight:800}
.cap-result-val.green{color:var(--green-bright)}.cap-result-val.red{color:var(--red-bright)}
.cap-result-pnl{font-family:'JetBrains Mono';font-size:10px;font-weight:600;margin-top:2px}
.cap-result-pnl.green{color:var(--green)}.cap-result-pnl.red{color:var(--red)}
.cap-result-label{font-size:8px;color:var(--text-muted);letter-spacing:0.8px;margin-top:1px}
.waterfall{border-top:1px solid var(--border);padding:16px 24px}
.wf-title{font-size:9px;color:var(--text-muted);letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:12px}
.wf-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0}
.wf-row.total{border-top:1px solid var(--border-light);margin-top:4px;padding-top:10px}
.wf-label{font-size:11px;color:var(--text-secondary);font-weight:500;min-width:140px}
.wf-val{font-family:'JetBrains Mono';font-size:12px;font-weight:700;text-align:right;min-width:140px}
.wf-val.green{color:var(--green-bright)}.wf-val.red{color:var(--red-bright)}.wf-val.white{color:var(--text-primary)}
.wf-bar{flex:1;margin:0 12px;height:4px;border-radius:2px;background:var(--bg-elevated);position:relative;overflow:hidden}
.wf-bar-fill{position:absolute;top:0;bottom:0;left:0;border-radius:2px;transition:width 0.3s ease}
.lc-footer{padding:10px 24px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
.lc-footer-text{font-size:9px;color:var(--text-muted);letter-spacing:0.8px}

/* ── MOBILE RESPONSIVE ── */
@media(max-width:620px){.ladder-card{width:100%;max-width:560px;border-radius:12px}}
@media(max-width:480px){
  .ladder-card{width:100%;border-radius:0;border-left:none;border-right:none}
  .lc-header{flex-direction:column;align-items:flex-start;gap:12px;padding:16px 16px 12px}
  .lc-header .header-right-lc{width:100%}
  .capital-input-wrap{justify-content:flex-start}
  .capital-label{text-align:left}
  .quick-amounts{justify-content:flex-start;flex-wrap:wrap}
  .quick-btn{padding:6px 12px;font-size:10px}
  .structure-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:0}
  .struct-cell{border-bottom:1px solid var(--border);padding:10px 8px}
  .struct-cell:nth-child(3){border-right:none}.struct-cell:nth-child(6){border-right:none}
  .struct-cell:nth-child(n+4){border-bottom:none}
  .struct-input{width:100%}
  .alloc-section{padding:14px 16px}
  .alloc-title{flex-direction:column;align-items:flex-start;gap:8px}
  .alloc-ratio-btns{flex-wrap:wrap}
  .ratio-btn{padding:5px 10px;font-size:10px}
  .alloc-name{font-size:11px}
  .alloc-detail{font-size:9px;word-break:break-word}
  .level{grid-template-columns:28px 1fr auto;grid-template-rows:auto auto;padding:10px 14px;gap:0 8px}
  .node{width:12px;height:12px;grid-row:span 2}.node::after{width:5px;height:5px}
  .level::before{left:28px}
  .level-info{grid-column:2;padding-left:8px}
  .level-name{font-size:12px}.level-desc{font-size:9px}
  .level-price{font-size:14px}
  .level-dist{grid-column:3;grid-row:2;font-size:10px}
  .level.current{padding-left:11px}
  .zone-sep{padding:0 14px}
  .capital-ladder-section{padding:14px 16px}
  .cap-level{grid-template-columns:1fr;grid-template-rows:auto auto auto;gap:8px}
  .cap-price-col{display:flex;align-items:center;gap:10px;text-align:left}
  .cap-detail{padding:0}.cap-action{font-size:12px}
  .cap-steps li{font-size:9px;line-height:1.6}.cap-steps .mono{font-size:9px}
  .cap-result{text-align:left;display:flex;align-items:baseline;gap:8px;padding-top:4px;border-top:1px dashed var(--border-light)}
  .cap-result-val{font-size:14px}.cap-result-pnl{font-size:10px}
  .cap-result-label{font-size:7px;margin-top:0;margin-left:auto}
  .waterfall{padding:14px 16px}
  .wf-row{flex-wrap:wrap;gap:4px}
  .wf-label{min-width:unset;width:100%;font-size:10px}
  .wf-bar{flex:1;min-width:60px;order:1}
  .wf-val{min-width:unset;font-size:11px;order:2}
  .wf-row.total{flex-wrap:nowrap}
  .wf-row.total .wf-val{font-size:11px;white-space:nowrap}
  .lc-footer{padding:10px 16px;flex-wrap:wrap;gap:4px}
  .lc-footer-text{font-size:8px}
}
@media(max-width:360px){
  .lc-ticker{font-size:20px}
  .capital-dollar,.capital-input{font-size:22px}
  .capital-input{width:100px}
  .quick-btn{padding:5px 8px;font-size:9px}
  .struct-cell{padding:8px 4px}.struct-val{font-size:11px}.struct-lbl{font-size:7px}
  .level-price{font-size:13px}.cap-steps li{font-size:8px}
}
"""


def _ticker_gradient(sym: str) -> str:
    """Generate a deterministic gradient color for a ticker icon."""
    h = sum(ord(c) for c in sym) % 360
    return f"linear-gradient(135deg, hsl({h},70%,45%), hsl({(h+40)%360},80%,35%))"


def _parse_t1_exit(t1_weight_str: str) -> float:
    """Parse T1 exit weight like '70-80%' into a decimal like 0.75."""
    import re
    nums = re.findall(r'[\d.]+', str(t1_weight_str))
    if nums:
        avg = sum(float(n) for n in nums) / len(nums)
        return avg / 100 if avg > 1 else avg
    return 0.75  # default


def _estimate_premiums(price, entry, call_strike, put_strike, exp_move_pct, iv_pct):
    """Estimate rough call/put premiums from available data."""
    if price <= 0:
        return 5.0, 1.0
    # Use expected move or IV to estimate
    vol = (exp_move_pct or iv_pct * 0.05 or 3.0) / 100
    # Simple Black-Scholes-ish approximation for ATM-ish options
    call_itm = max(0, price - call_strike) if call_strike > 0 else 0
    call_extrinsic = price * vol * 0.6
    call_prem = max(0.50, call_itm + call_extrinsic)

    put_itm = max(0, put_strike - price) if put_strike > 0 else 0
    put_extrinsic = price * vol * 0.15  # OTM puts cheaper
    put_prem = max(0.10, put_itm + put_extrinsic)

    return round(call_prem, 2), round(put_prem, 2)


def render_capital_ladder(d: dict) -> str:
    """Render the Capital Ladder Card — interactive capital deployment with
    entry/stop/targets, contract allocation, and outcome matrix.
    Replaces the old Execution Card."""

    sym = escape(d.get("symbol", ""))
    price = float(d.get("price", 0) or 0)
    direction = d.get("direction", "LONG")
    is_long = direction.upper() == "LONG"
    prefix = "mtf_long" if is_long else "mtf_short"

    # Key levels
    entry_low = float(d.get(f"{prefix}_entry_low", 0) or 0)
    entry_high = float(d.get(f"{prefix}_entry_high", 0) or 0)
    entry = round((entry_low + entry_high) / 2, 2) if (entry_low > 0 and entry_high > 0) else price
    t1 = float(d.get(f"{prefix}_t1", 0) or 0)
    t2 = float(d.get(f"{prefix}_t2", 0) or 0)
    stop = float(d.get("working_stop", 0) or d.get("hard_stop", 0) or 0)

    # Options data
    call_strike = float(d.get("opt_call_strike", 0) or 0)
    put_strike = float(d.get("opt_put_strike", 0) or 0)
    dte = int(d.get("opt_call_dte", 0) or 0)
    expiry = d.get("opt_call_expiry", "")
    iv_pct = float(d.get("iv_pct", 0) or 0)
    iv_level = d.get("iv_level", "")
    exp_move_pct = float(d.get("expected_move_pct", 0) or 0)

    # T1 exit weight
    t1_exit = _parse_t1_exit(d.get("t1_exit_weight", "75%"))
    t1_exit_int = int(t1_exit * 100)

    # Historical odds
    call_hit_3d = float(d.get("call_hit_3d", 0) or 0)

    # Estimate premiums
    call_prem, put_prem = _estimate_premiums(
        price, entry, call_strike, put_strike, exp_move_pct, iv_pct
    )

    # Position size / R label
    pos_size = d.get("position_size", "0.75R")

    # Distances from entry
    def _dist(target, base):
        if base <= 0 or target <= 0:
            return 0
        return (target - base) / base * 100

    t1_dist = _dist(t1, entry)
    t2_dist = _dist(t2, entry)
    stop_dist = _dist(stop, entry)

    # T1 description
    t1_desc = f"{call_hit_3d:.1f}% hit 3D &middot; Exit {t1_exit_int}% of calls &middot; Stop &rarr; BE" if call_hit_3d > 0 else f"Exit {t1_exit_int}% of calls &middot; Stop &rarr; BE"
    t2_desc = "Only if volume breaks &middot; Exit remaining"

    # Expiry display
    if expiry:
        from datetime import datetime as _dt
        try:
            exp_dt = _dt.strptime(expiry, "%Y-%m-%d")
            exp_display = exp_dt.strftime("%b %d")
        except Exception:
            exp_display = expiry
    elif dte > 0:
        exp_display = f"{dte}d out"
    else:
        exp_display = "—"

    dte_label = f"Expiry &middot; {dte}d" if dte > 0 else "Expiry"
    iv_display = f"{iv_pct:.1f}%" if iv_pct > 0 else "—"
    iv_lbl = f"IV &middot; {iv_level}" if iv_level else "IV"

    # Entry zone description
    ez_desc = f"Zone ${entry_low:,.0f}&ndash;${entry_high:,.0f}" if entry_low > 0 and entry_high > 0 else f"@ ${entry:,.2f}"

    # Gradient for ticker icon
    gradient = _ticker_gradient(sym)
    icon_letter = sym[0] if sym else "?"

    # Direction labels for primary / hedge
    if is_long:
        primary_type = "C"
        hedge_type = "P"
    else:
        primary_type = "P"
        hedge_type = "C"

    # Footer
    ts = d.get("timestamp", "")[:10] or datetime.utcnow().strftime("%b %d, %Y").upper()
    direction_label = "HEDGED LONG" if is_long else "HEDGED SHORT"

    return f"""<div class="card ladder-card" data-card-type="capital-ladder">
<style>{LADDER_CSS}</style>

<!-- HEADER -->
<div class="lc-header">
  <div class="lc-header-left">
    <div class="ticker-icon" style="background:{gradient}">{icon_letter}</div>
    <div>
      <div class="lc-ticker">{sym}</div>
      <div class="lc-sub">Capital Deployment Ladder</div>
    </div>
  </div>
  <div class="header-right-lc">
    <div class="capital-input-wrap">
      <span class="capital-dollar">$</span>
      <input type="text" class="capital-input" id="capitalInput" value="7,500">
    </div>
    <div class="capital-label">Trade Capital &middot; {escape(pos_size)}</div>
    <div class="quick-amounts">
      <button class="quick-btn" data-val="2500">$2.5K</button>
      <button class="quick-btn" data-val="5000">$5K</button>
      <button class="quick-btn active" data-val="7500">$7.5K</button>
      <button class="quick-btn" data-val="10000">$10K</button>
      <button class="quick-btn" data-val="15000">$15K</button>
      <button class="quick-btn" data-val="25000">$25K</button>
    </div>
  </div>
</div>

<!-- STRUCTURE STRIP -->
<div class="structure-strip">
  <div class="struct-cell">
    <div><input type="text" class="struct-input" id="callStrike" value="${call_strike:,.0f}"></div>
    <div class="struct-lbl">{'Call' if is_long else 'Put'} Strike</div>
  </div>
  <div class="struct-cell">
    <div><input type="text" class="struct-input amber" id="putStrike" value="${put_strike:,.0f}"></div>
    <div class="struct-lbl">Hedge Strike</div>
  </div>
  <div class="struct-cell">
    <div><input type="text" class="struct-input" id="callPrice" value="${call_prem:.2f}"></div>
    <div class="struct-lbl">{'Call' if is_long else 'Put'} Premium</div>
  </div>
  <div class="struct-cell">
    <div><input type="text" class="struct-input amber" id="putPrice" value="${put_prem:.2f}"></div>
    <div class="struct-lbl">Hedge Premium</div>
  </div>
  <div class="struct-cell">
    <div class="struct-val">{exp_display}</div>
    <div class="struct-lbl">{dte_label}</div>
  </div>
  <div class="struct-cell">
    <div class="struct-val green">{iv_display}</div>
    <div class="struct-lbl">{iv_lbl}</div>
  </div>
</div>

<!-- CONTRACT ALLOCATION -->
<div class="alloc-section">
  <div class="alloc-title">
    <span>Contract Allocation</span>
    <div class="alloc-ratio-btns">
      <button class="ratio-btn" data-call="100" data-put="0">100/0</button>
      <button class="ratio-btn" data-call="85" data-put="15">85/15</button>
      <button class="ratio-btn active" data-call="75" data-put="25">75/25</button>
      <button class="ratio-btn" data-call="65" data-put="35">65/35</button>
      <button class="ratio-btn" data-call="50" data-put="50">50/50</button>
    </div>
  </div>
  <div class="alloc-row">
    <div class="alloc-bar-label">
      <div class="alloc-name">Primary — <span class="strike" id="callLabel">${call_strike:,.0f}{primary_type} {exp_display}</span></div>
      <div><span class="alloc-amount" id="callAlloc">$5,625</span><span class="alloc-pct" id="callPctEl">75%</span></div>
    </div>
    <div class="alloc-track"><div class="alloc-fill call" id="callBar" style="width:75%"></div></div>
    <div class="alloc-detail" id="callDetail"></div>
  </div>
  <div class="alloc-row">
    <div class="alloc-bar-label">
      <div class="alloc-name">Hedge — <span class="hedge" id="putLabel">${put_strike:,.0f}{hedge_type}</span></div>
      <div><span class="alloc-amount" id="putAlloc">$1,875</span><span class="alloc-pct" id="putPctEl">25%</span></div>
    </div>
    <div class="alloc-track"><div class="alloc-fill put" id="putBar" style="width:25%"></div></div>
    <div class="alloc-detail" id="putDetail"></div>
  </div>
</div>

<!-- PRICE LADDER -->
<div class="ladder">
  <div class="level">
    <div class="node t2"></div>
    <div class="level-info"><div class="level-name">T2 — Conditional</div><div class="level-desc">{t2_desc}</div></div>
    <div class="level-price green">${t2:,.2f}</div>
    <div class="level-dist {'up' if t2_dist>=0 else 'dn'}">{t2_dist:+.2f}%</div>
  </div>
  <div class="level">
    <div class="node t1"></div>
    <div class="level-info"><div class="level-name">T1 — Primary Exit</div><div class="level-desc">{t1_desc}</div></div>
    <div class="level-price green">${t1:,.2f}</div>
    <div class="level-dist {'up' if t1_dist>=0 else 'dn'}">{t1_dist:+.2f}%</div>
  </div>
  <div class="zone-sep"><div class="zone-sep-line"><span class="zone-sep-label">current zone</span></div></div>
  <div class="level current">
    <div class="node entry"></div>
    <div class="level-info"><div class="level-name">Entry — ${entry:,.2f}</div><div class="level-desc" id="entryDesc">{ez_desc} &middot; Deploy full $7,500</div></div>
    <div class="level-price cyan">${entry:,.2f}</div>
    <div class="level-dist flat">YOU ARE HERE</div>
  </div>
  <div class="zone-sep"><div class="zone-sep-line"><span class="zone-sep-label">risk zone</span></div></div>
  <div class="level">
    <div class="node stop"></div>
    <div class="level-info"><div class="level-name">Working Stop</div><div class="level-desc">3&times; confirmed &middot; Exit all</div></div>
    <div class="level-price red">${stop:,.2f}</div>
    <div class="level-dist {'dn' if stop_dist<=0 else 'up'}">{stop_dist:+.2f}%</div>
  </div>
</div>

<!-- CAPITAL AT EACH LEVEL -->
<div class="capital-ladder-section">
  <div class="cap-title">Capital at Each Level</div>
  <div class="cap-level">
    <div class="cap-price-col"><div class="cap-price cyan">${entry:,.2f}</div><div class="cap-tag entry">ENTRY</div></div>
    <div class="cap-detail">
      <div class="cap-action">Deploy <span class="hl-cyan" id="entryDeploy">$7,500</span></div>
      <ul class="cap-steps" id="entrySteps"></ul>
    </div>
    <div class="cap-result">
      <div class="cap-result-val" style="color:var(--text-primary)" id="entryVal">$7,500</div>
      <div class="cap-result-pnl" style="color:var(--text-muted)">&mdash;</div>
      <div class="cap-result-label">DEPLOYED</div>
    </div>
  </div>
  <div class="cap-level">
    <div class="cap-price-col"><div class="cap-price green">${t1:,.2f}</div><div class="cap-tag t1">TARGET 1</div></div>
    <div class="cap-detail">
      <div class="cap-action" id="t1Action"></div>
      <ul class="cap-steps" id="t1Steps"></ul>
    </div>
    <div class="cap-result">
      <div class="cap-result-val green" id="t1Val"></div>
      <div class="cap-result-pnl green" id="t1Pnl"></div>
      <div class="cap-result-label">PORTFOLIO VALUE</div>
    </div>
  </div>
  <div class="cap-level">
    <div class="cap-price-col"><div class="cap-price green">${t2:,.2f}</div><div class="cap-tag t2">TARGET 2</div></div>
    <div class="cap-detail">
      <div class="cap-action" id="t2Action"></div>
      <ul class="cap-steps" id="t2Steps"></ul>
    </div>
    <div class="cap-result">
      <div class="cap-result-val green" id="t2Val"></div>
      <div class="cap-result-pnl green" id="t2Pnl"></div>
      <div class="cap-result-label">IF T2 HITS</div>
    </div>
  </div>
  <div class="cap-level">
    <div class="cap-price-col"><div class="cap-price red">${stop:,.2f}</div><div class="cap-tag stop">STOP</div></div>
    <div class="cap-detail">
      <div class="cap-action" id="stopAction"></div>
      <ul class="cap-steps" id="stopSteps"></ul>
    </div>
    <div class="cap-result">
      <div class="cap-result-val red" id="stopVal"></div>
      <div class="cap-result-pnl red" id="stopPnl"></div>
      <div class="cap-result-label">MAX LOSS W/ HEDGE</div>
    </div>
  </div>
</div>

<!-- WATERFALL -->
<div class="waterfall">
  <div class="wf-title">Outcome Matrix</div>
  <div class="wf-row">
    <div class="wf-label">T1 Only (most likely)</div>
    <div class="wf-bar"><div class="wf-bar-fill" id="wfT1Bar" style="background:var(--green)"></div></div>
    <div class="wf-val green" id="wfT1Val"></div>
  </div>
  <div class="wf-row">
    <div class="wf-label">T1 + T2 (full ride)</div>
    <div class="wf-bar"><div class="wf-bar-fill" id="wfT2Bar" style="background:var(--green-bright)"></div></div>
    <div class="wf-val green" id="wfT2Val"></div>
  </div>
  <div class="wf-row">
    <div class="wf-label">Stopped Out (hedged)</div>
    <div class="wf-bar"><div class="wf-bar-fill" id="wfStopHBar" style="background:var(--red)"></div></div>
    <div class="wf-val red" id="wfStopHVal"></div>
  </div>
  <div class="wf-row">
    <div class="wf-label">Stopped Out (no hedge)</div>
    <div class="wf-bar"><div class="wf-bar-fill" id="wfStopNBar" style="background:var(--red-bright)"></div></div>
    <div class="wf-val red" id="wfStopNVal"></div>
  </div>
  <div class="wf-row total">
    <div class="wf-label" style="font-weight:700;color:var(--text-primary)">Risk / Reward at T1</div>
    <div class="wf-val white" id="wfRR" style="font-size:13px;margin-left:auto"></div>
  </div>
</div>

<!-- FOOTER -->
<div class="lc-footer">
  <div class="lc-footer-text">{ts} &middot; {escape(pos_size)} &middot; {direction_label}</div>
  <div class="lc-footer-text" style="color:var(--amber)">Adjust premiums above for actual fills</div>
</div>

<script>
(function(){{
  const ENTRY={entry},T1={t1},T2={t2},STOP={stop};
  const T1_EXIT={t1_exit};
  let capital=7500,callPct=75,putPct=25;
  let callPremium={call_prem},putPremium={put_prem},callStrike={call_strike},putStrike={put_strike};
  const card=document.currentScript.closest('.ladder-card');
  const $=id=>card.querySelector('#'+id);
  const fmt=n=>Math.round(n).toLocaleString('en-US');
  const fmtD=(n,d2=2)=>n.toLocaleString('en-US',{{minimumFractionDigits:d2,maximumFractionDigits:d2}});
  const pn=str=>parseFloat(str.replace(/[$,\\s]/g,''))||0;
  const li=h=>'<li>'+h+'</li>';
  const m=v=>'<span class="mono">'+v+'</span>';
  function callVal(price,strike,prem){{
    const delta=Math.min(0.95,Math.max(0.15,0.5+(price/strike-1)*3.5));
    const move=(price-ENTRY)*delta;
    const decay=prem*0.03;
    return Math.max(0.10,prem+move-decay);
  }}
  function putVal(price,strike,prem){{
    if(price>=ENTRY){{const decay=prem*0.15;return Math.max(0.05,prem-(price-ENTRY)*0.04-decay);}}
    const delta=Math.min(0.6,Math.max(0.05,0.08+(strike/price-0.85)*2));
    const move=(ENTRY-price)*delta;
    return Math.max(0.05,prem+move-prem*0.08);
  }}
  function calc(){{
    const cBudget=capital*callPct/100;
    const pBudget=capital*putPct/100;
    const cCost=callPremium*100;
    const pCost=putPremium*100;
    const cContracts=cCost>0?Math.floor(cBudget/cCost):0;
    const pContracts=pCost>0&&putPct>0?Math.floor(pBudget/pCost):0;
    const cDeployed=cContracts*cCost;
    const pDeployed=pContracts*pCost;
    const reserve=capital-cDeployed-pDeployed;
    const cT1=callVal(T1,callStrike,callPremium);
    const cT2=callVal(T2,callStrike,callPremium);
    const cStop=callVal(STOP,callStrike,callPremium);
    const pT1=putVal(T1,putStrike,putPremium);
    const pT2=putVal(T2,putStrike,putPremium);
    const pStop=putVal(STOP,putStrike,putPremium);
    const cSoldT1=Math.max(cContracts>0?1:0,Math.round(cContracts*T1_EXIT));
    const cKeepT1=cContracts-cSoldT1;
    const cCashT1=cSoldT1*cT1*100;
    const cHeldT1=cKeepT1*cT1*100;
    const pHeldT1=pContracts*pT1*100;
    const portT1=cCashT1+cHeldT1+pHeldT1+reserve;
    const pnlT1=portT1-capital;
    const cCashT2=cCashT1+cKeepT1*cT2*100;
    const pCashT2=pContracts*pT2*100;
    const portT2=cCashT2+pCashT2+reserve;
    const pnlT2=portT2-capital;
    const cCashStop=cContracts*cStop*100;
    const pCashStop=pContracts*pStop*100;
    const portStop=cCashStop+pCashStop+reserve;
    const pnlStop=portStop-capital;
    const portStopNH=cCashStop+reserve;
    const pnlStopNH=portStopNH-capital;
    const hedgeSaved=pCashStop-pDeployed;
    $('callAlloc').textContent='$'+fmt(cBudget);
    $('callPctEl').textContent=callPct+'%';
    $('callBar').style.width=callPct+'%';
    $('callLabel').textContent='$'+callStrike+'{primary_type} {exp_display}';
    $('callDetail').innerHTML=cContracts>0?'~'+cContracts+' contracts @ <span>$'+fmtD(callPremium)+'</span> ea &rarr; <span>$'+fmt(cDeployed)+' deployed</span> &middot; $'+fmt(cBudget-cDeployed)+' reserve':'<span style="color:var(--red)">Budget too low for 1 contract ($'+fmt(cCost)+' min)</span>';
    $('putAlloc').textContent='$'+fmt(pBudget);
    $('putPctEl').textContent=putPct+'%';
    $('putBar').style.width=putPct+'%';
    $('putLabel').textContent='$'+putStrike+'{hedge_type}';
    $('putDetail').innerHTML=putPct===0?'<span style="color:var(--text-muted)">No hedge — unprotected downside</span>':pContracts>0?'~'+pContracts+' contracts @ <span>$'+fmtD(putPremium)+'</span> ea &rarr; <span>$'+fmt(pDeployed)+' deployed</span> &middot; $'+fmt(pBudget-pDeployed)+' reserve':'<span style="color:var(--red)">Budget too low for 1 contract ($'+fmt(pCost)+' min)</span>';
    $('entryDeploy').textContent='$'+fmt(capital);
    $('entryVal').textContent='$'+fmt(capital);
    $('entryDesc').textContent='{ez_desc} &middot; Deploy full $'+fmt(capital);
    let eSteps=[li('Buy '+cContracts+'&times; '+m('$'+callStrike+'{primary_type} {exp_display}')+' @ ~$'+fmtD(callPremium)+' = '+m('$'+fmt(cDeployed)))];
    if(pContracts>0)eSteps.push(li('Buy '+pContracts+'&times; '+m('$'+putStrike+'{hedge_type}')+' @ ~$'+fmtD(putPremium)+' = '+m('$'+fmt(pDeployed))));
    eSteps.push(li('Cash reserve: '+m('$'+fmt(reserve))+' (commissions)'));
    $('entrySteps').innerHTML=eSteps.join('');
    const cGainT1=callPremium>0?((cT1-callPremium)/callPremium*100):0;
    $('t1Action').innerHTML='Sell <span class="hl-green">'+cSoldT1+' of '+cContracts+'</span> calls &middot; Lock profit';
    let t1S=[li('$'+callStrike+'{primary_type} now ~$'+fmtD(cT1)+' ('+(cGainT1>=0?'+':'')+fmtD(cGainT1,0)+'%) &rarr; Sell '+cSoldT1+' = '+m('$'+fmt(cCashT1)))];
    if(cKeepT1>0)t1S.push(li('Keep '+cKeepT1+' contract'+(cKeepT1>1?'s':'')+' riding &rarr; stop at breakeven'));
    else t1S.push(li('All calls exited at T1'));
    if(pContracts>0)t1S.push(li('Puts decayed to ~'+m('$'+fmtD(pT1))+' &rarr; hold as hedge ('+m('$'+fmt(pHeldT1))+')'));
    $('t1Steps').innerHTML=t1S.join('');
    $('t1Val').textContent='$'+fmt(portT1);
    $('t1Val').className='cap-result-val '+(pnlT1>=0?'green':'red');
    $('t1Pnl').textContent=(pnlT1>=0?'+':'')+' $'+fmt(Math.abs(pnlT1))+' ('+(pnlT1>=0?'+':'')+fmtD(pnlT1/capital*100,0)+'%)';
    $('t1Pnl').className='cap-result-pnl '+(pnlT1>=0?'green':'red');
    const cGainT2=callPremium>0?((cT2-callPremium)/callPremium*100):0;
    $('t2Action').innerHTML='Sell <span class="hl-green">remaining</span> &middot; Close hedge';
    let t2S=[];
    if(cKeepT1>0)t2S.push(li('$'+callStrike+'{primary_type} now ~$'+fmtD(cT2)+' (+'+fmtD(cGainT2,0)+'%) &rarr; Sell '+cKeepT1+' remaining = '+m('$'+fmt(cKeepT1*cT2*100))));
    if(pContracts>0)t2S.push(li('Close hedge &rarr; '+m('$'+fmt(pCashT2))+' (salvage)'));
    t2S.push(li('Trade complete — fully exited'));
    $('t2Steps').innerHTML=t2S.join('');
    $('t2Val').textContent='$'+fmt(portT2);
    $('t2Pnl').textContent='+$'+fmt(pnlT2)+' (+'+fmtD(pnlT2/capital*100,0)+'%)';
    const cLossPct=callPremium>0?((cStop-callPremium)/callPremium*100):0;
    $('stopAction').innerHTML='Exit <span class="hl-red">all calls</span> &middot; '+(pContracts>0?'Evaluate hedge':'Full loss');
    let sS=[li('$'+callStrike+'{primary_type} drops to ~$'+fmtD(cStop)+' ('+fmtD(cLossPct,0)+'%) &rarr; Sell '+cContracts+' = '+m('$'+fmt(cCashStop))),li('Call loss: '+m('<span style="color:var(--red)">-$'+fmt(cDeployed-cCashStop)+'</span>'))];
    if(pContracts>0){{sS.push(li('Hedge rises to ~$'+fmtD(pStop)+' &rarr; now worth '+m('$'+fmt(pCashStop))));if(hedgeSaved>0)sS.push(li('Hedge offsets '+m('<span style="color:var(--green)">+$'+fmt(hedgeSaved)+'</span>')+' of call loss'));}}
    $('stopSteps').innerHTML=sS.join('');
    $('stopVal').textContent='$'+fmt(portStop);
    $('stopPnl').textContent='-$'+fmt(Math.abs(pnlStop))+' ('+fmtD(pnlStop/capital*100,0)+'%)';
    const scale=Math.max(Math.abs(pnlT2),Math.abs(pnlStopNH),1);
    $('wfT1Bar').style.width=Math.abs(pnlT1)/scale*80+'%';
    $('wfT1Val').textContent='+$'+fmt(pnlT1)+' (+'+fmtD(pnlT1/capital*100,0)+'%)';
    $('wfT2Bar').style.width=Math.abs(pnlT2)/scale*80+'%';
    $('wfT2Val').textContent='+$'+fmt(pnlT2)+' (+'+fmtD(pnlT2/capital*100,0)+'%)';
    $('wfStopHBar').style.width=Math.abs(pnlStop)/scale*80+'%';
    $('wfStopHVal').textContent='-$'+fmt(Math.abs(pnlStop))+' ('+fmtD(pnlStop/capital*100,0)+'%)';
    $('wfStopNBar').style.width=Math.abs(pnlStopNH)/scale*80+'%';
    $('wfStopNVal').textContent='-$'+fmt(Math.abs(pnlStopNH))+' ('+fmtD(pnlStopNH/capital*100,0)+'%)';
    const rr=Math.abs(pnlStop)>0?(pnlT1/Math.abs(pnlStop)).toFixed(1):'—';
    $('wfRR').textContent='$'+fmt(Math.abs(pnlStop))+' risk → $'+fmt(pnlT1)+' reward · '+rr+' : 1';
  }}
  $('capitalInput').addEventListener('input',function(){{capital=pn(this.value);if(capital>0)calc();}});
  $('capitalInput').addEventListener('focus',function(){{this.select();}});
  $('capitalInput').addEventListener('blur',function(){{if(capital>0)this.value=fmt(capital);}});
  card.querySelectorAll('.quick-btn').forEach(function(b){{
    b.addEventListener('click',function(){{
      card.querySelectorAll('.quick-btn').forEach(function(x){{x.classList.remove('active');}});
      this.classList.add('active');
      capital=parseInt(this.dataset.val);
      $('capitalInput').value=fmt(capital);
      calc();
    }});
  }});
  card.querySelectorAll('.ratio-btn').forEach(function(b){{
    b.addEventListener('click',function(){{
      card.querySelectorAll('.ratio-btn').forEach(function(x){{x.classList.remove('active');}});
      this.classList.add('active');
      callPct=parseInt(this.dataset.call);
      putPct=parseInt(this.dataset.put);
      calc();
    }});
  }});
  ['callStrike','putStrike','callPrice','putPrice'].forEach(function(id){{
    $(id).addEventListener('input',function(){{
      var v=pn(this.value);
      if(id==='callStrike')callStrike=v;
      if(id==='putStrike')putStrike=v;
      if(id==='callPrice')callPremium=v;
      if(id==='putPrice')putPremium=v;
      if(v>0)calc();
    }});
    $(id).addEventListener('focus',function(){{this.select();}});
  }});
  calc();
}})();
</script>
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
