import { useState, useMemo } from "react";

// ═══════════════════════════════════════════════════════════════
// DEMO DATA — In production, this comes from regime_scanner.py API
// Each symbol's data is the output of RegimeScanner.scan()
// ═══════════════════════════════════════════════════════════════

const DEMO_SYMBOLS = {
  IWM: { price: 265.29, atr: 3.52, atrPct: 1.33, levels: { dipMin: 0.21, dipMax: 0.35, validation: 0.49, halfExit: 0.74, fullTarget: 1.27, fastDrop: 0.39, putFlip: 0.56 } },
  AAPL: { price: 245.55, atr: 4.12, atrPct: 1.68, levels: { dipMin: 0.25, dipMax: 0.41, validation: 0.58, halfExit: 0.87, fullTarget: 1.48, fastDrop: 0.45, putFlip: 0.66 } },
  NVDA: { price: 138.85, atr: 5.89, atrPct: 4.24, levels: { dipMin: 0.35, dipMax: 0.59, validation: 0.82, halfExit: 1.24, fullTarget: 2.12, fastDrop: 0.65, putFlip: 0.94 } },
  TSLA: { price: 355.60, atr: 14.22, atrPct: 4.00, levels: { dipMin: 0.85, dipMax: 1.42, validation: 1.99, halfExit: 2.99, fullTarget: 5.12, fastDrop: 1.56, putFlip: 2.28 } },
  META: { price: 695.40, atr: 15.80, atrPct: 2.27, levels: { dipMin: 0.95, dipMax: 1.58, validation: 2.21, halfExit: 3.32, fullTarget: 5.69, fastDrop: 1.74, putFlip: 2.53 } },
  MSFT: { price: 411.20, atr: 7.35, atrPct: 1.79, levels: { dipMin: 0.44, dipMax: 0.74, validation: 1.03, halfExit: 1.54, fullTarget: 2.65, fastDrop: 0.81, putFlip: 1.18 } },
  AMD: { price: 118.30, atr: 5.45, atrPct: 4.61, levels: { dipMin: 0.33, dipMax: 0.55, validation: 0.76, halfExit: 1.14, fullTarget: 1.96, fastDrop: 0.60, putFlip: 0.87 } },
  AMZN: { price: 228.15, atr: 5.92, atrPct: 2.59, levels: { dipMin: 0.36, dipMax: 0.59, validation: 0.83, halfExit: 1.24, fullTarget: 2.13, fastDrop: 0.65, putFlip: 0.95 } },
  GOOG: { price: 185.30, atr: 4.18, atrPct: 2.26, levels: { dipMin: 0.25, dipMax: 0.42, validation: 0.59, halfExit: 0.88, fullTarget: 1.50, fastDrop: 0.46, putFlip: 0.67 } },
  SPY: { price: 605.10, atr: 7.15, atrPct: 1.18, levels: { dipMin: 0.43, dipMax: 0.72, validation: 1.00, halfExit: 1.50, fullTarget: 2.57, fastDrop: 0.79, putFlip: 1.14 } },
};

// Generate synthetic day data per symbol for demo
function generateDays(sym, info, count) {
  const days = [];
  const base = new Date("2026-02-20");
  let d = 0, tradingDay = 0;
  while (tradingDay < count) {
    const dt = new Date(base - d * 86400000);
    d++;
    if (dt.getDay() === 0 || dt.getDay() === 6) continue;
    tradingDay++;

    const seed = (sym.charCodeAt(0) * 31 + tradingDay * 17 + d * 7) % 100;
    const vol = info.atr * (0.6 + (seed % 40) / 40);
    const open = info.price + (seed - 50) * 0.1;
    const direction = seed > 45 ? 1 : -1;
    const high = open + vol * (0.3 + (seed % 30) / 60);
    const low = open - vol * (0.3 + ((seed + 13) % 30) / 60);
    const close = open + direction * vol * (0.1 + (seed % 20) / 80);
    const xs = Math.max(0, Math.round(seed / 4.5 - 2 + ((seed * 3) % 7)));
    const vwapXs = Math.round(xs * 2.2 + (seed % 8));
    const pctAbove = 20 + (seed % 60);

    let regime, strategy;
    if (xs <= 2) {
      regime = close > open ? "BULL" : "BEAR";
      strategy = close > open ? "MODE_A_LONG" : "MODE_A_SHORT";
    } else if ((open - low) >= info.levels.dipMin) {
      regime = "CHOPPY";
      strategy = "DIP-BUY";
    } else {
      regime = close > open ? "BULL" : "CHOPPY";
      strategy = "NO_DIP";
    }

    let outcome = "SKIP", pnl = 0;
    if (strategy === "DIP-BUY") {
      const hOff = high - open;
      if (hOff >= info.levels.fullTarget) { outcome = "WIN_FULL"; pnl = info.levels.fullTarget / open * 100; }
      else if (hOff >= info.levels.halfExit) { outcome = "WIN_HALF"; pnl = info.levels.halfExit / open * 100 * 0.75; }
      else if (hOff < info.levels.validation) {
        if ((open - low) >= info.levels.fastDrop) { outcome = "LOSS_FAST_DROP"; pnl = -info.levels.fastDrop / open * 100; }
        else { outcome = "LOSS_VALIDATION"; pnl = -info.levels.validation / open * 100 * 0.3; }
      } else { outcome = "LOSS_STOPPED"; pnl = -info.levels.dipMin / open * 100 * 0.5; }
    } else if (strategy.startsWith("MODE_A")) {
      const move = strategy === "MODE_A_LONG" ? (close - open) / open * 100 : (open - close) / open * 100;
      if (move > info.levels.fullTarget / open * 100) { outcome = "WIN_FULL"; pnl = move * 0.8; }
      else if (move > 0) { outcome = "WIN_HALF"; pnl = move * 0.5; }
      else { outcome = "LOSS_STOPPED"; pnl = move * 0.5; }
    }

    days.push({
      date: dt.toISOString().slice(0, 10), symbol: sym,
      open: +open.toFixed(2), high: +high.toFixed(2), low: +low.toFixed(2), close: +close.toFixed(2),
      crosses: xs, vwapXs, pctAbove: +pctAbove.toFixed(1),
      regime, strategy, outcome, pnl: +pnl.toFixed(3),
      highOff: +((high - open) / open * 100).toFixed(2),
      lowOff: +((open - low) / open * 100).toFixed(2),
      closeOff: +((close - open) / open * 100).toFixed(2),
      range: +((high - low) / open * 100).toFixed(2),
    });
  }
  return days;
}

const ALL_DATA = {};
Object.entries(DEMO_SYMBOLS).forEach(([sym, info]) => {
  ALL_DATA[sym] = generateDays(sym, info, 60);
});

const COLORS = {
  BULL: { bg: "#064e3b", text: "#34d399", badge: "#059669", glow: "rgba(52,211,153,0.12)" },
  BEAR: { bg: "#7f1d1d", text: "#f87171", badge: "#dc2626", glow: "rgba(248,113,113,0.12)" },
  CHOPPY: { bg: "#78350f", text: "#fbbf24", badge: "#d97706", glow: "rgba(251,191,36,0.12)" },
};

function StatCard({ label, value, sub, color, small }) {
  return (
    <div style={{
      background: "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)",
      border: `1px solid ${color}33`, borderRadius: 10, padding: small ? "10px 14px" : "14px 18px",
      flex: 1, minWidth: small ? 110 : 130
    }}>
      <div style={{ fontSize: 10, color: "#94a3b8", textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: small ? 22 : 28, fontWeight: 800, color, fontFamily: "monospace", lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "#64748b", marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

function Badge({ regime, size = "md" }) {
  const c = COLORS[regime] || COLORS.CHOPPY;
  const s = size === "sm" ? { fontSize: 9, padding: "1px 7px" } : { fontSize: 11, padding: "3px 10px" };
  return <span style={{ ...s, background: c.bg, color: c.text, borderRadius: 5, fontWeight: 700, display: "inline-block" }}>{regime}</span>;
}

function Bar({ value, max, color, label }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
        <span style={{ fontSize: 11, color: "#94a3b8" }}>{label}</span>
        <span style={{ fontSize: 11, color, fontWeight: 700, fontFamily: "monospace" }}>{value} ({pct.toFixed(0)}%)</span>
      </div>
      <div style={{ height: 7, background: "#1e293b", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: `linear-gradient(90deg, ${color}88, ${color})`, borderRadius: 4, transition: "width 0.4s" }} />
      </div>
    </div>
  );
}

export default function GenericRegimeTracker() {
  const symbols = Object.keys(DEMO_SYMBOLS);
  const [selectedSymbol, setSelectedSymbol] = useState("IWM");
  const [lookback, setLookback] = useState(30);
  const [view, setView] = useState("single"); // single | compare

  const info = DEMO_SYMBOLS[selectedSymbol];
  const allDays = ALL_DATA[selectedSymbol] || [];
  const filtered = useMemo(() => allDays.slice(0, lookback), [allDays, lookback]);
  const total = filtered.length;

  const counts = useMemo(() => {
    const c = { BULL: 0, BEAR: 0, CHOPPY: 0 };
    filtered.forEach(d => c[d.regime]++);
    return c;
  }, [filtered]);

  const strategies = useMemo(() => {
    const s = { "DIP-BUY": 0, MODE_A_LONG: 0, MODE_A_SHORT: 0, NO_DIP: 0, SKIP: 0 };
    filtered.forEach(d => { if (s[d.strategy] !== undefined) s[d.strategy]++; });
    return s;
  }, [filtered]);

  const perf = useMemo(() => {
    const traded = filtered.filter(d => d.outcome !== "SKIP");
    const wins = traded.filter(d => d.outcome.startsWith("WIN"));
    const losses = traded.filter(d => d.outcome.startsWith("LOSS"));
    const totalPnl = traded.reduce((s, d) => s + d.pnl, 0);
    const winPnl = wins.reduce((s, d) => s + d.pnl, 0);
    const lossPnl = Math.abs(losses.reduce((s, d) => s + d.pnl, 0));
    return {
      traded: traded.length, wins: wins.length, losses: losses.length,
      winRate: traded.length ? (wins.length / traded.length * 100).toFixed(1) : "0",
      totalPnl: totalPnl.toFixed(2), pf: lossPnl > 0 ? (winPnl / lossPnl).toFixed(1) : "∞",
    };
  }, [filtered]);

  const windows = useMemo(() => [5, 10, 20].map(w => {
    const sl = filtered.slice(0, Math.min(w, filtered.length));
    return { w, bull: sl.filter(d => d.regime === "BULL").length, bear: sl.filter(d => d.regime === "BEAR").length, choppy: sl.filter(d => d.regime === "CHOPPY").length, total: sl.length };
  }), [filtered]);

  const streak = useMemo(() => {
    if (!filtered.length) return { type: "CHOPPY", count: 0 };
    let t = filtered[0].regime, c = 1;
    for (let i = 1; i < filtered.length; i++) { if (filtered[i].regime === t) c++; else break; }
    return { type: t, count: c };
  }, [filtered]);

  // Comparison data for multi-symbol view
  const compData = useMemo(() => symbols.map(sym => {
    const days = (ALL_DATA[sym] || []).slice(0, lookback);
    const n = days.length;
    const bull = days.filter(d => d.regime === "BULL").length;
    const bear = days.filter(d => d.regime === "BEAR").length;
    const chop = days.filter(d => d.regime === "CHOPPY").length;
    const traded = days.filter(d => d.outcome !== "SKIP");
    const wins = traded.filter(d => d.outcome.startsWith("WIN"));
    const pnl = traded.reduce((s, d) => s + d.pnl, 0);
    return {
      sym, price: DEMO_SYMBOLS[sym].price, atr: DEMO_SYMBOLS[sym].atr,
      atrPct: DEMO_SYMBOLS[sym].atrPct, n, bull, bear, chop,
      chopPct: n ? (chop / n * 100).toFixed(0) : 0,
      dipBuy: days.filter(d => d.strategy === "DIP-BUY").length,
      modeA: days.filter(d => d.strategy.startsWith("MODE_A")).length,
      wr: traded.length ? (wins.length / traded.length * 100).toFixed(1) : "0",
      pnl: pnl.toFixed(2),
      score: n ? (chop / n * 50 + (traded.length ? wins.length / traded.length * 50 : 0)).toFixed(0) : 0,
    };
  }).sort((a, b) => b.score - a.score), [lookback]);

  const lvl = info?.levels || {};

  return (
    <div style={{ background: "linear-gradient(180deg, #0a0f1e 0%, #0d1117 100%)", color: "#e2e8f0", fontFamily: "'Inter', -apple-system, sans-serif", minHeight: "100vh", padding: 20 }}>

      {/* Header + Controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16, flexWrap: "wrap", gap: 10 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, background: "linear-gradient(135deg, #60a5fa, #a78bfa)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            REGIME SCANNER
          </h1>
          <div style={{ fontSize: 12, color: "#64748b", marginTop: 2 }}>Cross-Gate Strategy • Any Symbol • ATR-Scaled Levels</div>
        </div>

        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
          <div style={{ display: "flex", gap: 3 }}>
            {["single", "compare"].map(v => (
              <button key={v} onClick={() => setView(v)} style={{
                padding: "6px 14px", borderRadius: 7, border: "none", cursor: "pointer", fontSize: 12, fontWeight: 700,
                background: view === v ? "linear-gradient(135deg, #6366f1, #8b5cf6)" : "#1e293b",
                color: view === v ? "#fff" : "#64748b",
              }}>{v === "single" ? "Single" : "Compare"}</button>
            ))}
          </div>
          <div style={{ display: "flex", gap: 3 }}>
            {[10, 20, 30, 60].map(d => (
              <button key={d} onClick={() => setLookback(d)} style={{
                padding: "6px 12px", borderRadius: 7, border: "none", cursor: "pointer", fontSize: 12, fontWeight: 700, fontFamily: "monospace",
                background: lookback === d ? "linear-gradient(135deg, #3b82f6, #6366f1)" : "#1e293b",
                color: lookback === d ? "#fff" : "#64748b",
              }}>{d}D</button>
            ))}
          </div>
        </div>
      </div>

      {/* Symbol Selector */}
      <div style={{ display: "flex", gap: 4, marginBottom: 16, flexWrap: "wrap" }}>
        {symbols.map(sym => (
          <button key={sym} onClick={() => { setSelectedSymbol(sym); setView("single"); }} style={{
            padding: "7px 14px", borderRadius: 8, border: view === "single" && selectedSymbol === sym ? "2px solid #60a5fa" : "1px solid #334155",
            cursor: "pointer", fontSize: 12, fontWeight: 700, fontFamily: "monospace",
            background: view === "single" && selectedSymbol === sym ? "#1e3a5f" : "#0f172a",
            color: view === "single" && selectedSymbol === sym ? "#60a5fa" : "#94a3b8",
          }}>{sym}</button>
        ))}
      </div>

      {/* ═══════ COMPARE VIEW ═══════ */}
      {view === "compare" && (
        <div style={{ background: "#0f172a", borderRadius: 12, padding: 16, border: "1px solid #1e293b", overflowX: "auto" }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", marginBottom: 12, textTransform: "uppercase", letterSpacing: 1 }}>
            Watchlist Comparison — {lookback}D • Ranked by Dip-Buy Score
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
            <thead>
              <tr style={{ borderBottom: "2px solid #334155" }}>
                {["#", "Symbol", "Price", "ATR", "ATR%", "Bull", "Bear", "Chop", "Chop%", "Dip-Buy", "ModeA", "WR%", "P&L%", "Score"].map(h => (
                  <th key={h} style={{ padding: "7px 5px", color: "#64748b", fontWeight: 600, textAlign: h === "Symbol" ? "left" : "center", fontSize: 10, textTransform: "uppercase", letterSpacing: 0.5 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {compData.map((r, i) => (
                <tr key={r.sym} onClick={() => { setSelectedSymbol(r.sym); setView("single"); }}
                  style={{ borderBottom: "1px solid #1e293b", cursor: "pointer", background: i % 2 ? "#0a0f1e" : "transparent" }}
                  onMouseOver={e => e.currentTarget.style.background = "#1e293b22"}
                  onMouseOut={e => e.currentTarget.style.background = i % 2 ? "#0a0f1e" : "transparent"}>
                  <td style={{ padding: "9px 5px", textAlign: "center", color: i < 3 ? "#fbbf24" : "#64748b", fontWeight: 800, fontSize: 12 }}>{i + 1}</td>
                  <td style={{ fontWeight: 800, color: "#e2e8f0", fontFamily: "monospace" }}>{r.sym}</td>
                  <td style={{ textAlign: "center", fontFamily: "monospace", color: "#94a3b8" }}>${r.price}</td>
                  <td style={{ textAlign: "center", fontFamily: "monospace", color: "#94a3b8" }}>${r.atr}</td>
                  <td style={{ textAlign: "center", fontFamily: "monospace", color: r.atrPct > 3 ? "#f87171" : r.atrPct > 2 ? "#fbbf24" : "#34d399" }}>{r.atrPct}%</td>
                  <td style={{ textAlign: "center", color: "#34d399", fontFamily: "monospace", fontWeight: 700 }}>{r.bull}</td>
                  <td style={{ textAlign: "center", color: "#f87171", fontFamily: "monospace", fontWeight: 700 }}>{r.bear}</td>
                  <td style={{ textAlign: "center", color: "#fbbf24", fontFamily: "monospace", fontWeight: 700 }}>{r.chop}</td>
                  <td style={{ textAlign: "center", fontFamily: "monospace", color: r.chopPct >= 50 ? "#fbbf24" : "#94a3b8" }}>{r.chopPct}%</td>
                  <td style={{ textAlign: "center", fontFamily: "monospace", color: "#60a5fa", fontWeight: 700 }}>{r.dipBuy}</td>
                  <td style={{ textAlign: "center", fontFamily: "monospace", color: "#a78bfa" }}>{r.modeA}</td>
                  <td style={{ textAlign: "center", fontFamily: "monospace", color: parseFloat(r.wr) > 65 ? "#34d399" : parseFloat(r.wr) > 50 ? "#fbbf24" : "#f87171", fontWeight: 700 }}>{r.wr}%</td>
                  <td style={{ textAlign: "center", fontFamily: "monospace", color: parseFloat(r.pnl) > 0 ? "#34d399" : "#f87171", fontWeight: 700 }}>{r.pnl > 0 ? "+" : ""}{r.pnl}%</td>
                  <td style={{ textAlign: "center" }}>
                    <span style={{ background: i < 3 ? "linear-gradient(135deg, #d97706, #fbbf24)" : "#334155", color: i < 3 ? "#000" : "#94a3b8", padding: "2px 8px", borderRadius: 5, fontWeight: 800, fontSize: 11 }}>{r.score}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ═══════ SINGLE SYMBOL VIEW ═══════ */}
      {view === "single" && (<>

        {/* Top Stats */}
        <div style={{ display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
          <StatCard label={selectedSymbol} value={`$${info.price}`} sub={`ATR $${info.atr} (${info.atrPct}%)`} color="#60a5fa" />
          <StatCard label="Win Rate" value={`${perf.winRate}%`} sub={`${perf.wins}W / ${perf.losses}L`} color={parseFloat(perf.winRate) > 65 ? "#34d399" : "#fbbf24"} />
          <StatCard label="P&L" value={`${perf.totalPnl > 0 ? "+" : ""}${perf.totalPnl}%`} sub={`PF: ${perf.pf}x`} color={parseFloat(perf.totalPnl) > 0 ? "#34d399" : "#f87171"} />
          <StatCard label="Streak" value={streak.count} sub={streak.type} color={COLORS[streak.type]?.text || "#94a3b8"} />
          <StatCard label="Active Days" value={`${perf.traded}/${total}`} sub={`${strategies["DIP-BUY"]} dip / ${strategies.MODE_A_LONG + strategies.MODE_A_SHORT} dir`} color="#a78bfa" />
        </div>

        {/* Regime + Levels Row */}
        <div style={{ display: "flex", gap: 14, marginBottom: 14, flexWrap: "wrap" }}>

          {/* Regime Distribution */}
          <div style={{ flex: 2, minWidth: 300, background: "#0f172a", borderRadius: 12, padding: 16, border: "1px solid #1e293b" }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#94a3b8", marginBottom: 12, textTransform: "uppercase", letterSpacing: 1 }}>Regime — {lookback}D</div>
            <Bar value={counts.BULL} max={total} color="#34d399" label="🟢 BULL" />
            <Bar value={counts.BEAR} max={total} color="#f87171" label="🔴 BEAR" />
            <Bar value={counts.CHOPPY} max={total} color="#fbbf24" label="🟡 CHOPPY" />
            <div style={{ marginTop: 10, height: 26, borderRadius: 6, overflow: "hidden", display: "flex" }}>
              {[["BULL", "#059669", "#34d399"], ["CHOPPY", "#d97706", "#fbbf24"], ["BEAR", "#dc2626", "#f87171"]].map(([r, c1, c2]) => {
                const pct = total > 0 ? counts[r] / total * 100 : 0;
                return pct > 0 ? <div key={r} style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${c1}, ${c2})`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 800, color: r === "CHOPPY" ? "#000" : "#fff" }}>{pct > 8 ? `${pct.toFixed(0)}%` : ""}</div> : null;
              })}
            </div>
            {/* Rolling */}
            <div style={{ marginTop: 14 }}>
              <div style={{ fontSize: 11, color: "#64748b", marginBottom: 6 }}>Rolling Windows</div>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead><tr>{["", "Bull", "Chop", "Bear"].map(h => <th key={h} style={{ fontSize: 10, color: h === "Bull" ? "#34d399" : h === "Bear" ? "#f87171" : h === "Chop" ? "#fbbf24" : "#475569", textAlign: h ? "center" : "left", paddingBottom: 4 }}>{h}</th>)}</tr></thead>
                <tbody>{windows.map(w => <tr key={w.w}><td style={{ fontSize: 12, fontWeight: 700, color: "#e2e8f0", fontFamily: "monospace" }}>{w.w}D</td><td style={{ textAlign: "center", color: "#34d399", fontWeight: 800, fontFamily: "monospace" }}>{w.bull}</td><td style={{ textAlign: "center", color: "#fbbf24", fontWeight: 800, fontFamily: "monospace" }}>{w.choppy}</td><td style={{ textAlign: "center", color: "#f87171", fontWeight: 800, fontFamily: "monospace" }}>{w.bear}</td></tr>)}</tbody>
              </table>
            </div>
          </div>

          {/* Strategy Levels */}
          <div style={{ flex: 1, minWidth: 240, background: "#0f172a", borderRadius: 12, padding: 16, border: "1px solid #6366f144" }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#a78bfa", marginBottom: 12, textTransform: "uppercase", letterSpacing: 1 }}>⚡ ATR-Scaled Levels</div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 10 }}>Auto-calculated from {selectedSymbol} ATR ${info.atr}</div>
            {[
              ["Dip Entry Zone", `-$${lvl.dipMin} to -$${lvl.dipMax}`, "#60a5fa"],
              ["$0.50 Validation", `+$${lvl.validation}`, "#fbbf24"],
              ["Half Exit", `+$${lvl.halfExit}`, "#34d399"],
              ["Full Target (33%)", `+$${lvl.fullTarget}`, "#22c55e"],
              ["Fast Drop Exit", `-$${lvl.fastDrop}`, "#fb923c"],
              ["Put Flip", `-$${lvl.putFlip}`, "#f87171"],
              ["1% VWAP Ext", `+$${(info.price * 0.01).toFixed(2)}`, "#a78bfa"],
            ].map(([label, val, color]) => (
              <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid #1e293b" }}>
                <span style={{ fontSize: 11, color: "#94a3b8" }}>{label}</span>
                <span style={{ fontSize: 12, fontWeight: 700, color, fontFamily: "monospace" }}>{val}</span>
              </div>
            ))}
            <div style={{ marginTop: 10, padding: 8, background: "#1e1b4b", borderRadius: 8, fontSize: 10, color: "#a78bfa" }}>
              IWM equivalents: Dip $0.20-$0.35 → Validation $0.50 → Half $0.75 → Full $1.25
            </div>
          </div>
        </div>

        {/* Day-by-Day Table */}
        <div style={{ background: "#0f172a", borderRadius: 12, padding: 16, border: "1px solid #1e293b", overflowX: "auto" }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "#94a3b8", marginBottom: 12, textTransform: "uppercase", letterSpacing: 1 }}>
            {selectedSymbol} Day-by-Day — {lookback}D
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
            <thead>
              <tr style={{ borderBottom: "2px solid #334155" }}>
                {["Date", "Open", "High", "Low", "Close", "Xs", "Regime", "Strategy", "Outcome", "P&L%"].map(h => (
                  <th key={h} style={{ padding: "6px 4px", color: "#64748b", fontWeight: 600, textAlign: h === "Date" || h === "Strategy" || h === "Outcome" ? "left" : "center", fontSize: 10, textTransform: "uppercase" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((d, i) => {
                const isGreen = d.close > d.open;
                const isWin = d.outcome.startsWith("WIN");
                return (
                  <tr key={d.date} style={{ borderBottom: "1px solid #1e293b", background: i % 2 === 0 ? "transparent" : "#0a0f1e" }}>
                    <td style={{ padding: "8px 4px", fontWeight: 700, color: "#e2e8f0", fontFamily: "monospace" }}>{d.date.slice(5)}</td>
                    <td style={{ textAlign: "center", color: "#94a3b8", fontFamily: "monospace" }}>${d.open.toFixed(2)}</td>
                    <td style={{ textAlign: "center", color: "#34d399", fontFamily: "monospace" }}>${d.high.toFixed(2)}</td>
                    <td style={{ textAlign: "center", color: "#f87171", fontFamily: "monospace" }}>${d.low.toFixed(2)}</td>
                    <td style={{ textAlign: "center", color: isGreen ? "#34d399" : "#f87171", fontWeight: 700, fontFamily: "monospace" }}>${d.close.toFixed(2)}</td>
                    <td style={{ textAlign: "center", fontFamily: "monospace", color: d.crosses <= 2 ? "#a78bfa" : d.crosses >= 20 ? "#fbbf24" : "#94a3b8", fontWeight: d.crosses <= 2 ? 800 : 400 }}>{d.crosses}</td>
                    <td style={{ textAlign: "center" }}><Badge regime={d.regime} size="sm" /></td>
                    <td style={{ fontSize: 10, color: d.strategy === "DIP-BUY" ? "#60a5fa" : d.strategy.includes("LONG") ? "#34d399" : d.strategy.includes("SHORT") ? "#f87171" : "#64748b", fontWeight: 600 }}>{d.strategy}</td>
                    <td style={{ fontSize: 10, color: isWin ? "#34d399" : d.outcome === "SKIP" ? "#475569" : "#f87171", fontWeight: 600 }}>{d.outcome}</td>
                    <td style={{ textAlign: "center", fontFamily: "monospace", fontWeight: 700, color: d.pnl > 0 ? "#34d399" : d.pnl < 0 ? "#f87171" : "#475569" }}>
                      {d.pnl !== 0 ? `${d.pnl > 0 ? "+" : ""}${d.pnl.toFixed(2)}%` : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </>)}

      <div style={{ textAlign: "center", marginTop: 16, fontSize: 10, color: "#334155" }}>
        SEF Trading Systems • Generic Regime Scanner V1 • ATR-Scaled Cross-Gate Strategy
      </div>
    </div>
  );
}
