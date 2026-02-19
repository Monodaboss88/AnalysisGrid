# Task: Add Risk:Reward Analysis to Stock Catalyst Scanner

Add a **Risk:Reward Analysis** section to the existing `stock-catalyst-scanner.html` file. This section renders inside each ticker's collapsible section, between the daily returns chart and the outlier events table. It also adds R:R data to the Export Brief clipboard output.

## Overview of What to Add

3 things:
1. **CSS styles** for the R:R UI components
2. **Two JS functions** (`buildRiskReward` and `estimateProb`) that generate the R:R HTML
3. **Export Brief additions** — R:R scores and price levels appended to the clipboard output
4. **Mobile responsive** styles for R:R grid

---

## 1. CSS — Add After `.ev-wrap` Style

Find this line in the `<style>` block:
```css
.ev-wrap { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
```

Add ALL of the following CSS immediately AFTER it (before `.ev-hdr`):

```css
/* Risk:Reward Section */
.rr-section {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 20px;
}

.rr-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
}

.rr-header h4 { font-size: 0.95rem; font-weight: 600; }

.rr-body { padding: 20px; }

/* Score cards row */
.rr-scores {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 20px;
}

.rr-score-card {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px;
  text-align: center;
}

.rr-score-card .sc-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.6rem;
  font-weight: 700;
  margin-bottom: 2px;
}

.rr-score-card .sc-lbl {
  font-size: 0.7rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 500;
}

.rr-score-card .sc-sub {
  font-size: 0.73rem;
  color: var(--text-muted);
  font-family: 'JetBrains Mono', monospace;
  margin-top: 4px;
}

/* Score bar */
.score-bar {
  height: 6px;
  background: var(--border);
  border-radius: 3px;
  margin-top: 8px;
  overflow: hidden;
}

.score-bar .sb-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.5s ease;
}

/* Price levels ladder */
.rr-levels { margin-bottom: 16px; }

.rr-levels h5 {
  font-size: 0.82rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text-muted);
  margin-bottom: 12px;
}

.level-row {
  display: grid;
  grid-template-columns: 32px 1fr 90px 90px 80px 60px;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border-radius: 8px;
  margin-bottom: 4px;
  font-size: 0.84rem;
  transition: background 0.15s;
}

.level-row:hover { background: var(--bg-card-hover); }

.level-row .lr-icon { font-size: 1rem; text-align: center; }
.level-row .lr-label { color: var(--text-secondary); font-weight: 500; }
.level-row .lr-price { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.level-row .lr-dist { font-family: 'JetBrains Mono', monospace; color: var(--text-muted); font-size: 0.8rem; }
.level-row .lr-rr { font-family: 'JetBrains Mono', monospace; font-weight: 700; }
.level-row .lr-prob { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

.level-current { background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.2); }
.level-target { background: rgba(34, 197, 94, 0.04); }
.level-stop { background: rgba(239, 68, 68, 0.04); }

/* Scenario cards */
.rr-scenarios {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px;
}

.scenario-card {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
}

.scenario-card .sc-title {
  font-size: 0.85rem;
  font-weight: 700;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.scenario-card .sc-row {
  display: flex;
  justify-content: space-between;
  padding: 5px 0;
  font-size: 0.82rem;
  border-bottom: 1px solid rgba(30, 41, 59, 0.4);
}

.scenario-card .sc-row:last-child { border-bottom: none; }
.scenario-card .sc-row .sc-k { color: var(--text-muted); }
.scenario-card .sc-row .sc-v { font-family: 'JetBrains Mono', monospace; font-weight: 600; }

.rr-grade {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  font-weight: 700;
}

.grade-a { background: rgba(34,197,94,0.15); color: var(--green); }
.grade-b { background: rgba(59,130,246,0.15); color: var(--accent); }
.grade-c { background: rgba(245,158,11,0.15); color: var(--orange); }
.grade-d { background: rgba(239,68,68,0.15); color: var(--red); }
```

## 2. Mobile Responsive — Update `@media` Block

Find the existing `@media (max-width: 768px)` block and add these lines inside it:

```css
.rr-scores { grid-template-columns: 1fr 1fr; }
.level-row { grid-template-columns: 28px 1fr 80px 70px; font-size: 0.78rem; }
.level-row .lr-rr, .level-row .lr-prob { display: none; }
.rr-scenarios { grid-template-columns: 1fr; }
```

## 3. HTML Insertion — In `renderSection()`

Find this block inside the `renderSection` function:

```javascript
<div class="chart-wrap">
  <h3>Daily Returns (outliers highlighted)</h3>
  <canvas id="ch-${ticker}" height="170"></canvas>
</div>

${evTable}
```

Change it to:

```javascript
<div class="chart-wrap">
  <h3>Daily Returns (outliers highlighted)</h3>
  <canvas id="ch-${ticker}" height="170"></canvas>
</div>

${buildRiskReward(r, th)}

${evTable}
```

## 4. JavaScript Functions — Add Before `drawChart()`

Find `function drawChart(ticker, bars, outs, sd, th) {` and add these two functions BEFORE it:

```javascript
/* ── RISK:REWARD ENGINE ── */
function buildRiskReward(r, th) {
  const { ticker, outs, rs, up, down, avgMv, estYr, bars } = r;
  const price = rs.pC;
  const sd = rs.sd;
  const adr = rs.avgDR;
  const adrPct = rs.avgDRpct;
  const total = outs.length;

  // ── COMPUTE SCORES (0-100) ──

  // Trend Score: where is price relative to range? Higher = closer to high = bullish
  const trendScore = Math.round(((price - rs.pL) / (rs.pH - rs.pL)) * 100);

  // Directional Bias Score: based on up/down ratio of outlier days (50 = neutral)
  const biasScore = total > 0 ? Math.round((up / total) * 100) : 50;

  // Volatility Score: inverse — lower vol = higher score (calmer = better risk profile)
  const volScore = sd*100 < 1 ? 95 : sd*100 < 2 ? 80 : sd*100 < 3 ? 60 : sd*100 < 5 ? 40 : 20;

  // Catalyst Risk Score: higher frequency = more risk
  const freq = parseFloat(estYr) || 0;
  const catalystRisk = freq > 30 ? 90 : freq > 20 ? 70 : freq > 10 ? 50 : freq > 5 ? 30 : 15;

  // Volume Conviction Score: higher avg vol ratio on outliers = more institutional conviction
  const avgVR = total > 0 ? outs.reduce((s,o) => s+o.vr, 0) / total : 1;
  const volConviction = Math.min(Math.round(avgVR * 30), 100);

  // Overall Risk Score (0-100, higher = riskier)
  const riskScore = Math.round((100 - volScore) * 0.3 + catalystRisk * 0.35 + (100 - biasScore) * 0.2 + (100 - trendScore) * 0.15);

  // Overall Reward Score (0-100, higher = more upside potential)
  const rewardScore = Math.round(biasScore * 0.3 + trendScore * 0.2 + volConviction * 0.2 + (avgMv > 0 ? Math.min(avgMv * 100 * 8, 100) : 30) * 0.3);

  // R:R Ratio
  const rrRatio = riskScore > 0 ? (rewardScore / riskScore).toFixed(2) : '—';

  // Grade
  const rrNum = parseFloat(rrRatio) || 0;
  const grade = rrNum >= 1.5 ? 'A' : rrNum >= 1.0 ? 'B' : rrNum >= 0.7 ? 'C' : 'D';
  const gradeClass = `grade-${grade.toLowerCase()}`;

  // Score colors
  function scoreColor(val, inverse) {
    const v = inverse ? 100 - val : val;
    if (v >= 70) return 'var(--green)';
    if (v >= 50) return 'var(--accent)';
    if (v >= 30) return 'var(--orange)';
    return 'var(--red)';
  }

  function barColor(val, inverse) {
    const v = inverse ? 100 - val : val;
    if (v >= 70) return '#22c55e';
    if (v >= 50) return '#3b82f6';
    if (v >= 30) return '#f59e0b';
    return '#ef4444';
  }

  // ── PRICE LEVELS ──
  const levels = [];

  // Upside targets
  levels.push({ type: 'target', icon: '🎯', label: '+3σ Move (Extreme)', price: price * (1 + sd * 3), dist: sd * 3, prob: estimateProb(outs, sd, 3, 'up', total) });
  levels.push({ type: 'target', icon: '🎯', label: '+2σ Move (Major)', price: price * (1 + sd * 2), dist: sd * 2, prob: estimateProb(outs, sd, 2, 'up', total) });
  levels.push({ type: 'target', icon: '📍', label: '+1 ADR Target', price: price + adr, dist: adrPct, prob: 68 });
  levels.push({ type: 'target', icon: '📍', label: 'Period High', price: rs.pH, dist: (rs.pH - price) / price, prob: trendScore > 80 ? 40 : trendScore > 50 ? 25 : 15 });

  // Current
  levels.push({ type: 'current', icon: '⚡', label: 'Current Price', price: price, dist: 0, prob: null });

  // Downside risk
  levels.push({ type: 'stop', icon: '🛑', label: '-1 ADR Stop', price: price - adr, dist: -adrPct, prob: 68 });
  levels.push({ type: 'stop', icon: '🛑', label: 'Period Low', price: rs.pL, dist: (rs.pL - price) / price, prob: trendScore < 20 ? 40 : trendScore < 50 ? 25 : 10 });
  levels.push({ type: 'stop', icon: '⚠️', label: '-2σ Move (Major)', price: price * (1 - sd * 2), dist: -sd * 2, prob: estimateProb(outs, sd, 2, 'down', total) });
  levels.push({ type: 'stop', icon: '⚠️', label: '-3σ Move (Extreme)', price: price * (1 - sd * 3), dist: -sd * 3, prob: estimateProb(outs, sd, 3, 'down', total) });

  // ── SCENARIOS ──
  const stop1ADR = price - adr;
  const stop2ADR = price - (adr * 2);
  const risk1ADR = adr;
  const risk2ADR = adr * 2;

  const scenarios = [
    {
      name: 'Conservative',
      icon: '🛡️',
      entry: price,
      stop: stop1ADR,
      target: price + (adr * 2),
      risk: risk1ADR,
      reward: adr * 2,
      rr: 2.0,
      desc: '1 ADR stop, 2 ADR target'
    },
    {
      name: 'Standard',
      icon: '⚖️',
      entry: price,
      stop: stop1ADR,
      target: price * (1 + sd * 2),
      risk: risk1ADR,
      reward: price * sd * 2,
      rr: (price * sd * 2) / risk1ADR,
      desc: '1 ADR stop, 2σ target'
    },
    {
      name: 'Aggressive',
      icon: '🔥',
      entry: price,
      stop: stop2ADR,
      target: price * (1 + sd * 3),
      risk: risk2ADR,
      reward: price * sd * 3,
      rr: (price * sd * 3) / risk2ADR,
      desc: '2 ADR stop, 3σ target'
    },
    {
      name: 'Mean Reversion',
      icon: '🔄',
      entry: price,
      stop: price * (1 - sd * 1.5),
      target: price + (adr * 1.5),
      risk: price * sd * 1.5,
      reward: adr * 1.5,
      rr: (adr * 1.5) / (price * sd * 1.5),
      desc: '1.5σ stop, 1.5 ADR target'
    }
  ];

  // ── BUILD HTML ──
  let scoresHTML = `
    <div class="rr-scores">
      <div class="rr-score-card">
        <div class="sc-val" style="color:${scoreColor(biasScore, false)}">${biasScore}</div>
        <div class="sc-lbl">Bullish Bias</div>
        <div class="sc-sub">${up}/${total} up</div>
        <div class="score-bar"><div class="sb-fill" style="width:${biasScore}%;background:${barColor(biasScore,false)}"></div></div>
      </div>
      <div class="rr-score-card">
        <div class="sc-val" style="color:${scoreColor(trendScore, false)}">${trendScore}</div>
        <div class="sc-lbl">Trend Position</div>
        <div class="sc-sub">${trendScore > 70 ? 'Near highs' : trendScore > 30 ? 'Mid-range' : 'Near lows'}</div>
        <div class="score-bar"><div class="sb-fill" style="width:${trendScore}%;background:${barColor(trendScore,false)}"></div></div>
      </div>
      <div class="rr-score-card">
        <div class="sc-val" style="color:${scoreColor(volScore, false)}">${volScore}</div>
        <div class="sc-lbl">Vol Score</div>
        <div class="sc-sub">${sd*100 < 2 ? 'Low vol' : sd*100 < 3 ? 'Moderate' : sd*100 < 5 ? 'High' : 'Extreme'}</div>
        <div class="score-bar"><div class="sb-fill" style="width:${volScore}%;background:${barColor(volScore,false)}"></div></div>
      </div>
      <div class="rr-score-card">
        <div class="sc-val" style="color:${scoreColor(catalystRisk, true)}">${catalystRisk}</div>
        <div class="sc-lbl">Catalyst Risk</div>
        <div class="sc-sub">~${estYr}/yr events</div>
        <div class="score-bar"><div class="sb-fill" style="width:${catalystRisk}%;background:${barColor(catalystRisk,true)}"></div></div>
      </div>
      <div class="rr-score-card">
        <div class="sc-val" style="color:${scoreColor(volConviction, false)}">${volConviction}</div>
        <div class="sc-lbl">Vol Conviction</div>
        <div class="sc-sub">${avgVR.toFixed(1)}x avg vol</div>
        <div class="score-bar"><div class="sb-fill" style="width:${volConviction}%;background:${barColor(volConviction,false)}"></div></div>
      </div>
      <div class="rr-score-card" style="border-color:rgba(168,85,247,0.3)">
        <div class="sc-val" style="color:#c084fc">${rrRatio}</div>
        <div class="sc-lbl">R:R Score</div>
        <div class="sc-sub"><span class="rr-grade ${gradeClass}">Grade ${grade}</span></div>
      </div>
    </div>`;

  // Price levels
  let levelsHTML = `<div class="rr-levels"><h5>📐 Price Level Ladder</h5>`;
  levels.forEach(l => {
    const cls = l.type === 'current' ? 'level-current' : l.type === 'target' ? 'level-target' : 'level-stop';
    const distStr = l.dist === 0 ? '—' : (l.dist > 0 ? '+' : '') + (l.dist * 100).toFixed(2) + '%';
    const priceColor = l.type === 'target' ? 'color:var(--green)' : l.type === 'stop' ? 'color:var(--red)' : 'color:var(--accent)';
    const probStr = l.prob !== null ? l.prob + '%' : '—';
    const probColor = l.prob >= 50 ? 'color:var(--green)' : l.prob >= 20 ? 'color:var(--orange)' : 'color:var(--red)';

    // R:R vs 1 ADR stop
    let rrStr = '—';
    if (l.type === 'target' && adr > 0) {
      const reward = l.price - price;
      rrStr = (reward / adr).toFixed(1) + ':1';
    }

    levelsHTML += `
      <div class="level-row ${cls}">
        <span class="lr-icon">${l.icon}</span>
        <span class="lr-label">${l.label}</span>
        <span class="lr-price" style="${priceColor}">$${l.price.toFixed(2)}</span>
        <span class="lr-dist mono">${distStr}</span>
        <span class="lr-rr mono" style="color:var(--accent)">${rrStr}</span>
        <span class="lr-prob mono" style="${l.prob !== null ? probColor : ''}">${probStr}</span>
      </div>`;
  });
  levelsHTML += `
    <div style="display:grid;grid-template-columns:32px 1fr 90px 90px 80px 60px;gap:8px;padding:6px 12px 0;font-size:0.68rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px">
      <span></span><span></span><span>Price</span><span>Distance</span><span>R:R vs ADR</span><span>Hist %</span>
    </div>
  </div>`;

  // Scenarios
  let scenariosHTML = `<div class="rr-scenarios">`;
  scenarios.forEach(s => {
    const rrColor = s.rr >= 3 ? 'var(--green)' : s.rr >= 2 ? 'var(--accent)' : s.rr >= 1 ? 'var(--orange)' : 'var(--red)';
    const rrGrade = s.rr >= 3 ? 'A' : s.rr >= 2 ? 'B' : s.rr >= 1 ? 'C' : 'D';
    scenariosHTML += `
      <div class="scenario-card">
        <div class="sc-title">${s.icon} ${s.name} <span class="rr-grade grade-${rrGrade.toLowerCase()}">${s.rr.toFixed(1)}:1</span></div>
        <div class="sc-row"><span class="sc-k">Entry</span><span class="sc-v">$${s.entry.toFixed(2)}</span></div>
        <div class="sc-row"><span class="sc-k">Stop Loss</span><span class="sc-v" style="color:var(--red)">$${s.stop.toFixed(2)}</span></div>
        <div class="sc-row"><span class="sc-k">Target</span><span class="sc-v" style="color:var(--green)">$${s.target.toFixed(2)}</span></div>
        <div class="sc-row"><span class="sc-k">Risk ($)</span><span class="sc-v">$${s.risk.toFixed(2)}</span></div>
        <div class="sc-row"><span class="sc-k">Reward ($)</span><span class="sc-v">$${s.reward.toFixed(2)}</span></div>
        <div class="sc-row"><span class="sc-k">R:R Ratio</span><span class="sc-v" style="color:${rrColor}">${s.rr.toFixed(2)}:1</span></div>
        <div style="font-size:0.75rem;color:var(--text-muted);margin-top:8px">${s.desc}</div>
      </div>`;
  });
  scenariosHTML += `</div>`;

  return `
    <div class="rr-section">
      <div class="rr-header">
        <h4>⚖️ Risk : Reward Analysis</h4>
        <span style="font-size:0.8rem;color:var(--text-muted);font-family:'JetBrains Mono',monospace">
          ADR: $${adr.toFixed(2)} (${(adrPct*100).toFixed(2)}%) · 1σ: ${(sd*100).toFixed(2)}%
        </span>
      </div>
      <div class="rr-body">
        ${scoresHTML}
        ${levelsHTML}
        <h5 style="font-size:0.82rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--text-muted);margin:20px 0 14px">🎲 Trade Scenarios (from current price)</h5>
        ${scenariosHTML}
      </div>
    </div>`;
}

function estimateProb(outs, sd, multiplier, direction, totalOuts) {
  const qualifying = outs.filter(o => {
    if (direction === 'up') return o.ret >= sd * multiplier;
    return o.ret <= -sd * multiplier;
  }).length;

  if (totalOuts === 0) {
    if (multiplier >= 3) return 1;
    if (multiplier >= 2) return 5;
    return 16;
  }

  const histPct = Math.round((qualifying / Math.max(totalOuts, 1)) * 100);
  const theoretical = multiplier >= 3 ? 1 : multiplier >= 2 ? 5 : 16;
  return Math.max(Math.round(histPct * 0.6 + theoretical * 0.4), 1);
}
```

## 5. Export Brief — Add R:R Data to Clipboard Output

Find the `exportBrief()` function. Inside the `results.forEach(r => { ... })` loop, AFTER the volume conviction line:

```javascript
brief += `  Avg Outlier Vol:  ${avgVR.toFixed(1)}x avg ...
```

And BEFORE the outlier events table:

```javascript
// Outlier events table
if (r.outs.length > 0) {
```

Insert this block between them:

```javascript
    // R:R Scores
    const price = r.rs.pC;
    const trendScore = Math.round(((price - r.rs.pL) / (r.rs.pH - r.rs.pL)) * 100);
    const biasScore = r.outs.length > 0 ? Math.round((r.up / r.outs.length) * 100) : 50;
    const sdPct = r.rs.sd * 100;
    const volScore = sdPct < 1 ? 95 : sdPct < 2 ? 80 : sdPct < 3 ? 60 : sdPct < 5 ? 40 : 20;
    const freq = parseFloat(r.estYr) || 0;
    const catalystRisk = freq > 30 ? 90 : freq > 20 ? 70 : freq > 10 ? 50 : freq > 5 ? 30 : 15;
    const avgVRr = r.outs.length > 0 ? r.outs.reduce((s2,o) => s2+o.vr, 0) / r.outs.length : 1;
    const volConviction = Math.min(Math.round(avgVRr * 30), 100);
    const riskScore = Math.round((100 - volScore) * 0.3 + catalystRisk * 0.35 + (100 - biasScore) * 0.2 + (100 - trendScore) * 0.15);
    const rewardScore = Math.round(biasScore * 0.3 + trendScore * 0.2 + volConviction * 0.2 + (r.avgMv > 0 ? Math.min(r.avgMv * 100 * 8, 100) : 30) * 0.3);
    const rrRatio = riskScore > 0 ? (rewardScore / riskScore).toFixed(2) : '—';
    const rrNum = parseFloat(rrRatio) || 0;
    const grade = rrNum >= 1.5 ? 'A' : rrNum >= 1.0 ? 'B' : rrNum >= 0.7 ? 'C' : 'D';

    brief += `\n  RISK:REWARD SCORES\n`;
    brief += `  Bullish Bias:    ${biasScore}/100 (${r.up}/${r.outs.length} up)\n`;
    brief += `  Trend Position:  ${trendScore}/100 (${trendScore > 70 ? 'Near highs' : trendScore > 30 ? 'Mid-range' : 'Near lows'})\n`;
    brief += `  Vol Score:       ${volScore}/100 (${sdPct < 2 ? 'Low vol' : sdPct < 3 ? 'Moderate' : sdPct < 5 ? 'High' : 'Extreme'})\n`;
    brief += `  Catalyst Risk:   ${catalystRisk}/100 (~${r.estYr}/yr events)\n`;
    brief += `  Vol Conviction:  ${volConviction}/100 (${avgVRr.toFixed(1)}x avg vol)\n`;
    brief += `  Risk Score:      ${riskScore}/100\n`;
    brief += `  Reward Score:    ${rewardScore}/100\n`;
    brief += `  R:R Ratio:       ${rrRatio} — Grade ${grade}\n`;

    brief += `\n  KEY PRICE LEVELS\n`;
    brief += `  +3σ Target:      $${(price * (1 + r.rs.sd * 3)).toFixed(2)} (+${(r.rs.sd*300).toFixed(2)}%)\n`;
    brief += `  +2σ Target:      $${(price * (1 + r.rs.sd * 2)).toFixed(2)} (+${(r.rs.sd*200).toFixed(2)}%)\n`;
    brief += `  +1 ADR Target:   $${(price + r.rs.avgDR).toFixed(2)} (+${(r.rs.avgDRpct*100).toFixed(2)}%)\n`;
    brief += `  Period High:     $${r.rs.pH.toFixed(2)} (+${(((r.rs.pH - price)/price)*100).toFixed(2)}%)\n`;
    brief += `  ► CURRENT:       $${price.toFixed(2)}\n`;
    brief += `  -1 ADR Stop:     $${(price - r.rs.avgDR).toFixed(2)} (-${(r.rs.avgDRpct*100).toFixed(2)}%)\n`;
    brief += `  Period Low:      $${r.rs.pL.toFixed(2)} (${(((r.rs.pL - price)/price)*100).toFixed(2)}%)\n`;
    brief += `  -2σ Risk:        $${(price * (1 - r.rs.sd * 2)).toFixed(2)} (-${(r.rs.sd*200).toFixed(2)}%)\n`;
    brief += `  -3σ Risk:        $${(price * (1 - r.rs.sd * 3)).toFixed(2)} (-${(r.rs.sd*300).toFixed(2)}%)\n`;
```

---

## Score Formulas Reference

| Score | Formula | Scale |
|-------|---------|-------|
| Bullish Bias | `(upOutliers / totalOutliers) * 100` | 0=all down, 100=all up |
| Trend Position | `((price - periodLow) / (periodHigh - periodLow)) * 100` | 0=at lows, 100=at highs |
| Vol Score | Inverse of daily σ: <1%=95, <2%=80, <3%=60, <5%=40, 5%+=20 | Higher=calmer |
| Catalyst Risk | Based on est/year: >30=90, >20=70, >10=50, >5=30, else=15 | Higher=riskier |
| Vol Conviction | `min(avgVolumeRatio * 30, 100)` | Higher=more institutional |
| Risk Score | `(100-volScore)*0.3 + catalystRisk*0.35 + (100-biasScore)*0.2 + (100-trendScore)*0.15` | Composite |
| Reward Score | `biasScore*0.3 + trendScore*0.2 + volConviction*0.2 + avgMoveScore*0.3` | Composite |
| R:R Ratio | `rewardScore / riskScore` | ≥1.5=A, ≥1.0=B, ≥0.7=C, <0.7=D |

## Important Notes
- Do NOT modify any existing functions — only add new CSS, new functions, and insert `${buildRiskReward(r, th)}` in the template literal
- The `r` object passed to `buildRiskReward` contains: `{ ticker, bars, outs, rs, up, down, avgMv, estYr, ci }`
- `rs` comes from `rangeStats()` and includes: `pO, pC, pChg, pH, pL, hDate, lDate, range, rangePct, avgDR, avgDRpct, maxDR, minDR, maxDRpct, minDRpct, maxDRdate, minDRdate, sd, avgVol, days`
- Each outlier `o` in `outs` has: `date, dateStr, dateISO, ret, z, pc, c, dir, vr, news`
- The existing helper functions `fmtP()`, `fmtD()`, `fmtV()` are available globally
