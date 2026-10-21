"""Single-page visualization UI served at GET / (no external assets)."""

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LSTM 股票预测 · 资产配置</title>
<style>
:root{--bg:#0b1220;--panel:#131c2e;--panel2:#0f1729;--line:rgba(148,163,184,.18);
--text:#e2e8f0;--muted:#8ea0bb;--accent:#38bdf8;--pos:#34d399;--neg:#f87171;--amber:#fbbf24}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
background:radial-gradient(1200px 800px at 75% -10%, #123 0%, var(--bg) 55%);color:var(--text);line-height:1.6}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 70px}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;padding:18px 0 6px}
.kicker{color:var(--accent);text-transform:uppercase;letter-spacing:.2em;font-size:.76rem;font-weight:700}
h1{font-size:1.7rem;margin:.3rem 0 .2rem}
header p{color:var(--muted);margin:0;max-width:640px}
.badge{font-size:.78rem;font-weight:700;color:#052e16;background:linear-gradient(135deg,var(--pos),#6ee7b7);
padding:5px 12px;border-radius:999px;white-space:nowrap}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 20px;
box-shadow:0 10px 30px rgba(0,0,0,.25);margin-top:16px}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input[type=date],select{background:var(--panel2);color:var(--text);border:1px solid var(--line);
border-radius:10px;padding:8px 12px;font-size:.95rem}
button{border:none;border-radius:10px;padding:9px 18px;font-size:.92rem;cursor:pointer;
background:rgba(148,163,184,.15);color:var(--text)}
button:hover{background:rgba(148,163,184,.28)}
button.primary{background:var(--accent);color:#0b1120;font-weight:700}
button:disabled{opacity:.5;cursor:not-allowed}
.status{color:var(--muted);font-size:.88rem;margin-left:4px}
.status.err{color:var(--neg)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
@media (max-width:900px){.grid{grid-template-columns:1fr}}
h2{font-size:1.02rem;margin:0 0 8px;color:#cbd5e1;font-weight:600}
h2 small{color:var(--muted);font-weight:400;font-size:.78rem}
canvas{width:100%;height:auto;display:block}
table{width:100%;border-collapse:collapse;font-size:.9rem;margin-top:6px}
th,td{text-align:right;padding:7px 8px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:.8rem;margin-top:6px}
.legend i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
.note{color:var(--muted);font-size:.8rem;margin-top:12px}
footer{color:var(--muted);font-size:.8rem;text-align:center;margin-top:36px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <div class="kicker">LSTM Asset Allocation API</div>
    <h1>股票预测与配置建议</h1>
    <p>选择日期（使用该日及之前 252 个交易日的数据），查看冻结 LSTM 对未来 21 日收益的预测、隐含预测价格与组合权重。</p>
  </div>
  <span class="badge">离线 CPU 推理 · 冻结模型</span>
</header>

<div class="card">
  <div class="toolbar">
    <label>预测日期 <input type="date" id="dateInput"></label>
    <label>策略 <select id="strategy">
      <option value="lstm">LSTM 配置（作业要求）</option>
      <option value="qqq_gated">LSTM 择时 + 满仓 QQQ</option>
      <option value="qqq">只买 QQQ（基准）</option>
    </select></label>
    <button class="primary" id="runBtn">预测</button>
    <button id="latestBtn">最新日期</button>
    <span class="status" id="status">加载中…</span>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>预测未来 21 日收益 <small id="reqDate"></small></h2>
    <canvas id="retChart"></canvas>
  </div>
  <div class="card">
    <h2>建议配置权重</h2>
    <canvas id="wChart"></canvas>
  </div>
</div>

<div class="card">
  <h2>预测明细 <small>预测价格 = 最新收盘价 ×（1 + 预测收益）</small></h2>
  <table>
    <thead><tr><th>资产</th><th>最新收盘价</th><th>预测 21 日收益</th><th>隐含预测价格</th><th>建议权重</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
</div>

<div class="card">
  <h2>近 252 个交易日走势 <small>归一化（起点 = 100）</small></h2>
  <canvas id="priceChart"></canvas>
  <div class="legend" id="priceLegend"></div>
</div>

<div class="note">模型：60 天窗口 LSTM（64 隐层→实际 32 隐层冻结权重，训练见 training/train.py）。
数据截止 2023-12-29；预测仅为课程仿真，不构成投资建议。</div>

<footer>POST /predict 仍为严格契约端点 · GET /docs 查看接口文档</footer>
</div>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
const ETF = ["QQQ","SPY","IWM","IEF","TLT","GLD","SHY"];
const COLORS = ["#38bdf8","#34d399","#fbbf24","#f87171","#c084fc","#22d3ee","#94a3b8"];
const WCOLORS = {...Object.fromEntries(ETF.map((c,i)=>[c,COLORS[i]])), "CASH": "#64748b"};

function setup(canvas, h){
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.parentElement.clientWidth - 4;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx, w, h};
}

function drawReturns(data){
  const {ctx, w, h} = setup($("retChart"), 240);
  const vals = ETF.map(c => data.predicted_returns[c] * 100);
  const maxAbs = Math.max(0.6, ...vals.map(Math.abs));
  const padL = 44, padR = 12, padT = 14, padB = 30;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const zeroY = padT + plotH / 2;
  ctx.strokeStyle = "rgba(148,163,184,.4)";
  ctx.beginPath(); ctx.moveTo(padL, zeroY); ctx.lineTo(w - padR, zeroY); ctx.stroke();
  ctx.font = "11px system-ui"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  ctx.fillStyle = "#8ea0bb";
  for (const t of [-maxAbs, -maxAbs/2, 0, maxAbs/2, maxAbs]) {
    const y = zeroY - (t / maxAbs) * (plotH / 2);
    ctx.fillText(t.toFixed(1) + "%", padL - 8, y);
    if (t !== 0) {
      ctx.strokeStyle = "rgba(148,163,184,.12)";
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    }
  }
  const bw = plotW / ETF.length;
  ETF.forEach((c, i) => {
    const v = vals[i];
    const x = padL + i * bw + bw * 0.2;
    const bwid = bw * 0.6;
    const bh = Math.abs(v) / maxAbs * (plotH / 2);
    ctx.fillStyle = v >= 0 ? "#34d399" : "#f87171";
    if (v >= 0) ctx.fillRect(x, zeroY - bh, bwid, bh);
    else ctx.fillRect(x, zeroY, bwid, bh);
    ctx.fillStyle = "#cbd5e1"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(c, x + bwid/2, h - padB + 8);
    ctx.fillStyle = v >= 0 ? "#34d399" : "#f87171"; ctx.textBaseline = "bottom";
    ctx.fillText((v >= 0 ? "+" : "") + v.toFixed(2) + "%", x + bwid/2, v >= 0 ? zeroY - bh - 3 : zeroY + bh + 12);
  });
}

function drawWeights(data){
  const {ctx, w, h} = setup($("wChart"), 240);
  const entries = Object.entries(data.weights).filter(([k,v]) => v > 0.0005)
    .sort((a,b) => b[1]-a[1]);
  const rowH = Math.min(30, (h - 10) / Math.max(entries.length, 1));
  const padL = 46, padR = 56;
  ctx.font = "12px system-ui";
  entries.forEach(([k,v], i) => {
    const y = 8 + i * rowH;
    ctx.fillStyle = "#cbd5e1"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(k, padL - 8, y + rowH/2 - 4);
    const bw = (w - padL - padR) * v;
    ctx.fillStyle = WCOLORS[k] || "#94a3b8";
    ctx.beginPath();
    const bh = Math.max(10, rowH - 10);
    ctx.roundRect(padL, y, bw, bh, 4);
    ctx.fill();
    ctx.fillStyle = "#e2e8f0"; ctx.textAlign = "left";
    ctx.fillText((v*100).toFixed(1) + "%", padL + bw + 8, y + rowH/2 - 4);
  });
}

function drawPrices(data){
  const {ctx, w, h} = setup($("priceChart"), 260);
  const padL = 40, padR = 10, padT = 12, padB = 26;
  const norm = {};
  let ymin = Infinity, ymax = -Infinity;
  for (const c of ETF) {
    const s = data.history[c];
    const base = s[0];
    norm[c] = s.map(v => v / base * 100);
    for (const v of norm[c]) { if (v < ymin) ymin = v; if (v > ymax) ymax = v; }
  }
  const pad = (ymax - ymin) * 0.08 || 1;
  ymin -= pad; ymax += pad;
  const n = data.dates.length;
  const X = i => padL + (w - padL - padR) * i / (n - 1);
  const Y = v => padT + (h - padT - padB) * (1 - (v - ymin) / (ymax - ymin));
  ctx.font = "11px system-ui"; ctx.fillStyle = "#8ea0bb"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const v = ymin + (ymax - ymin) * i / 4;
    ctx.fillText(v.toFixed(0), padL - 6, Y(v));
    ctx.strokeStyle = "rgba(148,163,184,.12)";
    ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(w - padR, Y(v)); ctx.stroke();
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText(data.dates[0], padL + 30, h - padB + 6);
  ctx.fillText(data.dates[n-1], w - padR - 40, h - padB + 6);
  ETF.forEach((c, k) => {
    ctx.strokeStyle = COLORS[k]; ctx.lineWidth = 1.7;
    ctx.beginPath();
    norm[c].forEach((v, i) => { i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)); });
    ctx.stroke();
  });
  const legend = $("priceLegend");
  legend.innerHTML = ETF.map((c, k) => `<span><i style="background:${COLORS[k]}"></i>${c}</span>`).join("");
}

function fmtPct(v){ return (v >= 0 ? "+" : "") + (v*100).toFixed(2) + "%"; }

const STRATEGY_NAMES = {lstm: "LSTM 配置", qqq_gated: "LSTM 择时 + 满仓 QQQ", qqq: "只买 QQQ（基准）"};

function render(data){
  const label = STRATEGY_NAMES[data.strategy] || data.strategy;
  $("reqDate").textContent = "（数据截止 " + data.as_of_date + " · 策略：" + label + "）";
  $("dateInput").value = data.as_of_date;
  $("dateInput").min = data.min_date;
  $("dateInput").max = data.max_date;
  drawReturns(data);
  drawWeights(data);
  drawPrices(data);
  const rows = $("rows");
  rows.innerHTML = "";
  for (const c of ETF) {
    const r = data.predicted_returns[c];
    const price = data.last_close[c];
    const pred = price * (1 + r);
    const w = data.weights[c];
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${c}</td><td>${price.toFixed(2)}</td>
      <td class="${r>=0?'pos':'neg'}">${fmtPct(r)}</td>
      <td>${pred.toFixed(2)}</td><td>${(w*100).toFixed(1)}%</td>`;
    rows.appendChild(tr);
  }
  const cash = data.weights["CASH"];
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>CASH</td><td>1.00</td><td>0.00%</td><td>1.00</td><td>${(cash*100).toFixed(1)}%</td>`;
  rows.appendChild(tr);
}

async function run(date){
  const status = $("status");
  status.className = "status";
  status.textContent = "预测中…";
  $("runBtn").disabled = true;
  try {
    const params = new URLSearchParams({ strategy: $("strategy").value });
    if (date) params.set("as_of_date", date);
    const res = await fetch("/predict-ui?" + params.toString());
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
    render(data);
    status.textContent = "完成";
  } catch (err) {
    status.className = "status err";
    status.textContent = "失败：" + err.message;
  } finally {
    $("runBtn").disabled = false;
  }
}

$("runBtn").addEventListener("click", () => run($("dateInput").value));
$("latestBtn").addEventListener("click", () => run(null));
$("strategy").addEventListener("change", () => run($("dateInput").value || null));
window.addEventListener("resize", () => { if (window.__lastData) { drawReturns(window.__lastData); drawWeights(window.__lastData); drawPrices(window.__lastData); } });
const _render = render;
render = (d) => { window.__lastData = d; _render(d); };
run(null);
</script>
</body>
</html>
"""
