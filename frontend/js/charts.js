/**
 * charts.js — ApexCharts visualisations for the Analytics tab.
 *
 * Exports a single `renderCharts(metrics)` function that builds
 * rich chart areas:
 *   1. Fraud score histogram (gradient area chart)
 *   2. Risk tier donut with animated counts
 *   3. Action distribution (radial bar)
 *   4. Pattern breakdown (polar area)
 *   5. Transaction volume sparkline header stats
 */

// ── Shared ApexCharts base theme ─────────────────────────────────────────────
const BASE_OPTS = {
  theme: { mode: "dark" },
  chart: {
    background: "transparent",
    fontFamily: "Inter, system-ui, sans-serif",
    toolbar: { show: false },
    animations: {
      enabled: true,
      easing: "easeinout",
      speed: 700,
      animateGradually: { enabled: true, delay: 80 },
    },
  },
  grid: {
    borderColor: "rgba(255,255,255,0.06)",
    strokeDashArray: 4,
  },
  tooltip: {
    theme: "dark",
    style: { fontFamily: "Inter, system-ui, sans-serif" },
  },
};

let _charts = [];

function _destroy() {
  _charts.forEach((c) => { try { c.destroy(); } catch { } });
  _charts = [];
}

function _make(el, opts) {
  if (!el) return null;
  // Merge base options with provided opts (nested merge for chart key)
  const merged = {
    ...BASE_OPTS,
    ...opts,
    chart: { ...BASE_OPTS.chart, ...(opts.chart || {}) },
    grid: { ...BASE_OPTS.grid, ...(opts.grid || {}) },
    tooltip: { ...BASE_OPTS.tooltip, ...(opts.tooltip || {}) },
  };
  const c = new ApexCharts(el, merged);
  c.render();
  _charts.push(c);
  return c;
}

// ── Colour palettes ───────────────────────────────────────────────────────────
const RISK_COLORS = { LOW: "#22c55e", MEDIUM: "#eab308", HIGH: "#f97316", CRITICAL: "#ef4444" };
const ACTION_COLORS = { PASS: "#22c55e", SILENT_FLAG: "#eab308", HOLD: "#f97316", BLOCK: "#ef4444" };
const PATTERN_COLORS = { MULE_NETWORK: "#a855f7", ACCOUNT_TAKEOVER: "#ef4444", VELOCITY_SPIKE: "#f97316" };
const INDIGO = "#6366f1";
const PURPLE = "#a855f7";

export function renderCharts(metrics) {
  _destroy();

  const {
    score_histogram = Array(10).fill(0),
    risk_counts = {},
    action_counts = {},
    pattern_counts = {},
    total_transactions = 0,
    fraud_count = 0,
    fraud_rate = 0,
    avg_risk_score = 0,
  } = metrics;

  // ── Inject stat header cards ─────────────────────────────────────────────
  _renderStatHeader({ total_transactions, fraud_count, fraud_rate, avg_risk_score, action_counts });

  // ── 1. Fraud score bar chart (Transaction Monitoring Agent predictions) ──
  const barColors = score_histogram.map((_, i) => {
    if (i >= 8) return "#ef4444";       // 80-100%: critical red
    if (i >= 5) return "#f97316";       // 50-80%: high orange
    if (i >= 3) return "#eab308";       // 30-50%: medium yellow
    return INDIGO;                      // 0-30%: normal indigo
  });

  _make(document.getElementById("chart-histogram"), {
    chart: { type: "bar", height: 260 },
    series: [{ name: "Transactions", data: score_histogram }],
    xaxis: {
      categories: ["0–10%", "10–20%", "20–30%", "30–40%", "40–50%", "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"],
      labels: { style: { colors: "#94a3b8", fontSize: "10px" } },
      axisBorder: { show: false },
      axisTicks: { show: false },
    },
    yaxis: { labels: { style: { colors: "#94a3b8", fontSize: "11px" } } },
    colors: barColors,
    plotOptions: {
      bar: {
        distributed: true,
        borderRadius: 4,
        columnWidth: "70%",
        dataLabels: { position: "top" },
      },
    },
    dataLabels: {
      enabled: true,
      formatter: (val) => (val > 0 ? val : ""),
      style: { fontSize: "10px", colors: ["#94a3b8"] },
      offsetY: -18,
    },
    legend: { show: false },
    title: {
      text: "Fraud Score Distribution (Transaction Monitoring Agent)",
      style: { color: "#e2e8f0", fontSize: "13px", fontWeight: "600" },
    },
  });


  // ── 2. Risk tier donut ────────────────────────────────────────────────────
  const riskOrder = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
  const riskLabels = riskOrder.filter((k) => k in risk_counts);
  const riskValues = riskLabels.map((k) => risk_counts[k] || 0);

  if (riskLabels.length) {
    _make(document.getElementById("chart-risk-donut"), {
      chart: { type: "donut", height: 260 },
      series: riskValues,
      labels: riskLabels,
      colors: riskLabels.map((l) => RISK_COLORS[l] || INDIGO),
      legend: {
        position: "bottom",
        labels: { colors: "#94a3b8" },
        fontSize: "11px",
      },
      plotOptions: {
        pie: {
          donut: {
            size: "62%",
            labels: {
              show: true,
              total: {
                show: true,
                label: "Total",
                color: "#94a3b8",
                fontSize: "12px",
                formatter: (w) => w.globals.seriesTotals.reduce((a, b) => a + b, 0).toLocaleString(),
              },
              value: {
                color: "#e2e8f0",
                fontSize: "22px",
                fontWeight: "700",
              },
            },
          },
        },
      },
      dataLabels: { enabled: false },
      title: {
        text: "Risk Tier Distribution",
        style: { color: "#e2e8f0", fontSize: "13px", fontWeight: "600" },
      },
      stroke: { width: 0 },
    });
  } else {
    const el = document.getElementById("chart-risk-donut");
    if (el) el.innerHTML = `<p class="text-slate-500 text-sm text-center pt-10">No risk data.</p>`;
  }

  // ── 3. Action distribution — radial bar ───────────────────────────────────
  const actionOrder = ["PASS", "SILENT_FLAG", "HOLD", "BLOCK"];
  const actionLabels = actionOrder.filter((k) => (action_counts[k] || 0) > 0);
  const actionValues = actionLabels.map((k) => action_counts[k] || 0);
  const totalActions = actionValues.reduce((a, b) => a + b, 0) || 1;
  // Radial bar expects percentage values
  const actionPct = actionValues.map((v) => Math.round((v / totalActions) * 100));

  if (actionLabels.length) {
    _make(document.getElementById("chart-actions"), {
      chart: { type: "radialBar", height: 260 },
      series: actionPct,
      labels: actionLabels.map((l) => l.replace("_", " ")),
      colors: actionLabels.map((l) => ACTION_COLORS[l] || INDIGO),
      plotOptions: {
        radialBar: {
          offsetY: 0,
          startAngle: -135,
          endAngle: 135,
          hollow: { size: "20%" },
          track: { background: "rgba(255,255,255,0.05)", strokeWidth: "97%" },
          dataLabels: {
            name: { fontSize: "11px", color: "#94a3b8", offsetY: -10 },
            value: { fontSize: "14px", color: "#e2e8f0", fontWeight: "700", formatter: (v) => `${v}%` },
            total: {
              show: true,
              label: "Actions",
              color: "#94a3b8",
              fontSize: "11px",
              formatter: () => totalActions.toLocaleString(),
            },
          },
        },
      },
      legend: {
        show: true,
        position: "bottom",
        labels: { colors: "#94a3b8" },
        fontSize: "11px",
        formatter: (label, opts) => `${label}: ${actionValues[opts.seriesIndex]}`,
      },
      title: {
        text: "Action Distribution",
        style: { color: "#e2e8f0", fontSize: "13px", fontWeight: "600" },
      },
    });
  } else {
    const el = document.getElementById("chart-actions");
    if (el) el.innerHTML = `<p class="text-slate-500 text-sm text-center pt-10">No action data.</p>`;
  }

  // ── 4. Pattern breakdown — polar area ────────────────────────────────────
  const patLabels = Object.keys(PATTERN_COLORS).filter((k) => (pattern_counts[k] || 0) > 0);
  const patValues = patLabels.map((k) => pattern_counts[k] || 0);

  if (patLabels.length) {
    _make(document.getElementById("chart-patterns"), {
      chart: { type: "polarArea", height: 260 },
      series: patValues,
      labels: patLabels.map((l) => l.replace("_", " ")),
      colors: patLabels.map((l) => PATTERN_COLORS[l] || INDIGO),
      legend: {
        position: "bottom",
        labels: { colors: "#94a3b8" },
        fontSize: "11px",
      },
      plotOptions: {
        polarArea: {
          rings: { strokeWidth: 1, strokeColor: "rgba(255,255,255,0.06)" },
          spokes: { strokeWidth: 1, connectorColors: "rgba(255,255,255,0.06)" },
        },
      },
      fill: { opacity: 0.8 },
      stroke: { colors: ["transparent"] },
      dataLabels: {
        enabled: true,
        formatter: (val, opts) => opts.w.globals.labels[opts.seriesIndex],
        style: { fontSize: "11px", colors: ["#ffffff"] },
        dropShadow: { enabled: false },
      },
      title: {
        text: "Attack Pattern Breakdown",
        style: { color: "#e2e8f0", fontSize: "13px", fontWeight: "600" },
      },
    });
  } else {
    const el = document.getElementById("chart-patterns");
    if (el) el.innerHTML = `<p class="text-slate-500 text-sm text-center pt-8 px-3">No attack patterns detected in this dataset.</p>`;
  }

  // ── 5. Score histogram mini sparkline in Analytics header ────────────────
  _renderScoreSparklines(score_histogram, fraud_count, total_transactions);
}

// ── Stat header cards (injected into analytics-stat-row) ─────────────────────
function _renderStatHeader({ total_transactions, fraud_count, fraud_rate, avg_risk_score, action_counts }) {
  const row = document.getElementById("analytics-stat-row");
  if (!row) return;
  const blocked = action_counts.BLOCK || 0;
  const held = action_counts.HOLD || 0;
  row.innerHTML = `
    <div class="glass p-4 flex items-center gap-3">
      <div class="w-10 h-10 rounded-xl bg-indigo-500/15 flex items-center justify-center flex-shrink-0">
        <i class="ph ph-list-numbers text-indigo-400 text-xl"></i>
      </div>
      <div>
        <p class="text-slate-400 text-xs">Analysed</p>
        <p class="text-white font-bold text-xl">${total_transactions.toLocaleString()}</p>
      </div>
    </div>
    <div class="glass p-4 flex items-center gap-3">
      <div class="w-10 h-10 rounded-xl bg-red-500/15 flex items-center justify-center flex-shrink-0">
        <i class="ph ph-warning text-red-400 text-xl"></i>
      </div>
      <div>
        <p class="text-slate-400 text-xs">Fraud Detected</p>
        <p class="text-red-400 font-bold text-xl">${fraud_count.toLocaleString()} <span class="text-sm font-normal opacity-70">(${fraud_rate}%)</span></p>
      </div>
    </div>
    <div class="glass p-4 flex items-center gap-3">
      <div class="w-10 h-10 rounded-xl bg-orange-500/15 flex items-center justify-center flex-shrink-0">
        <i class="ph ph-hand text-orange-400 text-xl"></i>
      </div>
      <div>
        <p class="text-slate-400 text-xs">Blocked / Held</p>
        <p class="text-orange-400 font-bold text-xl">${blocked + held}</p>
      </div>
    </div>
    <div class="glass p-4 flex items-center gap-3">
      <div class="w-10 h-10 rounded-xl bg-purple-500/15 flex items-center justify-center flex-shrink-0">
        <i class="ph ph-activity text-purple-400 text-xl"></i>
      </div>
      <div>
        <p class="text-slate-400 text-xs">Avg Risk Score</p>
        <p class="text-purple-400 font-bold text-xl">${Number(avg_risk_score).toFixed(3)}</p>
      </div>
    </div>
  `;
}

// ── Mini sparkline bars inline with score tiers ───────────────────────────────
function _renderScoreSparklines(histogram, fraudCount, total) {
  const el = document.getElementById("chart-score-spark");
  if (!el) return;
  const max = Math.max(...histogram, 1);
  const highRiskPct = histogram.slice(5).reduce((a, b) => a + b, 0);
  el.innerHTML = `
    <div class="flex flex-col gap-1">
      <p class="text-slate-400 text-xs mb-2 font-medium">Score Bucket Breakdown</p>
      ${histogram.map((v, i) => {
    const pct = Math.round((v / max) * 100);
    const isHigh = i >= 5;
    return `
          <div class="flex items-center gap-2">
            <span class="text-slate-500 text-[10px] w-14 flex-shrink-0">${i * 10}–${i * 10 + 10}%</span>
            <div class="flex-1 bg-white/5 rounded-full h-1.5 overflow-hidden">
              <div class="h-full rounded-full transition-all duration-700"
                   style="width:${pct}%; background:${isHigh ? '#ef4444' : '#6366f1'}; opacity:${0.4 + pct / 130}"></div>
            </div>
            <span class="text-slate-400 text-[10px] w-8 text-right">${v}</span>
          </div>
        `;
  }).join("")}
    </div>
    <div class="mt-3 pt-3 border-t border-white/10 flex items-center justify-between">
      <span class="text-slate-400 text-xs">High-risk (≥50%) transactions</span>
      <span class="text-red-400 font-semibold text-sm">${highRiskPct}</span>
    </div>
  `;
}
