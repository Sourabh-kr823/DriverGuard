"""
dashboard/app.py
─────────────────
Flask + Socket.IO live dashboard — enhanced v2.
"""

import csv
import io
import time
import threading
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from flask import Flask, jsonify, render_template_string, Response
from flask_socketio import SocketIO

if TYPE_CHECKING:
    from modules.alert.alert_manager import AlertManager
    from modules.database.db_manager import DatabaseManager


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VSM Live Dashboard</title>
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
<style>
:root{--bg:#07090f;--bg2:#0c1018;--bg3:#111824;--border:#1c2a3a;
  --text:#c4d4e4;--text2:#6a8aaa;--green:#00e5a0;--amber:#f59e0b;
  --red:#ef4444;--blue:#3b82f6;--font-mono:'Courier New',monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:system-ui,sans-serif;font-size:12px}
.hdr{display:flex;align-items:center;gap:8px;padding:8px 14px;background:var(--bg2);
  border-bottom:1px solid var(--border);position:sticky;top:0;z-index:10}
.logo{font-family:var(--font-mono);font-size:14px;font-weight:bold;color:var(--green);letter-spacing:2px}
.badge{font-family:var(--font-mono);font-size:9px;padding:2px 7px;border-radius:3px;
  background:var(--bg3);color:var(--text2);border:1px solid var(--border)}
.live{background:#0a2e1a;color:var(--green);border-color:#00b87d}
.htime{margin-left:auto;font-family:var(--font-mono);color:var(--green)}
.alert-strip{display:none;background:var(--red);color:#fff;font-family:var(--font-mono);
  font-size:11px;font-weight:bold;text-align:center;padding:5px;animation:blink .6s infinite}
.alert-strip.show{display:block}
.prox-strip{display:none;background:#7c2d00;color:#fed7aa;font-family:var(--font-mono);
  font-size:11px;font-weight:bold;text-align:center;padding:5px 14px;
  border-bottom:2px solid #f97316;letter-spacing:.5px}
.prox-strip.show{display:block}
.prox-source{font-size:9px;opacity:.7;margin-left:8px;text-transform:uppercase}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.5}}
.main{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;padding:8px}
.panel{background:var(--bg2);border:1px solid var(--border);border-radius:6px;overflow:hidden}
.ph{display:flex;align-items:center;justify-content:space-between;padding:6px 10px;
  border-bottom:1px solid var(--border);font-size:10px;font-weight:600;letter-spacing:.5px;
  text-transform:uppercase;color:var(--text2);background:var(--bg3)}
.mc-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;padding:7px}
.mc{background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:5px 7px}
.ml{font-size:8px;letter-spacing:.8px;text-transform:uppercase;color:#334455;margin-bottom:2px}
.mv{font-family:var(--font-mono);font-size:15px;font-weight:bold;color:var(--green)}
.mv.warn{color:var(--amber)}.mv.danger{color:var(--red)}
.risk-row{display:flex;align-items:center;gap:10px;padding:5px 8px 4px}
.rb{font-family:var(--font-mono);font-size:12px;font-weight:bold;padding:3px 14px;border-radius:3px}
.r-low{background:#0a2e1a;color:var(--green);border:1px solid #00b87d}
.r-moderate{background:#2d1e00;color:var(--amber);border:1px solid #a16207}
.r-high{background:#2d0808;color:var(--red);border:1px solid #991b1b;animation:blink .7s infinite}
.det-list{padding:5px 7px;display:flex;flex-direction:column;gap:3px;min-height:44px}
.di{display:flex;align-items:center;gap:6px;border:1px solid var(--border);border-radius:4px;
  padding:3px 6px;font-family:var(--font-mono);font-size:10px;background:var(--bg3)}
.sd{width:7px;height:7px;border-radius:50%}
.sev-minor{background:#4ade80}.sev-moderate{background:var(--amber)}.sev-severe{background:var(--red)}
.gps-row{display:flex;flex-wrap:wrap;gap:10px;padding:6px 10px}
.gf{font-family:var(--font-mono);font-size:10px}
.gf .k{color:var(--text2)}.gf .v{color:var(--green);margin-left:3px}
.sw{padding:8px;background:var(--bg2);margin:8px;border-radius:6px}
.st{font-size:9px;color:var(--text2);letter-spacing:.5px;text-transform:uppercase;
  margin-bottom:6px;display:flex;justify-content:space-between;align-items:center}
.export-btn{font-family:var(--font-mono);font-size:9px;color:var(--green);text-decoration:none;
  padding:2px 7px;border:1px solid #00b87d;border-radius:3px;background:#0a2e1a}
.export-btn:hover{background:#0d3d22}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:0 8px 8px}
.stat-box{background:var(--bg2);border:1px solid var(--border);border-radius:6px;
  padding:12px;text-align:center}
.stat-val{font-family:var(--font-mono);font-size:28px;font-weight:bold;
  color:var(--green);line-height:1.1}
.stat-lbl{font-size:9px;color:var(--text2);letter-spacing:.5px;text-transform:uppercase;margin-top:4px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:0 8px 8px}
.ah-item{display:flex;align-items:center;gap:8px;padding:4px 8px;
  border-bottom:1px solid var(--border);font-family:var(--font-mono);font-size:10px}
.ah-item:last-child{border-bottom:none}
.ah-badge{font-size:9px;padding:1px 6px;border-radius:3px;font-weight:600;flex-shrink:0}
.ah-high{background:#2d0808;color:var(--red)}.ah-moderate{background:#2d1e00;color:var(--amber)}
</style>
</head>
<body>

<div class="hdr">
  <div class="logo">VSM <span style="color:var(--text2)">Live</span></div>
  <span class="badge live" id="conn-badge">● CONNECTING...</span>
  <span class="badge">DMS 30 FPS</span>
  <span class="badge">ROAD 30 FPS</span>
  <span class="badge">NEO-6M GPS</span>
  <div class="htime" id="htime">--:--:--</div>
  <button onclick="recalibrateDriver()" id="recal-btn"
    style="font-family:var(--font-mono);font-size:9px;padding:3px 10px;
           border:1px solid #00b87d;border-radius:3px;background:#0a2e1a;
           color:var(--green);cursor:pointer;margin-left:8px">
    NEW DRIVER
  </button>
</div>

<div class="alert-strip" id="alert-strip">
  WARNING: DROWSINESS DETECTED — PLEASE TAKE A BREAK
</div>

<div class="main">
  <!-- DMS Panel with circular gauge -->
  <div class="panel">
    <div class="ph"><span>Driver Monitoring</span><span>MediaPipe · Auto-Cal</span></div>
    <div class="mc-grid">
      <div class="mc"><div class="ml">EAR</div><div class="mv" id="v-ear">—</div></div>
      <div class="mc"><div class="ml">MAR</div><div class="mv" id="v-mar">—</div></div>
      <div class="mc"><div class="ml">YAW</div><div class="mv" id="v-yaw">—</div></div>
      <div class="mc"><div class="ml">PITCH</div><div class="mv" id="v-pitch">—</div></div>
      <div class="mc"><div class="ml">EYE FR</div><div class="mv" id="v-ecf">—</div></div>
      <div class="mc"><div class="ml">YAWN FR</div><div class="mv" id="v-yf">—</div></div>
    </div>
    <div class="risk-row">
      <div class="rb r-low" id="risk-badge">LOW</div>
      <span id="d-state" style="font-family:var(--font-mono);font-size:11px;color:var(--green)">ALERT</span>
    </div>
    <!-- Circular gauge -->
    <div style="display:flex;justify-content:center;padding:4px 0 8px">
      <svg viewBox="0 0 130 78" width="150" height="90">
        <path d="M 13,65 A 52,52 0 0,1 117,65" fill="none" stroke="#1c2a3a" stroke-width="13" stroke-linecap="round"/>
        <path id="gauge-arc" d="M 13,65 A 52,52 0 0,1 117,65" fill="none" stroke="#00e5a0"
          stroke-width="13" stroke-linecap="round" stroke-dasharray="163.4" stroke-dashoffset="163.4"/>
        <text id="gauge-pct" x="65" y="58" text-anchor="middle" fill="#00e5a0"
          font-size="20" font-family="monospace" font-weight="bold">0%</text>
        <text x="65" y="74" text-anchor="middle" fill="#6a8aaa"
          font-size="8" font-family="sans-serif" letter-spacing="1">RISK SCORE</text>
      </svg>
    </div>
  </div>

  <!-- Road Damage Panel with counters -->
  <div class="panel">
    <div class="ph"><span>Road Damage</span><span>YOLOv8-nano · RDD2022</span></div>
    <div class="det-list" id="det-list">
      <div style="color:var(--text2);font-size:10px;padding:4px">No detections</div>
    </div>
    <div style="border-top:1px solid var(--border);padding:6px 7px">
      <div style="font-size:8px;color:var(--text2);letter-spacing:.5px;text-transform:uppercase;margin-bottom:5px">Session counts</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px">
        <div class="mc"><div class="ml">Potholes</div>
          <div class="mv" id="cnt-pothole" style="color:var(--red)">0</div></div>
        <div class="mc"><div class="ml">Cracks</div>
          <div class="mv" id="cnt-crack" style="color:var(--amber)">0</div></div>
        <div class="mc"><div class="ml">Rutting</div>
          <div class="mv" id="cnt-rutting" style="color:var(--blue)">0</div></div>
      </div>
    </div>
  </div>

  <!-- GPS Panel -->
  <div class="panel">
    <div class="ph"><span>GPS — NEO-6M</span></div>
    <div class="gps-row">
      <div class="gf"><span class="k">LAT</span><span class="v" id="g-lat">—</span></div>
      <div class="gf"><span class="k">LON</span><span class="v" id="g-lon">—</span></div>
      <div class="gf"><span class="k">SPD</span><span class="v" id="g-spd">—</span></div>
    </div>
    <div class="gps-row">
      <div class="gf"><span class="k">BUZZ</span><span class="v" id="g-buzz">SILENT</span></div>
      <div class="gf"><span class="k">REASON</span><span class="v" id="g-reason">—</span></div>
    </div>
  </div>
</div>

<!-- Session statistics -->
<div class="stat-grid">
  <div class="stat-box">
    <div class="stat-val" id="stat-total">—</div>
    <div class="stat-lbl">Total Events</div>
  </div>
  <div class="stat-box">
    <div class="stat-val" style="color:var(--red)" id="stat-high">—</div>
    <div class="stat-lbl">HIGH Alerts</div>
  </div>
  <div class="stat-box">
    <div class="stat-val" style="color:var(--amber)" id="stat-avg-ear">—</div>
    <div class="stat-lbl">Avg EAR</div>
  </div>
  <div class="stat-box">
    <div class="stat-val" style="color:var(--blue)" id="stat-road">—</div>
    <div class="stat-lbl">Road Events</div>
  </div>
</div>

<!-- Alert history + State chart -->
<div class="two-col">
  <div class="sw" style="margin:0">
    <div class="st"><span>Recent Alerts — HIGH &amp; MODERATE</span></div>
    <div id="alert-history" style="max-height:180px;overflow-y:auto">
      <div style="color:var(--text2);font-size:10px;padding:4px">No alerts yet</div>
    </div>
  </div>
  <div class="sw" style="margin:0">
    <div class="st"><span>Driver State Distribution</span></div>
    <div style="height:160px;display:flex;align-items:center;justify-content:center">
      <canvas id="state-chart"></canvas>
    </div>
  </div>
</div>

<!-- EAR Trend Chart -->
<div class="sw">
  <div class="st">
    <span>EAR TREND — last 60 events</span>
    <a class="export-btn" href="/api/export/driver" download>Export Driver CSV</a>
  </div>
  <canvas id="ear-chart" height="70"></canvas>
</div>

<!-- Road Hazard Map -->
<div class="sw">
  <div class="st">
    <span>ROAD HAZARD MAP — click pins for details</span>
    <a class="export-btn" href="/api/export/road" download>Export Road CSV</a>
  </div>
  <div id="map" style="height:260px;border-radius:4px"></div>
  <div style="display:flex;gap:14px;padding:6px 4px 2px;font-size:9px;
    color:var(--text2);font-family:var(--font-mono)">
    <span><span style="color:#ef4444">&#9679;</span> Severe</span>
    <span><span style="color:#f59e0b">&#9679;</span> Moderate</span>
    <span><span style="color:#22c55e">&#9679;</span> Minor</span>
    <span style="margin-left:auto">Scroll=zoom · Drag=pan · Click pin=details</span>
  </div>
</div>

<!-- Driver Events Table -->
<div class="sw" style="overflow-x:auto">
  <div class="st"><span>RECENT DRIVER EVENTS</span></div>
  <div id="evt-table"></div>
</div>

<script>
const socket = io();
const $ = id => document.getElementById(id);
let lastRisk = 'low';

socket.on("connect",    () => { $("conn-badge").textContent = "● LIVE"; });
socket.on("disconnect", () => { $("conn-badge").textContent = "● OFFLINE"; });

function updateGauge(score, riskLevel) {
  const arc = $("gauge-arc"), pct = $("gauge-pct");
  const color = riskLevel === "high" ? "#ef4444" : riskLevel === "moderate" ? "#f59e0b" : "#00e5a0";
  arc.setAttribute("stroke-dashoffset", 163.4 * (1 - Math.min(score, 1)));
  arc.setAttribute("stroke", color);
  pct.textContent = Math.round(score * 100) + "%";
  pct.setAttribute("fill", color);
}

socket.on("vsm_update", data => {
  $("v-ear").textContent = data.ear.toFixed(2);
  $("v-ear").className   = "mv" + (data.ear < 0.22 ? " danger" : "");
  $("v-mar").textContent = data.mar.toFixed(2);
  $("v-mar").className   = "mv" + (data.mar > 1.70 ? " warn" : "");
  $("v-yaw").textContent = (data.yaw > 0 ? "+" : "") + data.yaw.toFixed(0) + "°";
  $("v-pitch").textContent = (data.pitch > 0 ? "+" : "") + data.pitch.toFixed(0) + "°";
  $("v-ecf").textContent = (data.eye_frames || 0) + " / 20";
  $("v-yf").textContent  = (data.yawn_frames || 0) + " / 45";
  const rb = $("risk-badge");
  rb.textContent = data.risk_level.toUpperCase();
  rb.className   = "rb r-" + data.risk_level;
  $("d-state").textContent = data.driver_state;
  $("d-state").style.color = data.risk_level === "high" ? "var(--red)" :
    data.risk_level === "moderate" ? "var(--amber)" : "var(--green)";
  updateGauge(data.risk_score || 0, data.risk_level);
  $("alert-strip").className = "alert-strip" + (data.risk_level === "high" ? " show" : "");
  const dl = $("det-list");
  dl.innerHTML = "";
  if (!data.road_detections || !data.road_detections.length) {
    dl.innerHTML = '<div style="color:var(--text2);font-size:10px;padding:4px">No detections</div>';
  } else {
    data.road_detections.forEach(d => {
      const row = document.createElement("div"); row.className = "di";
      row.innerHTML = `<div class="sd sev-${d.severity}"></div>${d.class_name||d.class} · ${d.severity} · ${d.confidence?(d.confidence*100).toFixed(0):"?"}%`;
      dl.appendChild(row);
    });
  }
  $("g-lat").textContent  = data.lat ? data.lat.toFixed(4) + "N" : "—";
  $("g-lon").textContent  = data.lon ? data.lon.toFixed(4) + "E" : "—";
  $("g-spd").textContent  = data.speed_kmh ? data.speed_kmh.toFixed(0) + " km/h" : "—";
  $("g-buzz").textContent = data.buzzer_active ? "ACTIVE" : "SILENT";
  $("g-buzz").style.color = data.buzzer_active ? "var(--red)" : "var(--green)";
  $("g-reason").textContent = data.buzzer_reason || "—";
  if (data.risk_level === "high" && lastRisk !== "high") {
    try {
      const ctx = new AudioContext(), osc = ctx.createOscillator(), gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination); osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
      osc.start(); osc.stop(ctx.currentTime + 0.5);
    } catch(e) {}
  }
  lastRisk = data.risk_level;
});

setInterval(() => { $("htime").textContent = new Date().toLocaleTimeString(); }, 1000);

function recalibrateDriver() {
  const btn = $("recal-btn");
  btn.textContent = "CALIBRATING...";
  btn.style.color = "#f59e0b";
  btn.style.borderColor = "#a16207";
  btn.disabled = true;
  fetch("/api/recalibrate", { method: "POST" })
    .then(r => r.json())
    .then(d => {
      // Show progress then restore button after ~4s (calibration takes ~3s)
      setTimeout(() => {
        btn.textContent = "NEW DRIVER";
        btn.style.color = "var(--green)";
        btn.style.borderColor = "#00b87d";
        btn.disabled = false;
      }, 4000);
    })
    .catch(() => {
      btn.textContent = "NEW DRIVER";
      btn.style.color = "var(--green)";
      btn.style.borderColor = "#00b87d";
      btn.disabled = false;
    });
}

function refreshStats() {
  fetch("/api/session_stats").then(r => r.json()).then(d => {
    $("stat-total").textContent   = d.total_driver;
    $("stat-high").textContent    = d.high_count;
    $("stat-avg-ear").textContent = d.avg_ear.toFixed(2);
    $("stat-road").textContent    = d.road_total;
    $("cnt-pothole").textContent  = d.pothole;
    $("cnt-crack").textContent    = d.crack;
    $("cnt-rutting").textContent  = d.rutting;
    stateChart.data.datasets[0].data = [d.alert_count, d.drowsy_count, d.distracted_count, d.fatigued_count];
    stateChart.update("none");
  }).catch(() => {});
}
refreshStats();
setInterval(refreshStats, 5000);

function refreshAlertHistory() {
  fetch("/api/events/driver").then(r => r.json()).then(rows => {
    const alerts = rows.filter(r => r.risk === "high" || r.risk === "moderate");
    const ah = $("alert-history");
    if (!alerts.length) {
      ah.innerHTML = '<div style="color:var(--text2);font-size:10px;padding:4px">No alerts yet</div>';
      return;
    }
    ah.innerHTML = alerts.slice(0, 10).map(r => {
      const bc = r.risk === "high" ? "ah-high" : "ah-moderate";
      return `<div class="ah-item">
        <span class="ah-badge ${bc}">${r.risk.toUpperCase()}</span>
        <span style="color:var(--text2)">${(r.ts_iso||"").slice(11,19)}</span>
        <span style="color:var(--text)">${r.state||""}</span>
        <span style="margin-left:auto;color:var(--text2)">EAR ${(r.ear||0).toFixed(2)}</span>
      </div>`;
    }).join("");
  }).catch(() => {});
}
refreshAlertHistory();
setInterval(refreshAlertHistory, 5000);

const stateChart = new Chart(document.getElementById("state-chart"), {
  type: "doughnut",
  data: {
    labels: ["ALERT","DROWSY","DISTRACTED","FATIGUED"],
    datasets: [{
      data: [0,0,0,0],
      backgroundColor: ["#00e5a0","#ef4444","#f59e0b","#3b82f6"],
      borderWidth: 0, hoverOffset: 4
    }]
  },
  options: {
    animation: false,
    plugins: {
      legend: { position:"right", labels:{ color:"#6a8aaa", font:{size:9}, padding:8, boxWidth:10 } }
    },
    cutout: "65%"
  }
});

const earChart = new Chart(document.getElementById("ear-chart"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      { data:[], borderColor:"#00e5a0", borderWidth:1.5, pointRadius:0, tension:0.4, fill:false },
      { data:[], borderColor:"#ef4444", borderWidth:1, borderDash:[4,4], pointRadius:0, fill:false }
    ]
  },
  options: {
    animation: false,
    plugins: { legend: { display:false } },
    scales: {
      x: { display:false },
      y: { min:0, max:0.55, ticks:{color:"#6a8aaa",font:{size:9}}, grid:{color:"rgba(100,100,100,.15)"} }
    }
  }
});

function refreshChart() {
  fetch("/api/ear_history").then(r => r.json()).then(data => {
    earChart.data.labels           = data.map(d => (d.ts||"").slice(11,19));
    earChart.data.datasets[0].data = data.map(d => d.ear);
    earChart.data.datasets[1].data = data.map(() => 0.22);
    earChart.update("none");
  }).catch(() => {});
}
refreshChart();
setInterval(refreshChart, 3000);

const map = L.map("map").setView([12.9716, 77.5946], 14);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",{attribution:"OpenStreetMap"}).addTo(map);
const markers = L.layerGroup().addTo(map);
function refreshMap() {
  fetch("/api/events/road").then(r => r.json()).then(events => {
    markers.clearLayers();
    events.forEach(e => {
      if (!e.lat || !e.lon) return;
      const c = e.severity==="severe" ? "#ef4444" : e.severity==="moderate" ? "#f59e0b" : "#22c55e";
      L.circleMarker([e.lat,e.lon],{color:c,fillColor:c,fillOpacity:.75,radius:7,weight:1.5})
       .bindPopup("<b>"+(e.class_name||"damage")+"</b><br>Severity: "+e.severity+"<br>Confidence: "+Math.round(e.confidence*100)+"%<br>"+(e.ts_iso||"").slice(0,19))
       .addTo(markers);
    });
  }).catch(() => {});
}
refreshMap();
setInterval(refreshMap, 5000);

function refreshTable() {
  fetch("/api/events/driver").then(r => r.json()).then(rows => {
    const cols = ["time","ear","mar","risk","state","lat","lon"];
    let h = '<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:10px">';
    h += "<tr>"+cols.map(c=>`<th style="padding:4px 8px;background:#0d1b2a;color:#00e5a0;text-align:left">${c}</th>`).join("")+"</tr>";
    rows.slice(0,12).forEach(r => {
      const rc = r.risk==="high"?"#ef4444":r.risk==="moderate"?"#f59e0b":"#6a8aaa";
      const vals = [(r.ts_iso||"").slice(11,19),(r.ear||0).toFixed(3),(r.mar||0).toFixed(3),
        `<span style="color:${rc}">${r.risk||""}</span>`,r.state||"",
        r.lat?(+r.lat).toFixed(4):"—",r.lon?(+r.lon).toFixed(4):"—"];
      h += "<tr>"+vals.map(v=>`<td style="padding:3px 8px;border-bottom:1px solid #1c2a3a;color:#c4d4e4">${v}</td>`).join("")+"</tr>";
    });
    $("evt-table").innerHTML = h+"</table>";
  }).catch(() => {});
}
refreshTable();
setInterval(refreshTable, 5000);
</script>
</body>
</html>"""


def create_app(alert_manager, db_manager, cfg: dict, proximity_manager=None):
    app      = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    push_interval = 1.0 / 10

    # ── Session scope ─────────────────────────────────────────────────────
    # Record the moment this session started; all DB queries are filtered to
    # rows with ts_iso >= this value so historical data never bleeds in.
    session_start_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    logger.info(f"[Dashboard] Session start: {session_start_iso}")

    def _filter_session(rows: list) -> list:
        """Return only rows that belong to the current session."""
        return [r for r in rows if (r.get("ts_iso") or "") >= session_start_iso]
    # ──────────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template_string(_DASHBOARD_HTML)

    @app.route("/api/state")
    def api_state():
        return jsonify(alert_manager.state.__dict__)

    @app.route("/api/events/driver")
    def api_driver_events():
        return jsonify(_filter_session(db_manager.query_recent_driver_events(50)))

    @app.route("/api/events/road")
    def api_road_events():
        return jsonify(_filter_session(db_manager.query_recent_road_events(50)))

    @app.route("/api/stats")
    def api_stats():
        return jsonify(db_manager.query_stats())

    @app.route("/api/recalibrate", methods=["POST"])
    def api_recalibrate():
        """Reset DMS calibration for a new driver. Models stay loaded."""
        try:
            alert_manager.dms.recalibrate()
            logger.info("[Dashboard] Recalibration requested via dashboard")
            return jsonify({"status": "recalibrating",
                            "message": "Sit normally — calibrating for new driver (~3s)"})
        except Exception as e:
            logger.error(f"[Dashboard] Recalibrate error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/ear_history")
    def api_ear_history():
        rows = _filter_session(db_manager.query_recent_driver_events(60))
        return jsonify([{"ts":r.get("ts_iso",""),"ear":r.get("ear",0.0),
                         "mar":r.get("mar",0.0),"risk":r.get("risk","low")} for r in rows])

    @app.route("/api/session_stats")
    def api_session_stats():
        driver_rows = _filter_session(db_manager.query_recent_driver_events(500))
        road_rows   = _filter_session(db_manager.query_recent_road_events(500))
        total    = len(driver_rows)
        high     = sum(1 for r in driver_rows if r.get("risk") == "high")
        moderate = sum(1 for r in driver_rows if r.get("risk") == "moderate")
        avg_ear  = round(sum(r.get("ear",0) for r in driver_rows) / max(total,1), 3)
        alert_c      = sum(1 for r in driver_rows if r.get("state") == "ALERT")
        drowsy_c     = sum(1 for r in driver_rows if r.get("state") == "DROWSY")
        distracted_c = sum(1 for r in driver_rows if r.get("state") == "DISTRACTED")
        fatigued_c   = sum(1 for r in driver_rows if r.get("state") == "FATIGUED")
        road_total = len(road_rows)
        pothole  = sum(1 for r in road_rows if "pothole"  in (r.get("class_name") or ""))
        crack    = sum(1 for r in road_rows if "crack"    in (r.get("class_name") or ""))
        rutting  = sum(1 for r in road_rows if "rutting"  in (r.get("class_name") or ""))
        repair   = sum(1 for r in road_rows if "repair"   in (r.get("class_name") or ""))
        return jsonify({"total_driver":total,"high_count":high,"moderate_count":moderate,
            "avg_ear":avg_ear,"alert_count":alert_c,"drowsy_count":drowsy_c,
            "distracted_count":distracted_c,"fatigued_count":fatigued_c,
            "road_total":road_total,"pothole":pothole,"crack":crack,
            "rutting":rutting,"repair":repair})

    @app.route("/api/proximity")
    def api_proximity():
        """Return the current proximity alert (if any)."""
        prox = getattr(api_proximity, "_manager", None)
        if prox is None:
            return jsonify(None)
        return jsonify(prox.latest_alert)

    @app.route("/api/export/driver")
    def export_driver_csv():
        rows = _filter_session(db_manager.query_recent_driver_events(500))
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        return Response(output.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition":"attachment; filename=driver_events.csv"})

    @app.route("/api/export/road")
    def export_road_csv():
        rows = _filter_session(db_manager.query_recent_road_events(500))
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        return Response(output.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition":"attachment; filename=road_events.csv"})

    def _push_loop():
        logger.info("[Dashboard] Push loop started")
        while True:
            time.sleep(push_interval)
            try:
                s   = alert_manager.state
                msg = {k: v for k, v in s.__dict__.items() if k != "annotated_frame"}
                socketio.emit("vsm_update", msg)
                # Push proximity alert separately for targeted UI handling
                if proximity_manager is not None:
                    pa = proximity_manager.latest_alert
                    socketio.emit("proximity_alert", pa or {})
            except Exception as e:
                logger.debug(f"[Dashboard] Push error: {e}")

    socketio.start_background_task(_push_loop)
    return app, socketio