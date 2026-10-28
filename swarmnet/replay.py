"""Replay dashboard: one self-contained HTML file that plays back a recorded mission.

    from swarmnet.replay import save_html
    save_html(world.trace(), "replays/auction.html")
"""
import json
import os

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SwarmNet Replay</title>
<style>
:root{--bg:#fcfcfb;--panel:#f3f2ee;--ink:#0b0b0b;--muted:#52514e;--line:#dddcd6;
 --work:#2a78d6;--home:#6250d6;--idle:#8f8e87;--dead:#e34948;--held:#eda100;--dlv:#1baf7a;--open:#b9b8b0;
 --link:#1baf7a;--link2:#c9c8c0;--base:#0b0b0b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--panel:#242423;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3a38;
 --work:#3987e5;--home:#9085e9;--idle:#8f8e87;--dead:#e66767;--held:#c98500;--dlv:#199e70;--open:#5b5a55;--link:#199e70;--link2:#4a4a47;--base:#f0efec}}
:root[data-theme="dark"]{--bg:#1a1a19;--panel:#242423;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3a38;
 --work:#3987e5;--home:#9085e9;--idle:#8f8e87;--dead:#e66767;--held:#c98500;--dlv:#199e70;--open:#5b5a55;--link:#199e70;--link2:#4a4a47;--base:#f0efec}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 14px}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px}
.stat{background:var(--panel);border-radius:8px;padding:8px 12px;min-width:110px}
.stat b{display:block;font-size:18px}.stat span{color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}
@media (max-width:820px){.grid{grid-template-columns:1fr}}
canvas{width:100%;height:auto;background:var(--panel);border-radius:8px;display:block}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:10px 0}
button{background:var(--ink);color:var(--bg);border:0;border-radius:6px;padding:7px 14px;font:inherit;cursor:pointer}
input[type=range]{flex:1;min-width:140px}
label{color:var(--muted);font-size:13px;display:flex;align-items:center;gap:4px}
.side{background:var(--panel);border-radius:8px;padding:14px}
.side h2{font-size:14px;margin:0 0 8px}
.leg{display:grid;grid-template-columns:18px 1fr;gap:6px 8px;align-items:center;font-size:13px;margin-bottom:14px}
.dot{width:12px;height:12px;border-radius:50%}
.ln{width:16px;height:0;border-top:2px solid}
.ev{font-size:13px;color:var(--muted);margin:0;padding-left:16px}
table{width:100%;border-collapse:collapse;font-size:13px}td{padding:3px 0;border-bottom:1px solid var(--line)}td:last-child{text-align:right}
</style></head><body><div class="wrap">
<h1 id="title">SwarmNet replay</h1><p class="sub" id="subtitle"></p>
<div class="stats">
 <div class="stat"><b id="s-time">0</b><span>seconds</span></div>
 <div class="stat"><b id="s-dlv">0</b><span>readings delivered</span></div>
 <div class="stat"><b id="s-alive">0</b><span>drones flying</span></div>
 <div class="stat"><b id="s-contact">0</b><span>drones in contact with base</span></div>
 <div class="stat"><b id="s-lam">0</b><span>network health (algebraic connectivity)</span></div>
</div>
<div class="grid"><div>
 <canvas id="cv" width="900" height="900"></canvas>
 <div class="controls">
  <button id="play">Play</button>
  <input type="range" id="scrub" min="0" value="0" aria-label="Time">
  <label>Speed <select id="speed"><option value="1">1x</option><option value="3" selected>3x</option><option value="8">8x</option></select></label>
  <label><input type="checkbox" id="c-links" checked> radio links</label>
  <label><input type="checkbox" id="c-range"> radio range</label>
  <label><input type="checkbox" id="c-tgt" checked> targets</label>
 </div>
</div>
<div class="side">
 <h2>Legend</h2>
 <div class="leg">
  <div class="dot" style="background:var(--work)"></div><div>Drone working (flying to a point or surveying)</div>
  <div class="dot" style="background:var(--home)"></div><div>Drone heading home or recharging</div>
  <div class="dot" style="background:var(--idle)"></div><div>Drone parked at base</div>
  <div class="dot" style="background:var(--dead)"></div><div>Drone failed</div>
  <div class="dot" style="background:var(--open);border:1px solid var(--muted)"></div><div>Point not surveyed yet</div>
  <div class="dot" style="background:var(--held)"></div><div>Surveyed, reading not home yet</div>
  <div class="dot" style="background:var(--dlv)"></div><div>Reading delivered to base</div>
  <div class="ln" style="border-color:var(--link)"></div><div>Radio link connected to base</div>
  <div class="ln" style="border-color:var(--link2);border-top-style:dashed"></div><div>Radio link cut off from base</div>
 </div>
 <h2>Events</h2><ul class="ev" id="events"></ul>
 <h2 style="margin-top:14px">Final result</h2><table id="final"></table>
</div></div></div>
<script>
const TRACE = __TRACE__;
const M = TRACE.meta, F = TRACE.frames;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const $ = id => document.getElementById(id);
const names = {central:'Central planner',auction:'Auction swarm (CBBA)',nearest:'Nearest-task swarm',learned:'Learned swarm (GNN + MAPPO)'};
$('title').textContent = 'SwarmNet replay: ' + (names[M.method] || M.method);
$('subtitle').textContent = `${M.n_drones} drones, ${M.n_tasks} survey points, radio range ${M.radio_range >= 1e5 ? 'unlimited' : M.radio_range + ' m'}, ${M.field} m field`;
$('scrub').max = F.length - 1;
$('events').innerHTML = M.events.length ? M.events.map(e => `<li>t = ${e.t}s: drone ${e.drone} ${e.what}${e.lost ? ` (${e.lost} readings lost)` : ''}</li>`).join('') : '<li>No failures</li>';
const fm = M.metrics, rows = [['Finished', fm.finished ? 'yes' : 'no'], ['Finish time', fm.makespan ? fm.makespan + ' s' : '-'],
 ['Readings delivered', Math.round(fm.delivered_frac * 100) + '%'], ['Total distance', Math.round(fm.total_distance) + ' m'],
 ['Repeat surveys', fm.repeat_surveys], ['Messages sent', fm.messages], ['Mean connectivity', fm.mean_lambda2.toFixed(2)]];
$('final').innerHTML = rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');

const PAD = 40, S = (cv.width - 2 * PAD) / M.field;
const X = x => PAD + x * S, Y = y => cv.height - PAD - y * S;
let idx = 0, playing = false, acc = 0, last = 0;

function draw() {
  const f = F[idx], W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = css('--line'); ctx.lineWidth = 1; ctx.strokeRect(X(0), Y(M.field), M.field * S, M.field * S);
  const nodes = f.d.map(d => [d[0], d[1]]).concat([M.base]);
  const inBase = new Set(f.bc);
  if ($('c-range').checked) {
    ctx.fillStyle = css('--link'); ctx.globalAlpha = 0.06;
    f.d.forEach(d => { if (d[2]) { ctx.beginPath(); ctx.arc(X(d[0]), Y(d[1]), M.radio_range * S, 0, 7); ctx.fill(); } });
    ctx.globalAlpha = 1;
  }
  for (let j = 0; j < M.tasks.length; j++) {
    const s = f.ts[j], t = M.tasks[j];
    ctx.beginPath(); ctx.arc(X(t[0]), Y(t[1]), 4.5, 0, 7);
    if (s === '0') { ctx.fillStyle = css('--open'); ctx.fill(); }
    else { ctx.fillStyle = s === '1' ? css('--held') : css('--dlv'); ctx.fill(); }
  }
  if ($('c-links').checked) {
    for (const [a, b] of f.l) {
      const ok = inBase.has(a) && inBase.has(b);
      ctx.strokeStyle = ok ? css('--link') : css('--link2'); ctx.lineWidth = ok ? 2 : 1.5;
      ctx.setLineDash(ok ? [] : [5, 4]);
      ctx.beginPath(); ctx.moveTo(X(nodes[a][0]), Y(nodes[a][1])); ctx.lineTo(X(nodes[b][0]), Y(nodes[b][1])); ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  ctx.fillStyle = css('--base'); ctx.fillRect(X(M.base[0]) - 9, Y(M.base[1]) - 9, 18, 18);
  ctx.fillStyle = css('--muted'); ctx.font = '13px system-ui'; ctx.fillText('base', X(M.base[0]) + 13, Y(M.base[1]) - 4);
  f.d.forEach((d, i) => {
    const [x, y, alive, mode, held, tgt] = d;
    if ($('c-tgt').checked && alive && tgt) {
      ctx.strokeStyle = css('--muted'); ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(x), Y(y)); ctx.lineTo(X(tgt[0]), Y(tgt[1])); ctx.stroke(); ctx.globalAlpha = 1;
    }
    const col = !alive ? css('--dead') : (mode === 1 || mode === 2) ? css('--work') : (mode === 3 || mode === 4) ? css('--home') : css('--idle');
    if (!alive) {
      ctx.strokeStyle = col; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(X(x) - 6, Y(y) - 6); ctx.lineTo(X(x) + 6, Y(y) + 6); ctx.moveTo(X(x) + 6, Y(y) - 6); ctx.lineTo(X(x) - 6, Y(y) + 6); ctx.stroke();
      return;
    }
    ctx.beginPath(); ctx.arc(X(x), Y(y), 8, 0, 7); ctx.fillStyle = col; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = css('--bg'); ctx.stroke();
    if (held) { ctx.fillStyle = css('--held'); ctx.beginPath(); ctx.arc(X(x) + 8, Y(y) - 8, 4, 0, 7); ctx.fill(); }
  });
  const delivered = (f.ts.match(/2/g) || []).length, alive = f.d.filter(d => d[2]).length;
  $('s-time').textContent = f.t; $('s-dlv').textContent = `${delivered} / ${M.n_tasks}`;
  $('s-alive').textContent = `${alive} / ${M.n_drones}`;
  $('s-contact').textContent = f.bc.filter(n => n < M.n_drones).length;
  $('s-lam').textContent = f.lam2.toFixed(2);
  $('scrub').value = idx;
}
function loop(ts) {
  const dt = last ? (ts - last) / 1000 : 0; last = ts;
  if (playing) { acc += dt * 10 * Number($('speed').value);
    while (acc >= 1) { acc -= 1; if (idx < F.length - 1) idx++; else { playing = false; $('play').textContent = 'Play'; } }
    draw(); }
  requestAnimationFrame(loop);
}
$('play').onclick = () => { if (idx >= F.length - 1) idx = 0; playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play'; };
$('scrub').oninput = e => { idx = Number(e.target.value); draw(); };
['c-links', 'c-range', 'c-tgt'].forEach(id => $(id).onchange = draw);
draw(); requestAnimationFrame(loop);
</script></body></html>
"""


def build_html(trace):
    return TEMPLATE.replace("__TRACE__", json.dumps(trace, separators=(",", ":")))


def save_html(trace, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_html(trace))
    return path
