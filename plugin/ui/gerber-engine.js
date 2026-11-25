var GEngine = (function () {
"use strict";
const TAU = Math.PI * 2;

/* ---------------- expression evaluator for aperture macros ---------------- */
function evalExpr(src, vars) {
  const s = src.replace(/\s+/g, "").replace(/[xX]/g, "*");
  let i = 0;
  function num() {
    if (s[i] === "$") { i++; let k = ""; while (/[0-9]/.test(s[i])) k += s[i++]; return vars[k] || 0; }
    let k = ""; while (i < s.length && /[0-9.]/.test(s[i])) k += s[i++];
    return k === "" ? 0 : parseFloat(k);
  }
  function factor() {
    if (s[i] === "+") { i++; return factor(); }
    if (s[i] === "-") { i++; return -factor(); }
    if (s[i] === "(") { i++; const v = expr(); if (s[i] === ")") i++; return v; }
    return num();
  }
  function term() { let v = factor(); while (s[i] === "*" || s[i] === "/") { const o = s[i++]; const r = factor(); v = o === "*" ? v * r : v / r; } return v; }
  function expr() { let v = term(); while (s[i] === "+" || s[i] === "-") { const o = s[i++]; const r = term(); v = o === "+" ? v + r : v - r; } return v; }
  return expr();
}

/* ---------------- Gerber ---------------- */
function parseGerber(text, name) {
  const st = { zero: "L", dec: 6, int: 3, unit: 1, apertures: {}, macros: {}, ap: null, x: 0, y: 0,
    interp: 1, quad: 75, region: false, pol: "D", lastD: 2 };
  const prims = [], warnings = new Set();
  let fileFunction = null, contours = [], contour = null, sr = null;

  function coord(v) {
    if (v == null) return null;
    let val;
    if (v.indexOf(".") >= 0) val = parseFloat(v);
    else {
      let sign = 1; if (v[0] === "-" || v[0] === "+") { if (v[0] === "-") sign = -1; v = v.slice(1); }
      if (st.zero === "T") { while (v.length < st.int + st.dec) v += "0"; }
      val = sign * parseInt(v, 10) / Math.pow(10, st.dec);
    }
    return val * st.unit;
  }
  function defineAperture(body) {
    const m = body.match(/^ADD(\d+)([^,]*)(?:,(.*))?$/);
    if (!m) return;
    const code = +m[1], tpl = m[2], params = m[3] ? m[3].split("X").map(parseFloat) : [];
    const u = st.unit;
    let ap;
    if (tpl === "C") ap = { s: "C", d: params[0] * u };
    else if (tpl === "R") ap = { s: "R", w: params[0] * u, h: params[1] * u };
    else if (tpl === "O") ap = { s: "O", w: params[0] * u, h: params[1] * u };
    else if (tpl === "P") ap = { s: "P", d: params[0] * u, n: params[1] | 0, rot: params[2] || 0 };
    else if (st.macros[tpl]) ap = { s: "M", prims: expandMacro(st.macros[tpl], params) };
    else { warnings.add("Неизвестная апертура " + tpl); ap = { s: "C", d: 0 }; }
    ap.r = apRadius(ap);
    st.apertures[code] = ap;
  }
  function expandMacro(stmts, params) {
    const vars = {}; params.forEach((p, k) => (vars[k + 1] = p));
    const out = [], u = st.unit;
    for (const raw of stmts) {
      const t = raw.trim(); if (!t) continue;
      if (t[0] === "0" && (t.length === 1 || t[1] === " " || t[1] === ",")) continue;
      if (t[0] === "$") { const k = t.indexOf("="); if (k > 0) { vars[t.slice(1, k)] = evalExpr(t.slice(k + 1), vars); } continue; }
      const parts = t.split(",").map(p => p.trim());
      const code = parseInt(parts[0], 10);
      const v = parts.slice(1).map(p => evalExpr(p, vars));
      if (code === 1) out.push({ c: 1, e: v[0], d: v[1] * u, x: v[2] * u, y: v[3] * u, rot: v[4] || 0 });
      else if (code === 20 || code === 2) out.push({ c: 20, e: v[0], w: v[1] * u, x1: v[2] * u, y1: v[3] * u, x2: v[4] * u, y2: v[5] * u, rot: v[6] || 0 });
      else if (code === 21) out.push({ c: 21, e: v[0], w: v[1] * u, h: v[2] * u, x: v[3] * u, y: v[4] * u, rot: v[5] || 0 });
      else if (code === 22) out.push({ c: 22, e: v[0], w: v[1] * u, h: v[2] * u, x: v[3] * u, y: v[4] * u, rot: v[5] || 0 });
      else if (code === 4) { const n = v[1] | 0; const pts = []; for (let k = 0; k <= n; k++) pts.push([v[2 + 2 * k] * u, v[3 + 2 * k] * u]); out.push({ c: 4, e: v[0], pts, rot: v[4 + 2 * n] || 0 }); }
      else if (code === 5) out.push({ c: 5, e: v[0], n: v[1] | 0, x: v[2] * u, y: v[3] * u, d: v[4] * u, rot: v[5] || 0 });
      else if (code === 7) out.push({ c: 7, e: 1, x: v[0] * u, y: v[1] * u, od: v[2] * u, id: v[3] * u, gap: v[4] * u, rot: v[5] || 0 });
      else if (code === 6) { out.push({ c: 1, e: 1, d: v[2] * u, x: v[0] * u, y: v[1] * u, rot: 0 }); warnings.add("Муаровый примитив заменён кругом"); }
      else warnings.add("Неизвестный примитив макроса " + code);
    }
    return out;
  }
  function addPrim(p) { p.pol = st.pol; prims.push(p); }
  function arcPoints(x0, y0, x1, y1, I, J, cw) {
    let cx, cy;
    if (st.quad === 74) {
      let best = null;
      for (const [sx, sy] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
        const ccx = x0 + sx * Math.abs(I), ccy = y0 + sy * Math.abs(J);
        const r0 = Math.hypot(x0 - ccx, y0 - ccy), r1 = Math.hypot(x1 - ccx, y1 - ccy);
        let a0 = Math.atan2(y0 - ccy, x0 - ccx), a1 = Math.atan2(y1 - ccy, x1 - ccx);
        let span = cw ? a0 - a1 : a1 - a0; while (span < 0) span += TAU;
        const err = Math.abs(r0 - r1) + (span > Math.PI / 2 + 1e-6 ? 1e3 : 0);
        if (!best || err < best.err) best = { err, ccx, ccy };
      }
      cx = best.ccx; cy = best.ccy;
    } else { cx = x0 + I; cy = y0 + J; }
    const r = Math.hypot(x0 - cx, y0 - cy);
    let a0 = Math.atan2(y0 - cy, x0 - cx), a1 = Math.atan2(y1 - cy, x1 - cx);
    let span;
    if (cw) { span = a0 - a1; while (span <= 1e-9) span += TAU; if (st.quad === 74 && span > Math.PI / 2 + 1e-6) span = Math.min(span, Math.PI / 2); }
    else { span = a1 - a0; while (span <= 1e-9) span += TAU; }
    if (Math.hypot(x1 - x0, y1 - y0) < 1e-9 && st.quad === 74) span = 0;
    const n = Math.max(2, Math.ceil(span / (Math.PI / 48)));
    const pts = [];
    for (let k = 1; k <= n; k++) { const a = cw ? a0 - span * k / n : a0 + span * k / n; pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); }
    pts[pts.length - 1] = [x1, y1];
    return pts;
  }
  function endSR() {
    if (!sr) return;
    const block = prims.slice(sr.start);
    for (let ix = 0; ix < sr.X; ix++) for (let iy = 0; iy < sr.Y; iy++) {
      if (ix === 0 && iy === 0) continue;
      const dx = ix * sr.I, dy = iy * sr.J;
      for (const p of block) prims.push(shiftPrim(p, dx, dy));
    }
    sr = null;
  }
  function word(cmd) {
    if (/^G0?4/.test(cmd)) return;
    if (/^M0?2/.test(cmd)) return;
    const re = /([GDMXYIJ])([+-]?[\d.]+)/g; let m; const w = {}; const gs = [];
    while ((m = re.exec(cmd))) { if (m[1] === "G") gs.push(+m[2]); else w[m[1]] = m[2]; }
    for (const g of gs) {
      if (g === 1 || g === 2 || g === 3) st.interp = g;
      else if (g === 36) { st.region = true; contours = []; contour = null; }
      else if (g === 37) { if (contour && contour.length > 2) contours.push(contour); if (contours.length) addPrim({ t: "R", contours }); st.region = false; contours = []; contour = null; }
      else if (g === 74 || g === 75) st.quad = g;
      else if (g === 70) st.unit = 25.4; else if (g === 71) st.unit = 1;
    }
    let d = w.D != null ? +w.D : null;
    if (d != null && d >= 10) { st.ap = st.apertures[d] || null; if (!st.ap) warnings.add("Апертура D" + d + " не определена"); return; }
    const hasXY = w.X != null || w.Y != null;
    if (d == null) { if (!hasXY) return; d = st.lastD; }
    const nx = w.X != null ? coord(w.X) : st.x, ny = w.Y != null ? coord(w.Y) : st.y;
    const I = w.I != null ? coord(w.I) : 0, J = w.J != null ? coord(w.J) : 0;
    st.lastD = d;
    if (d === 1) {
      const pts = st.interp === 1 ? [[nx, ny]] : arcPoints(st.x, st.y, nx, ny, I, J, st.interp === 2);
      if (st.region) { if (!contour) contour = [[st.x, st.y]]; for (const p of pts) contour.push(p); }
      else if (st.ap) { let px = st.x, py = st.y; for (const p of pts) { addPrim({ t: "L", ap: st.ap, x1: px, y1: py, x2: p[0], y2: p[1] }); px = p[0]; py = p[1]; } }
    } else if (d === 2) {
      if (st.region) { if (contour && contour.length > 2) contours.push(contour); contour = [[nx, ny]]; }
    } else if (d === 3) {
      if (st.ap) addPrim({ t: "F", ap: st.ap, x: nx, y: ny });
    }
    st.x = nx; st.y = ny;
  }
  function extended(body) {
    const cmd = body.slice(0, 2);
    if (cmd === "FS") {
      const m = body.match(/^FS([LTD]?)([AI]?)X(\d)(\d)Y(\d)(\d)/);
      if (m) { st.zero = m[1] === "T" ? "T" : "L"; st.int = +m[3]; st.dec = +m[4]; if (m[2] === "I") warnings.add("Инкрементные координаты не поддерживаются"); }
    } else if (cmd === "MO") st.unit = body.slice(2, 4) === "IN" ? 25.4 : 1;
    else if (cmd === "AD") defineAperture(body.replace(/\*$/, ""));
    else if (cmd === "LP") st.pol = body[2] === "C" ? "C" : "D";
    else if (cmd === "SR") {
      endSR();
      const m = body.match(/X(\d+)Y(\d+)I([\d.+-]+)J([\d.+-]+)/);
      if (m && (+m[1] > 1 || +m[2] > 1)) sr = { X: +m[1], Y: +m[2], I: parseFloat(m[3]) * st.unit, J: parseFloat(m[4]) * st.unit, start: prims.length };
    } else if (cmd === "TF") {
      const m = body.match(/^TF\.FileFunction,([^*]*)/); if (m) fileFunction = m[1];
    }
  }
  // tokenizer
  let i = 0; const n = text.length;
  while (i < n) {
    const ch = text[i];
    if (ch === "%") {
      const j = text.indexOf("%", i + 1); if (j < 0) break;
      const block = text.slice(i + 1, j); i = j + 1;
      if (block.replace(/\s/g, "").startsWith("AM")) {
        const stmts = block.split("*").map(s => s.replace(/[\r\n]/g, ""));
        const nm = stmts[0].trim().slice(2);
        st.macros[nm] = stmts.slice(1).filter(s => s.trim().length);
      } else {
        for (const part of block.split("*")) { const b = part.replace(/[\r\n\s]/g, ""); if (b) extended(b); }
      }
    } else if (ch === "\r" || ch === "\n" || ch === " " || ch === "\t") i++;
    else {
      const j = text.indexOf("*", i); if (j < 0) break;
      const cmd = text.slice(i, j).replace(/[\r\n]/g, ""); i = j + 1;
      word(cmd.trim());
    }
  }
  endSR();
  for (const p of prims) p.bb = primBBox(p);
  return { name, prims, fileFunction, warnings: [...warnings] };
}
function shiftPrim(p, dx, dy) {
  if (p.t === "F") return { t: "F", ap: p.ap, x: p.x + dx, y: p.y + dy, pol: p.pol };
  if (p.t === "L") return { t: "L", ap: p.ap, x1: p.x1 + dx, y1: p.y1 + dy, x2: p.x2 + dx, y2: p.y2 + dy, pol: p.pol };
  return { t: "R", contours: p.contours.map(c => c.map(q => [q[0] + dx, q[1] + dy])), pol: p.pol };
}
function apRadius(ap) {
  if (ap.s === "C") return ap.d / 2;
  if (ap.s === "R" || ap.s === "O") return Math.hypot(ap.w, ap.h) / 2;
  if (ap.s === "P") return ap.d / 2;
  let r = 0;
  for (const q of ap.prims) {
    if (q.c === 1) r = Math.max(r, Math.hypot(q.x, q.y) + q.d / 2);
    else if (q.c === 20) r = Math.max(r, Math.hypot(q.x1, q.y1) + q.w, Math.hypot(q.x2, q.y2) + q.w);
    else if (q.c === 21 || q.c === 22) r = Math.max(r, Math.hypot(Math.abs(q.x) + q.w, Math.abs(q.y) + q.h));
    else if (q.c === 4) for (const p of q.pts) r = Math.max(r, Math.hypot(p[0], p[1]));
    else if (q.c === 5) r = Math.max(r, Math.hypot(q.x, q.y) + q.d / 2);
    else if (q.c === 7) r = Math.max(r, Math.hypot(q.x, q.y) + q.od / 2);
  }
  return r;
}
function primBBox(p) {
  if (p.t === "F") { const r = p.ap.r; return [p.x - r, p.y - r, p.x + r, p.y + r]; }
  if (p.t === "L") { const r = p.ap.r; return [Math.min(p.x1, p.x2) - r, Math.min(p.y1, p.y2) - r, Math.max(p.x1, p.x2) + r, Math.max(p.y1, p.y2) + r]; }
  let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity;
  for (const ct of p.contours) for (const q of ct) { if (q[0] < a) a = q[0]; if (q[1] < b) b = q[1]; if (q[0] > c) c = q[0]; if (q[1] > d) d = q[1]; }
  return [a, b, c, d];
}

/* ---------------- Excellon ---------------- */
function parseExcellon(text, name) {
  const lines = text.split(/\r?\n/);
  let unit = 1, zeros = "TZ", fmtI = 3, fmtD = 3, explicitFmt = false, tool = null, header = true, plated = null;
  const tools = {}, holes = [], warnings = new Set(), npthTools = new Set(); let typeNP = false;
  const fileSpan=text.match(/TF\.FileFunction,(?:Plated|NonPlated),(\d+),(\d+)/i);
  const span=fileSpan ? [+fileSpan[1],+fileSpan[2]].sort((a,b)=>a-b) : null;
  let x = 0, y = 0;
  function num(v) {
    if (v.indexOf(".") >= 0) return parseFloat(v) * unit;
    let s = 1; if (v[0] === "-" || v[0] === "+") { if (v[0] === "-") s = -1; v = v.slice(1); }
    if (zeros === "LZ") { while (v.length < fmtI + fmtD) v += "0"; }
    return s * parseInt(v, 10) / Math.pow(10, fmtD) * unit;
  }
  for (let raw of lines) {
    let l = raw.trim(); if (!l) continue;
    if (l[0] === ";") {
      const f = l.match(/FILE_FORMAT\s*=\s*(\d+):(\d+)/i); if (f) { fmtI = +f[1]; fmtD = +f[2]; explicitFmt = true; }
      if (/TYPE\s*=\s*NON_PLATED|TA\.AperFunction,NonPlated/i.test(l)) { typeNP = true; continue; }
      if (/TYPE\s*=\s*PLATED|TA\.AperFunction,Plated/i.test(l)) { typeNP = false; continue; }
      if (/TA\.AperFunction,NonPlated/i.test(l) || /NPTH/i.test(l)) plated = plated === true ? true : false;
      if (/TA\.AperFunction,Plated/i.test(l) || /\bPTH\b/.test(l)) plated = true;
      continue;
    }
    if (/^(METRIC|M71)/.test(l)) { unit = 1; if (!explicitFmt) { fmtI = 3; fmtD = 3; } }
    if (/^(INCH|M72)/.test(l)) { unit = 25.4; if (!explicitFmt) { fmtI = 2; fmtD = 4; } }
    if (/,LZ/.test(l) || l === "LZ") zeros = "LZ";
    if (/,TZ/.test(l) || l === "TZ") zeros = "TZ";
    if (l === "%" || l === "M95") { header = false; continue; }
    const td = l.match(/^T(\d+)(?:[^C]*?)C([\d.]+)/);
    if (td) { tools[+td[1]] = parseFloat(td[2]) * unit; if (typeNP) npthTools.add(+td[1]); if (!header) tool = +td[1]; continue; }
    const ts = l.match(/^T(\d+)\s*$/);
    if (ts) { tool = +ts[1]; continue; }
    if (/^(M30|M00)/.test(l)) break;
    if (/G85/.test(l)) warnings.add("Пазы (G85) учтены как отверстия в начальной точке");
    const m = l.match(/^(?:G0?[015])?X?([+-]?[\d.]+)?/);
    if (/[XY]/.test(l) && !/^G0?0/.test(l)) {
      const mx = l.match(/X([+-]?[\d.]+)/), my = l.match(/Y([+-]?[\d.]+)/);
      if (mx) x = num(mx[1]); if (my) y = num(my[1]);
      if (tool != null && tools[tool] != null && !npthTools.has(tool)) holes.push({ x, y, d: tools[tool], span });
    }
  }
  return { name, holes, plated, warnings: [...warnings] };
}

/* ---------------- file role detection ---------------- */
function detectRole(name, text) {
  const n = name.toLowerCase();
  if (/^\s*(m48|;)/i.test(text.slice(0, 200)) || /\b(m48)\b/i.test(text.slice(0, 2000)) && !/%FS/.test(text.slice(0, 2000))) {
    const header=text.slice(0,16000);
    if (/TF\.FileFunction,NonPlated/i.test(header) || /npth|non.?plated/i.test(n)) return "npth";
    if (!/TA\.AperFunction,Plated|TF\.FileFunction,Plated/i.test(header) && /TA\.AperFunction,NonPlated/i.test(header)) return "npth";
    return "drill";
  }
  const ff = (text.match(/%TF\.FileFunction,([^*]*)\*%/) || [])[1];
  if (ff) {
    if (/^Copper/i.test(ff)) { const m = ff.match(/,L(\d+),(Top|Bot|Inr)/i); return "copper:" + (m ? (m[2].toLowerCase() === "top" ? "Top" : m[2].toLowerCase() === "bot" ? "Bottom" : "In" + (m[1] - 1)) : "?"); }
    return "ignore";
  }
  if (/\.(gtl|cmp|top)$/.test(n) || /f[._-]cu/.test(n) || /top.?copper|copper.?top/.test(n)) return "copper:Top";
  if (/\.(gbl|sol|bot)$/.test(n) || /b[._-]cu/.test(n) || /bottom.?copper|copper.?bot/.test(n)) return "copper:Bottom";
  const inner = n.match(/\.g(\d+)$|in(\d+)[._-]cu|\.gp(\d+)$|\.ly(\d+)$/);
  if (inner) return "copper:In" + (inner[1] || inner[2] || inner[3] || inner[4]);
  if (/\.(drl|xln|exc|drd|nc|tap)$/.test(n)) return /npth/.test(n) ? "npth" : "drill";
  if (!/(mask|silk|paste|legend|outline|overlay|stencil|fab|assembly|courtyard|keepout|profile|drill|edge|mech)/.test(n)) {
    if (/(^|[^a-z])top([^a-z]|$)/.test(n)) return "copper:Top";
    if (/(^|[^a-z])(bottom|bot)([^a-z]|$)/.test(n)) return "copper:Bottom";
  }
  if (/%FS/.test(text.slice(0, 3000)) && /copper|cu\b/.test(n)) return "copper:?";
  return "ignore";
}

/* ---------------- rendering ---------------- */
function polyPath(ctx, pts) { ctx.moveTo(pts[0][0], pts[0][1]); for (let k = 1; k < pts.length; k++) ctx.lineTo(pts[k][0], pts[k][1]); ctx.closePath(); }
function rot(p, a) { if (!a) return p; const c = Math.cos(a * Math.PI / 180), s = Math.sin(a * Math.PI / 180); return [p[0] * c - p[1] * s, p[0] * s + p[1] * c]; }
function circlePts(cx, cy, r, n) { const o = []; for (let k = 0; k < n; k++) { const a = TAU * k / n; o.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); } return o; }
function drawAperture(ctx, ap, x, y, clear) {
  const fillPoly = pts => { ctx.beginPath(); polyPath(ctx, pts.map(p => [p[0] + x, p[1] + y])); ctx.fill(); };
  if (ap.s === "C") { ctx.beginPath(); ctx.arc(x, y, ap.d / 2, 0, TAU); ctx.fill(); return; }
  if (ap.s === "R") { ctx.fillRect(x - ap.w / 2, y - ap.h / 2, ap.w, ap.h); return; }
  if (ap.s === "O") { fillPoly(obroundPts(ap.w, ap.h)); return; }
  if (ap.s === "P") { const o = []; for (let k = 0; k < ap.n; k++) { const a = (ap.rot * Math.PI / 180) + TAU * k / ap.n; o.push([ap.d / 2 * Math.cos(a), ap.d / 2 * Math.sin(a)]); } fillPoly(o); return; }
  // macro
  for (const q of ap.prims) {
    const off = q.e === 0 || clear;
    ctx.globalCompositeOperation = off ? "destination-out" : "source-over";
    let polys = [];
    if (q.c === 1) polys = [circlePts(q.x, q.y, q.d / 2, 48).map(p => rot(p, q.rot))];
    else if (q.c === 20) { const dx = q.x2 - q.x1, dy = q.y2 - q.y1, L = Math.hypot(dx, dy) || 1, nx = -dy / L * q.w / 2, ny = dx / L * q.w / 2;
      polys = [[[q.x1 + nx, q.y1 + ny], [q.x2 + nx, q.y2 + ny], [q.x2 - nx, q.y2 - ny], [q.x1 - nx, q.y1 - ny]].map(p => rot(p, q.rot))]; }
    else if (q.c === 21) polys = [[[q.x - q.w / 2, q.y - q.h / 2], [q.x + q.w / 2, q.y - q.h / 2], [q.x + q.w / 2, q.y + q.h / 2], [q.x - q.w / 2, q.y + q.h / 2]].map(p => rot(p, q.rot))];
    else if (q.c === 22) polys = [[[q.x, q.y], [q.x + q.w, q.y], [q.x + q.w, q.y + q.h], [q.x, q.y + q.h]].map(p => rot(p, q.rot))];
    else if (q.c === 4) polys = [q.pts.map(p => rot(p, q.rot))];
    else if (q.c === 5) { const o = []; for (let k = 0; k < q.n; k++) { const a = TAU * k / q.n; o.push([q.x + q.d / 2 * Math.cos(a), q.y + q.d / 2 * Math.sin(a)]); } polys = [o.map(p => rot(p, q.rot))]; }
    else if (q.c === 7) polys = thermalPolys(q);
    for (const pl of polys) { ctx.beginPath(); polyPath(ctx, pl.map(p => [p[0] + x, p[1] + y])); ctx.fill(); }
  }
  ctx.globalCompositeOperation = clear ? "destination-out" : "source-over";
}
function obroundPts(w, h) {
  const r = Math.min(w, h) / 2, o = [];
  if (w >= h) { const c = w / 2 - r; for (let k = 0; k <= 24; k++) { const a = -Math.PI / 2 + Math.PI * k / 24; o.push([c + r * Math.cos(a), r * Math.sin(a)]); } for (let k = 0; k <= 24; k++) { const a = Math.PI / 2 + Math.PI * k / 24; o.push([-c + r * Math.cos(a), r * Math.sin(a)]); } }
  else { const c = h / 2 - r; for (let k = 0; k <= 24; k++) { const a = Math.PI * k / 24; o.push([r * Math.cos(a), c + r * Math.sin(a)]); } for (let k = 0; k <= 24; k++) { const a = Math.PI + Math.PI * k / 24; o.push([r * Math.cos(a), -c + r * Math.sin(a)]); } }
  return o;
}
function thermalPolys(q) {
  const ro = q.od / 2, ri = q.id / 2, g = q.gap / 2, out = [];
  for (let s = 0; s < 4; s++) {
    const base = s * Math.PI / 2;
    const ao = Math.asin(Math.min(1, g / ro)), ai = Math.asin(Math.min(1, g / Math.max(ri, 1e-9)));
    const pts = [];
    for (let k = 0; k <= 12; k++) { const a = base + ao + (Math.PI / 2 - 2 * ao) * k / 12; pts.push([q.x + ro * Math.cos(a), q.y + ro * Math.sin(a)]); }
    for (let k = 12; k >= 0; k--) { const a = base + ai + (Math.PI / 2 - 2 * ai) * k / 12; pts.push([q.x + ri * Math.cos(a), q.y + ri * Math.sin(a)]); }
    out.push(pts.map(p => rot([p[0] - q.x, p[1] - q.y], q.rot)).map(p => [p[0] + q.x, p[1] + q.y]));
  }
  return out;
}
function renderPrim(ctx, p, scale) {
  const clear = p.pol === "C";
  ctx.globalCompositeOperation = clear ? "destination-out" : "source-over";
  if (p.t === "F") drawAperture(ctx, p.ap, p.x, p.y, clear);
  else if (p.t === "L") {
    const ap = p.ap, L = Math.hypot(p.x2 - p.x1, p.y2 - p.y1);
    if (ap.s === "C") {
      if (L < 1e-9) { ctx.beginPath(); ctx.arc(p.x1, p.y1, ap.d / 2, 0, TAU); ctx.fill(); }
      else { ctx.lineWidth = Math.max(ap.d, 0.5 / scale); ctx.lineCap = "round"; ctx.beginPath(); ctx.moveTo(p.x1, p.y1); ctx.lineTo(p.x2, p.y2); ctx.stroke(); }
    } else if (ap.s === "R") {
      const hw = ap.w / 2, hh = ap.h / 2, pts = [];
      for (const [cx, cy] of [[p.x1, p.y1], [p.x2, p.y2]]) for (const [sx, sy] of [[1, 1], [1, -1], [-1, -1], [-1, 1]]) pts.push([cx + sx * hw, cy + sy * hh]);
      const h = hull(pts); ctx.beginPath(); polyPath(ctx, h); ctx.fill();
    } else {
      const d = ap.s === "M" ? 2 * ap.r : Math.min(ap.w || ap.d, ap.h || ap.d);
      ctx.lineWidth = d; ctx.lineCap = "round"; ctx.beginPath(); ctx.moveTo(p.x1, p.y1); ctx.lineTo(p.x2, p.y2); ctx.stroke();
    }
  } else {
    for (const c of p.contours) { ctx.beginPath(); polyPath(ctx, c); ctx.fill(); }
  }
  ctx.globalCompositeOperation = "source-over";
}
function segDist(px, py, ax, ay, bx, by) { const dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy; let t = L2 ? ((px - ax) * dx + (py - ay) * dy) / L2 : 0; t = Math.max(0, Math.min(1, t)); return Math.hypot(px - ax - t * dx, py - ay - t * dy); }
function hull(pts) {
  pts = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const p of pts) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
  for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); }
  up.pop(); lo.pop(); return lo.concat(up);
}

/* ---------------- spatial index ---------------- */
function buildIndex(prims, cell) {
  const map = new Map();
  prims.forEach((p, k) => {
    const [a, b, c, d] = p.bb;
    const i0 = Math.floor(a / cell), i1 = Math.floor(c / cell), j0 = Math.floor(b / cell), j1 = Math.floor(d / cell);
    if ((i1 - i0 + 1) * (j1 - j0 + 1) > 4000) { (map.get("big") || map.set("big", []).get("big")).push(k); return; }
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) { const key = i + "," + j; let arr = map.get(key); if (!arr) map.set(key, (arr = [])); arr.push(k); }
  });
  return { cell, map };
}
function query(idx, prims, x0, y0, x1, y1) {
  const out = new Set(idx.map.get("big") || []);
  const c = idx.cell;
  for (let i = Math.floor(x0 / c); i <= Math.floor(x1 / c); i++) for (let j = Math.floor(y0 / c); j <= Math.floor(y1 / c); j++) { const arr = idx.map.get(i + "," + j); if (arr) for (const k of arr) out.add(k); }
  return [...out].sort((a, b) => a - b).filter(k => { const bb = prims[k].bb; return bb[2] >= x0 && bb[0] <= x1 && bb[3] >= y0 && bb[1] <= y1; });
}

/* ---------------- EDT (Felzenszwalb) ---------------- */
function edt1(f, n, d, v, z) {
  let k = 0; v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) { k--; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
    k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]; }
}
function edt(mask, W, H) { // distance (px) from copper pixel centre to nearest non-copper pixel centre
  const INF = 1e20, g = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) g[i] = mask[i] ? INF : 0;
  const N = Math.max(W, H), f = new Float64Array(N), d = new Float64Array(N), v = new Int32Array(N), z = new Float64Array(N + 1);
  for (let x = 0; x < W; x++) { for (let y = 0; y < H; y++) f[y] = g[y * W + x]; edt1(f, H, d, v, z); for (let y = 0; y < H; y++) g[y * W + x] = d[y]; }
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) { for (let x = 0; x < W; x++) f[x] = g[y * W + x]; edt1(f, W, d, v, z); for (let x = 0; x < W; x++) out[y * W + x] = Math.sqrt(d[x]); }
  return out;
}

/* ---------------- per-pad analysis ---------------- */
const CLASS_K = { 3: 0.2, 2: 0.2, 1: 0.3 };

function rasterWindow(makeCanvas, prims, ids, cx, cy, H, N, filter) {
  const cv = makeCanvas(N, N), ctx = cv.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalCompositeOperation = "source-over"; ctx.clearRect(0, 0, N, N);
  const s = N / (2 * H);
  ctx.setTransform(s, 0, 0, -s, (H - cx) * s, (H + cy) * s);
  ctx.fillStyle = "#000"; ctx.strokeStyle = "#000";
  for (const k of ids) { const p = prims[k]; if (filter && !filter(p)) continue; renderPrim(ctx, p, s); }
  const data = ctx.getImageData(0, 0, N, N).data, mask = new Uint8Array(N * N);
  for (let i = 0; i < N * N; i++) mask[i] = data[4 * i + 3] > 127 ? 1 : 0;
  return mask;
}

function analyzeLayer(makeCanvas, layer, holes, opt, progress) {
  const prims = layer.prims, idx = layer.idx || (layer.idx = buildIndex(prims, 2));
  const pads = [];
  const Nmax = opt.res || 400, pxTarget = opt.px || 0.005;
  for (let h = 0; h < holes.length; h++) {
    const hole = holes[h], Rd = hole.d / 2;
    if (hole.span && layer.number && (layer.number < hole.span[0] || layer.number > hole.span[1])) continue;
    // pad flash at the hole
    const near = query(idx, prims, hole.x - 0.05, hole.y - 0.05, hole.x + 0.05, hole.y + 0.05);
    let flash = null;
    for (const k of near) { const p = prims[k]; if (p.t === "F" && p.pol === "D" && Math.hypot(p.x - hole.x, p.y - hole.y) < 0.03) { if (!flash || p.ap.r > flash.ap.r) flash = p; } }
    let padStrokes = [];
    if (!flash) {
      for (const k of near.length ? query(idx, prims, hole.x - 3, hole.y - 3, hole.x + 3, hole.y + 3) : []) {
        const p = prims[k]; if (p.t !== "L" || p.pol !== "D" || p.ap.s !== "C") continue;
        const w = p.ap.d, d1 = Math.hypot(p.x1 - hole.x, p.y1 - hole.y), d2 = Math.hypot(p.x2 - hole.x, p.y2 - hole.y);
        if (w > 1.3 * Rd * 2 * 0.5 && d1 <= w * 1.01 && d2 <= w * 1.01 && segDist(hole.x, hole.y, p.x1, p.y1, p.x2, p.y2) < 0.03) padStrokes.push(p);
      }
    }
    const ext = flash ? flash.ap.r : padStrokes.length ? Math.max(...padStrokes.map(p => Math.max(Math.hypot(p.x1 - hole.x, p.y1 - hole.y), Math.hypot(p.x2 - hole.x, p.y2 - hole.y)) + p.ap.d / 2)) : Rd * 2;
    const Hw = Math.max(ext, Rd) + Rd + opt.margin;
    const N = Math.max(140, Math.min(Nmax, Math.round(2 * Hw / pxTarget)));
    const ids = query(idx, prims, hole.x - Hw, hole.y - Hw, hole.x + Hw, hole.y + Hw);
    const mask = rasterWindow(makeCanvas, prims, ids, hole.x, hole.y, Hw, N);
    const px = 2 * Hw / N, c0 = (N - 1) / 2;
    const ic = Math.round(c0) * N + Math.round(c0);
    if (!mask[ic]) { if (progress) progress(h, holes.length); continue; } // no copper at the hole on this layer
    const E = edt(mask, N, N);
    for (let i = 0; i < E.length; i++) E[i] = E[i] * px; // empirically unbiased with bilinear lookup (|err| < 1 um off-peak)
    const lookup = (x, y) => { // x,y mm relative to hole centre
      const fx = c0 + x / px, fy = c0 - y / px; if (fx < 0 || fy < 0 || fx > N - 1 || fy > N - 1) return -1;
      const x0 = Math.floor(fx), y0 = Math.floor(fy), x1 = Math.min(x0 + 1, N - 1), y1 = Math.min(y0 + 1, N - 1), tx = fx - x0, ty = fy - y0;
      return (E[y0 * N + x0] * (1 - tx) + E[y0 * N + x1] * tx) * (1 - ty) + (E[y1 * N + x0] * (1 - tx) + E[y1 * N + x1] * tx) * ty;
    };
    const ePlane = lookup(0, 0);
    if (ePlane >= Hw - Rd * 0.2) { pads.push({ hole: h, x: hole.x, y: hole.y, Rd, plane: true }); if (progress) progress(h, holes.length); continue; }
    // flash-only mask for pad outline
    let flashMask = null, padInsc;
    if (flash) {
      flashMask = rasterWindow(makeCanvas, [flash], [0], hole.x, hole.y, Hw, N);
      const ap = flash.ap;
      if (ap.s === "C") padInsc = ap.d / 2;
      else if (ap.s === "R" || ap.s === "O") padInsc = Math.min(ap.w, ap.h) / 2;
      else if (ap.s === "P") padInsc = ap.d / 2 * Math.cos(Math.PI / ap.n);
      else { const Ef = edt(flashMask, N, N); let mx = 0; for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) mx = Math.max(mx, Ef[(Math.round(c0) + dy) * N + Math.round(c0) + dx]); padInsc = mx * px; }
    }
    else if (padStrokes.length) {
      flashMask = rasterWindow(makeCanvas, padStrokes, padStrokes.map((_, k) => k), hole.x, hole.y, Hw, N);
      const Ef = edt(flashMask, N, N); let mx = 0; for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) mx = Math.max(mx, Ef[(Math.round(c0) + dy) * N + Math.round(c0) + dx]); padInsc = mx * px;
    }
    else padInsc = ePlane;
    // traces attached to the pad
    const traces = [];
    for (const k of ids) {
      const p = prims[k]; if (p.t !== "L" || p.pol !== "D" || p.ap.s !== "C" || padStrokes.includes(p)) continue;
      const w = p.ap.d; if (w >= 1.9 * padInsc) continue; // as wide as the pad: part of the land, not a conductor
      const dx = p.x2 - p.x1, dy = p.y2 - p.y1, L2 = dx * dx + dy * dy; if (L2 < 1e-12) continue;
      let t = ((hole.x - p.x1) * dx + (hole.y - p.y1) * dy) / L2; t = Math.max(0, Math.min(1, t));
      const qx = p.x1 + t * dx - hole.x, qy = p.y1 + t * dy - hole.y;
      if (Math.hypot(qx, qy) > Math.max(padInsc, Rd * 0.5) + 1e-6) continue;
      for (const [ex, ey] of [[p.x1 - hole.x, p.y1 - hole.y], [p.x2 - hole.x, p.y2 - hole.y]]) {
        const L = Math.hypot(ex - qx, ey - qy); if (L < Math.max(padInsc, Rd) * 0.6) continue;
        const ux = (ex - qx) / L, uy = (ey - qy) / L;
        if (traces.some(o => Math.abs(o.ux * ux + o.uy * uy) > 0.9986 && o.ux * ux + o.uy * uy > 0 && Math.abs((qx - o.qx) * uy - (qy - o.qy) * ux) < w / 4)) continue;
        traces.push({ w, qx, qy, ux, uy, len: L });
      }
    }
    // sections for each trace, starting where the axis leaves the pad flash
    const isCu = (m, x, y) => { const ix = Math.round(c0 + x / px), iy = Math.round(c0 - y / px); return ix >= 0 && iy >= 0 && ix < N && iy < N ? m[iy * N + ix] : 0; };
    for (const tr of traces) {
      const vx = -tr.uy, vy = tr.ux; let t0 = 0;
      const fm = flashMask || mask;
      while (t0 < Hw && isCu(fm, tr.qx + t0 * tr.ux, tr.qy + t0 * tr.uy)) t0 += px;
      const tEnd = Math.min(Hw - px * 2, t0 + 2.2 * Rd + 0.1, tr.len);
      const ts = [], as = [], bs = [];
      for (let t = t0; t <= tEnd; t += px) {
        const x = tr.qx + t * tr.ux, y = tr.qy + t * tr.uy;
        if (!isCu(mask, x, y)) { ts.push(t); as.push(0); bs.push(0); continue; }
        let a = 0; while (a < Hw && isCu(mask, x - (a + px) * vx, y - (a + px) * vy)) a += px;
        let b = 0; while (b < Hw && isCu(mask, x + (b + px) * vx, y + (b + px) * vy)) b += px;
        ts.push(t); as.push(-a - px / 2); bs.push(b + px / 2);
      }
      tr.vx = vx; tr.vy = vy; tr.t0 = t0; tr.ts = Float32Array.from(ts); tr.as = Float32Array.from(as); tr.bs = Float32Array.from(bs);
    }
    // teardrop: a dark region that does not cover the hole centre but covers the trace axis just outside the pad
    const inPoly = (pts, x, y) => { let c = false; for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) { const [xi, yi] = pts[i], [xj, yj] = pts[j]; if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c; } return c; };
    const inRegion = (p, x, y) => p.contours.some(ct => inPoly(ct, x, y));
    for (const k of ids) {
      const p = prims[k]; if (p.t !== "R" || p.pol !== "D") continue;
      if (inRegion(p, hole.x, hole.y)) continue;
      for (const tr of traces) {
        const t = tr.t0 + Math.max(2 * px, 0.01);
        if (inRegion(p, hole.x + tr.qx + t * tr.ux, hole.y + tr.qy + t * tr.uy)) tr.tear = true;
      }
    }
    const tear=traces.some(tr=>tr.tear);
    pads.push({ hole: h, x: hole.x, y: hole.y, Rd, padInsc, flashR: flash ? flash.ap.r : null, flashShape: flash ? flash.ap.s : (padStrokes.length ? "stroke" : null), traces, tear,
      E, N, px, c0, Hw, lookup, mask });
    if (progress) progress(h, holes.length);
  }
  return pads;
}

/* acceptance test for a hole offset (ox, oy) on a pad (mm) */
function makeTester(pad, crit) {
  const { Rd } = pad, e = crit.erosion || 0, cls = crit.cls;
  const kred = crit.kred != null ? crit.kred : CLASS_K[cls];
  const ringReq = crit.ringReq;
  const NS = 72, cs = new Float64Array(NS), sn = new Float64Array(NS);
  for (let k = 0; k < NS; k++) { cs[k] = Math.cos(TAU * k / NS); sn[k] = Math.sin(TAU * k / NS); }
  const brLimit = crit.breakout ?? (cls === 2 ? 90 : 180);
  return function (ox, oy) {
    const ec = pad.lookup(ox, oy); if (ec < 0) return false;
    // ring / breakout
    if (cls === 3) { if (ec - e - Rd < ringReq - 1e-9) return false; }
    else if (ec - e < Rd) {
      let out = 0; for (let k = 0; k < NS; k++) { const v = pad.lookup(ox + Rd * cs[k], oy + Rd * sn[k]); if (v <= e) out++; }
      if (out * 360 / NS > brLimit + 1e-9) return false;
    }
    if (crit.junction === "none") return true;
    for (const tr of pad.traces) {
      const wReq = (1 - kred) * tr.w;
      const rx = ox - tr.qx, ry = oy - tr.qy, cu = rx * tr.ux + ry * tr.uy, cv = rx * tr.vx + ry * tr.vy;
      if (crit.junction === "author") {
        if (tr.tear) continue; // Этот упрощённый критерий относится только к данному соединению.
        if (Math.hypot(tr.qx, tr.qy) > 0.25 * pad.padInsc) continue; // author's geometry assumes a trace aimed at the pad centre
        const R = pad.padInsc, x1 = tr.w / 2; if (R <= x1) continue;
        const y1 = Math.sqrt(R * R - x1 * x1), a = wReq / 2 + Rd;
        if (cu < y1) { if (Math.hypot(cu - y1, cv - x1) + Math.hypot(cu - y1, cv + x1) < 2 * a) return false; }
        else if (Math.abs(cv) < a) return false;
        continue;
      }
      if (cu + Rd < tr.t0) continue;
      const ts = tr.ts;
      for (let j = 0; j < ts.length; j++) {
        const dt = ts[j] - cu; if (dt <= -Rd || dt >= Rd) continue;
        const half = Math.sqrt(Rd * Rd - dt * dt);
        const a = tr.as[j] + e, b = tr.bs[j] - e;
        const ov = Math.max(0, Math.min(b, cv + half) - Math.max(a, cv - half));
        if ((b - a) - ov < wReq - 1e-9) return false;
      }
    }
    return true;
  };
}

/* offset acceptance map for one pad */
function buildMap(pad, crit, Rmax, G) {
  const test = makeTester(pad, crit);
  const step = 2 * Rmax / (G - 1), map = new Uint8Array(G * G);
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
    const ox = -Rmax + i * step, oy = -Rmax + j * step;
    if (ox * ox + oy * oy > Rmax * Rmax) continue;
    map[j * G + i] = test(ox, oy) ? 1 : 0;
  }
  return { map, G, step, Rmax };
}
function mapOk(M, ox, oy) {
  const i = Math.round((ox + M.Rmax) / M.step), j = Math.round((oy + M.Rmax) / M.step);
  if (i < 0 || j < 0 || i >= M.G || j >= M.G) return 0;
  return M.map[j * M.G + i];
}
/* bivariate normal probability over the map's accepted cells */
function mapProb(M, mx, my, sxx, syy, sxy) {
  if (sxx + syy === 0) return mapOk(M, mx, my);
  const det = sxx * syy - sxy * sxy;
  if (det <= Number.EPSILON * (sxx + syy) ** 2) {
    // При одном источнике смещения (например, только поворот) нормальный закон лежит на прямой.
    const vx = Math.sqrt(sxx), vy = vx > 0 ? sxy / vx : Math.sqrt(syy);
    const edges = [-9, 9];
    for (const [mean, direction] of [[mx, vx], [my, vy]]) {
      if (!direction) continue;
      for (let i=0; i<=M.G; ++i) {
        const t=(-M.Rmax+(i-0.5)*M.step-mean)/direction;
        if(t>-9 && t<9) edges.push(t);
      }
    }
    edges.sort((a,b)=>a-b);
    let probability=0;
    for(let i=1; i<edges.length; ++i){
      const mid=(edges[i-1]+edges[i])/2;
      if(mapOk(M,mx+vx*mid,my+vy*mid)) probability+=Phi(edges[i])-Phi(edges[i-1]);
    }
    return Math.min(1, probability);
  }
  // При очень малом СКО масса может целиком лежать между узлами растра.
  if (sxy === 0 && Math.min(sxx, syy) < M.step*M.step) {
    function weights(mean, sigma) {
      const edges = Array.from({length:M.G+1}, (_,i) => Phi((-M.Rmax+(i-0.5)*M.step-mean)/sigma));
      return edges.slice(1).map((v,i) => v-edges[i]);
    }
    const wx=weights(mx,Math.sqrt(sxx)), wy=weights(my,Math.sqrt(syy));
    let probability=0;
    for(let j=0; j<M.G; ++j) for(let i=0; i<M.G; ++i) {
      if(M.map[j*M.G+i]) probability += wx[i]*wy[j];
    }
    return Math.min(1,probability);
  }
  const ixx = syy / det, iyy = sxx / det, ixy = -sxy / det, norm = M.step * M.step / (TAU * Math.sqrt(det));
  let P = 0;
  for (let j = 0; j < M.G; j++) { const oy = -M.Rmax + j * M.step - my; for (let i = 0; i < M.G; i++) { if (!M.map[j * M.G + i]) continue; const ox = -M.Rmax + i * M.step - mx; P += Math.exp(-0.5 * (ixx * ox * ox + 2 * ixy * ox * oy + iyy * oy * oy)); } }
  return Math.min(1, P * norm);
}

/* Допустимое смещение по вписанной окружности и допуск ширины проводника. */
function erf(x) { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); }
const Phi = z => (1 + erf(z / Math.SQRT2)) / 2;
function Proute(w, h, cls, xi, F, loss, gain) {
  const k = { 3: [0.2, 0.2], 2: [0.3, 0.2], 1: [0.3, 0.3] }[cls];
  const s = Math.sqrt(2 * (xi * xi + (h / F) * (h / F)));
  return Phi((gain ?? k[0]) * w / s) - Phi(-(loss ?? k[1]) * w / s);
}
function RCof(cls, Rv, Rd, ring=0.05, breakout=cls===2 ? 90 : 180) {
  if (cls === 3) return Rv - Rd - ring;
  const projection = Rd * Math.cos(breakout * Math.PI / 360);
  return Math.sqrt(Math.max(Rv*Rv - Rd*Rd + projection*projection, 0)) - projection;
}
function circularPad(pad, cls, wminLayer, sigma, crit) {
  const D = 2 * pad.padInsc;
  const wmE = wminLayer * (1 - crit.kred);
  let R = RCof(cls, D / 2, pad.Rd, crit.ringReq, crit.breakout);
  for (const tr of pad.traces) {
    if (tr.tear) continue;
    const y1 = Math.sqrt(Math.max((D / 2) ** 2 - tr.w * tr.w / 4, 0));
    const b2 = (2 * pad.Rd + wmE) ** 2 - tr.w * tr.w;
    if (b2 > 0) R = Math.min(R, y1 - Math.sqrt(b2) / 2);
  }
  const p = R <= 0 ? 0 : 1 - Math.exp(-R * R / (2 * sigma * sigma));
  return { R, p };
}

/* ---------------- RNG ---------------- */
function rng(seed) { let a = seed >>> 0; return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gaussPair(r) { const u = Math.max(r(), 1e-12), v = r(); const R = Math.sqrt(-2 * Math.log(u)); return [R * Math.cos(TAU * v), R * Math.sin(TAU * v)]; }

/* ---------------- example: reconstruction of fig. 92 ---------------- */
function tearPoly(D, d, w, cls) {
  const Rv = D / 2, Rd = d / 2, wm = w, y1 = Math.sqrt(Rv * Rv - w * w / 4), a = wm / 2 + Rd, b2 = ((2 * Rd + wm) ** 2 - w * w) / 4;
  if (b2 <= 0) return null; const b = Math.sqrt(b2), RC = RCof(cls, Rv, Rd);
  if (RC <= y1 - b) return null;
  const s = a * a - b * b, disc = a * a * (s + y1 * y1) - RC * RC * s;
  let y2 = disc >= 0 ? (a * a * y1 - b * Math.sqrt(disc)) / s : Math.min(y1, RC); if (y2 > RC) y2 = RC;
  const ph = Math.atan2(y2, Math.sqrt(Math.max(RC * RC - y2 * y2, 0)));
  const y1p = RC + b, x1 = w / 2, x3 = Rv * Math.cos(ph), y3 = Rv * Math.sin(ph);
  return [[x1, y1p], [x3, y3], [-x3, y3], [-x1, y1p]]; // (v, u) in trace frame
}
function exampleFig92(withTear, tearClass) {
  const regs = [
    { name: "Область 1", w: 0.2, D: 1.3, x0: 5, y0: 35 },
    { name: "Область 2", w: 0.3, D: 1.4, x0: 5, y0: 5 },
    { name: "Область 3", w: 0.4, D: 1.6, x0: 55, y0: 5 },
    { name: "Область 4", w: 0.5, D: 1.8, x0: 55, y0: 35 }];
  const f = v => Math.round(v * 1e6);
  let g = "%TF.FileFunction,Copper,L1,Top*%\n%FSLAX46Y46*%\n%MOMM*%\nG01*\n";
  regs.forEach((r, k) => { g += `%ADD${10 + k}C,${r.D}*%\n%ADD${20 + k}C,${r.w}*%\n`; });
  regs.forEach((r, k) => {
    g += `D${20 + k}*\n`;
    for (let j = 0; j < 5; j++) { const y = r.y0 + 5 * j; g += `X${f(r.x0)}Y${f(y)}D02*\nX${f(r.x0 + 40)}Y${f(y)}D01*\n`; }
    for (let i = 0; i < 9; i++) { const x = r.x0 + 5 * i; g += `X${f(x)}Y${f(r.y0)}D02*\nX${f(x)}Y${f(r.y0 + 20)}D01*\n`; }
    g += `D${10 + k}*\n`;
    for (let j = 0; j < 5; j++) for (let i = 0; i < 9; i++) g += `X${f(r.x0 + 5 * i)}Y${f(r.y0 + 5 * j)}D03*\n`;
    if (withTear) {
      const poly = tearPoly(r.D, 1.0, r.w, tearClass);
      if (poly) for (let j = 0; j < 5; j++) for (let i = 0; i < 9; i++) {
        const cx = r.x0 + 5 * i, cy = r.y0 + 5 * j;
        for (const [ux, uy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          if ((i === 0 && ux < 0) || (i === 8 && ux > 0) || (j === 0 && uy < 0) || (j === 4 && uy > 0)) continue;
          const vx = -uy, vy = ux;
          const pts = poly.map(([v, u]) => [cx + u * ux + v * vx, cy + u * uy + v * vy]);
          g += "G36*\n" + `X${f(pts[0][0])}Y${f(pts[0][1])}D02*\n` + pts.slice(1).concat([pts[0]]).map(p => `X${f(p[0])}Y${f(p[1])}D01*\n`).join("") + "G37*\n";
        }
      }
    }
  });
  g += "M02*\n";
  let d = "M48\n; FILE_FORMAT=3:3\nMETRIC,LZ\nT1C1.000\n%\nG90\nG05\nT1\n";
  regs.forEach(r => { for (let j = 0; j < 5; j++) for (let i = 0; i < 9; i++) d += `X${(r.x0 + 5 * i).toFixed(3)}Y${(r.y0 + 5 * j).toFixed(3)}\n`; });
  d += "M30\n";
  return { files: [{ name: withTear ? "fig92_tear-F_Cu.gbr" : "fig92-F_Cu.gbr", text: g }, { name: "fig92-PTH.drl", text: d }],
    regions: regs.map(r => ({ name: r.name, w: r.w, D: r.D, bb: [r.x0 - 1.5, r.y0 - 1.5, r.x0 + 41.5, r.y0 + 21.5] })) };
}

/* ---------------- pipeline ---------------- */
function circleProb(R, mx, my, s) { // P(|X| <= R) for X ~ N((mx,my), s^2 I)
  if (R <= 0) return 0;
  if (s === 0) return Number(Math.hypot(mx, my) <= R);
  if (Math.abs(mx) < 1e-12 && Math.abs(my) < 1e-12) return 1 - Math.exp(-R * R / (2 * s * s));
  const lo = Math.max(-R, my - 9*s), hi = Math.min(R, my + 9*s);
  if (hi <= lo) return 0;
  const n = 160, h = (hi-lo) / n; let P = 0;
  for (let j = 0; j <= n; j++) { const y = lo + j * h; const xm = Math.sqrt(Math.max(R * R - y * y, 0));
    const fy = Math.exp(-(y - my) * (y - my) / (2 * s * s)) / (Math.sqrt(TAU) * s) * (Phi((xm - mx) / s) - Phi((-xm - mx) / s));
    P += fy * (j === 0 || j === n ? 1 : (j % 2 ? 4 : 2)); }
  return Math.min(1, P * h / 3);
}
function layerStats(layer) {
  let Ltot = 0, wmin = Infinity;
  for (const p of layer.prims) if (p.t === "L" && p.pol === "D" && p.ap.s === "C") { Ltot += Math.hypot(p.x2 - p.x1, p.y2 - p.y1); }
  for (const pad of layer.pads || []) for (const tr of pad.traces || []) wmin = Math.min(wmin, tr.w);
  if (!Number.isFinite(wmin)) { for (const p of layer.prims) if (p.t === "L" && p.pol === "D" && p.ap.s === "C" && p.ap.d > 0.02) wmin = Math.min(wmin, p.ap.d); }
  return { Ltot, wmin: Number.isFinite(wmin) ? wmin : null };
}
function sigmaMax(err, span) {
  const s = Math.sqrt(Math.max(err.sg, err.sgInner ?? err.sg) ** 2 + err.sl * err.sl + err.sd * err.sd + (err.srot * span) ** 2 + (err.ssc * span) ** 2);
  return s + Math.hypot(err.mx, err.my) / 3;
}
async function analyze(makeCanvas, model, crit, err, opts, progress, tick) {
  // model: {layers:[{name,prims}], holes:[...]}
  const bb = holeBBox(model.holes); model.center = [(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2];
  const span = Math.hypot(bb[2] - bb[0], bb[3] - bb[1]) / 2;
  const Rmax = Math.max(0.12, 6.5 * sigmaMax(err, span));
  model.Rmax = Rmax; model.crit = Object.assign({}, crit);
  let done = 0; const total = model.layers.length * model.holes.length;
  for (const L of model.layers) {
    L.criteria = L.inner ? {...crit, ringReq: crit.ringInner ?? crit.ringReq} : crit;
    L.pads = [];
    const B = 40;
    for (let k = 0; k < model.holes.length; k += B) {
      const sub = model.holes.slice(k, k + B);
      const pads = analyzeLayer(makeCanvas, L, sub, { res: opts.res, margin: Math.max(0.25, Rmax) });
      for (const p of pads) {
        p.hole += k;
        if (!p.plane) { const lim = Math.min(Rmax, p.Hw - p.Rd); p.M = buildMap(p, L.criteria, lim, opts.G || 121); }
        delete p.E; delete p.mask; delete p.lookup;
        L.pads.push(p);
      }
      done += sub.length; if (progress) progress(done / total);
      if (tick) await tick();
    }
    Object.assign(L, layerStats(L));
  }
  return model;
}
function holeBBox(holes) { let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity; for (const h of holes) { a = Math.min(a, h.x); b = Math.min(b, h.y); c = Math.max(c, h.x); d = Math.max(d, h.y); } return holes.length ? [a, b, c, d] : [0, 0, 0, 0]; }
function padCov(p, err, ctr) {
  const sg = p.layer.inner ? (err.sgInner ?? err.sg) : err.sg;
  const dx = p.x - ctr[0], dy = p.y - ctr[1], base = sg * sg + err.sl * err.sl + err.sd * err.sd;
  const r2 = err.srot * err.srot, s2 = err.ssc * err.ssc;
  return { sxx: base + r2 * dy * dy + s2 * dx * dx, syy: base + r2 * dx * dx + s2 * dy * dy, sxy: -r2 * dx * dy };
}
async function evaluate(model, err, route, opts, tick) {
  const cls = model.crit.cls, ctr = model.center, T = opts.trials || 2000;
  const all = [];
  for (const L of model.layers) for (const p of L.pads) if (!p.plane) { p.layer = L; all.push(p); }
  // analytic marginals and author's per-pad values
  let logProd = 0, logAfix = 0;
  for (const p of all) {
    const c = padCov(p, err, ctr);
    p.pm = p.M ? mapProb(p.M, err.mx, err.my, c.sxx, c.syy, c.sxy) : 0;
    const own = p.traces.length ? Math.min(...p.traces.map(t => t.w)) : null;
    const wl = (opts.wminMode === "pad" && own) ? own : (p.layer.wmin || own || 0.1);
    const sg = p.layer.inner ? (err.sgInner ?? err.sg) : err.sg;
    const sAuth = Math.hypot(sg, err.sl, err.sd);
    const circle = circularPad(p, cls, wl, sAuth, p.layer.criteria);
    p.paf = (err.mx || err.my) ? circleProb(circle.R, err.mx, err.my, sAuth) : circle.p;
    p.Raf = circle.R;
    logProd += Math.log(p.pm);
    logAfix += Math.log(p.paf);
  }
  // Monte Carlo
  const R = rng(opts.seed || 12345);
  const fails = new Map(all.map(p => [p, []]));
  const holeN = model.holes.length, dX = new Float64Array(holeN), dY = new Float64Array(holeN);
  const CH = 100;
  for (let t = 0; t < T; t++) {
    for (let h = 0; h < holeN; h++) { const g = gaussPair(R); dX[h] = g[0] * err.sd; dY[h] = g[1] * err.sd; }
    for (const L of model.layers) {
      const g = gaussPair(R), th = gaussPair(R)[0] * err.srot, sc = gaussPair(R);
      const sg = L.inner ? (err.sgInner ?? err.sg) : err.sg;
      const gx = err.mx + g[0] * sg, gy = err.my + g[1] * sg, sx = sc[0] * err.ssc, sy = sc[1] * err.ssc;
      for (const p of L.pads) {
        if (p.plane) continue;
        const l = gaussPair(R), rx = p.x - ctr[0], ry = p.y - ctr[1];
        const ox = gx - th * ry + sx * rx + l[0] * err.sl + dX[p.hole];
        const oy = gy + th * rx + sy * ry + l[1] * err.sl + dY[p.hole];
        if (!p.M || !mapOk(p.M, ox, oy)) fails.get(p).push(t);
      }
    }
    if (tick && t % CH === CH - 1) { await tick(t / T); if (opts.isStale && opts.isStale()) return null; }
  }
  for (const p of all) p.fails = fails.get(p);
  // route
  const layerRows = model.layers.map(L => {
    const pr = L.wmin ? Proute(L.wmin, route.h, cls, route.xi, route.F, model.crit.kred, route.gain) : 1;
    const Neff = route.lcorr > 0 ? Math.max(1, L.Ltot / route.lcorr) : 1;
    return { name: L.name, pads: L.pads.filter(p => !p.plane).length, planes: L.pads.filter(p => p.plane).length, wmin: L.wmin, Ltot: L.Ltot, pr, prN: Math.pow(pr, Neff), Neff };
  });
  const prAll = layerRows.reduce((a, r) => a * r.pr, 1), prNAll = layerRows.reduce((a, r) => a * r.prN, 1);
  return { pads: all, T, layerRows, prAll, prNAll, prodMarg: Math.exp(logProd), authorFix: Math.exp(logAfix) };
}
function subset(res, filter) {
  const ps = res.pads.filter(filter), failT = new Set();
  let lp = 0, laf = 0;
  for (const p of ps) { for (const t of p.fails) failT.add(t); lp += Math.log(p.pm); laf += Math.log(p.paf); }
  const k = res.T - failT.size;
  return { n: ps.length, mc: k / res.T, mcK: k, prodMarg: Math.exp(lp), authorFix: Math.exp(laf), ci: wilson(k, res.T) };
}
function wilson(k, n) { const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [Math.max(0, c - h), Math.min(1, c + h)]; }

return { parseGerber, parseExcellon, detectRole, renderPrim, buildIndex, query, analyzeLayer, makeTester, buildMap, mapOk, mapProb,
  circularPad, Proute, RCof, rng, gaussPair, exampleFig92, CLASS_K, edt, rasterWindow, analyze, evaluate, subset, holeBBox, circleProb, sigmaMax };
})();
if (typeof module !== "undefined") module.exports = GEngine;
