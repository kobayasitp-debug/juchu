/* 写真の文字読み取り（PaddleOCR v5 mobile / ONNX Runtime Web）
 * すべてiPadの中で処理します（ネット不要）。
 * 1) 文字のある場所を探す（det）→ 2) 1行ずつ切り出して読む（rec）→ 3) 行ごとに並べ直す
 */
(function () {
  'use strict';
  const BASE = new URL('./', document.currentScript ? document.currentScript.src : location.href).href;
  const DET_MAX = 1920;      // 検出に使う画像の長辺（大きいほど小さな文字に強いが遅い）
  const BIN_T = 0.3, BOX_T = 0.55, UNCLIP = 1.6, MIN_SIDE = 3;
  let det = null, rec = null, keys = null, loading = null;

  async function init(onProg) {
    if (det && rec) return;
    if (loading) return loading;
    loading = (async () => {
      const ort = window.ort;
      if (!ort) throw new Error('読み取りエンジンを読み込めませんでした');
      ort.env.wasm.wasmPaths = BASE;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.logLevel = 'error';
      const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
      onProg && onProg('準備中（1/3）', 0.05);
      det = await ort.InferenceSession.create(BASE + 'det.ort', opt);
      onProg && onProg('準備中（2/3）', 0.12);
      rec = await ort.InferenceSession.create(BASE + 'rec.onnx', opt);
      onProg && onProg('準備中（3/3）', 0.18);
      const txt = await (await fetch(BASE + 'dict.txt')).text();
      const lines = txt.replace(/\r/g, '').split('\n');
      // 辞書ファイル中の空行は文字ではないので除く（残すと1文字ずつずれる）
      keys = ['', ...lines.filter(l => l !== '')];
    })();
    try { await loading; } finally { loading = null; }
  }

  /* ---------- 画像 ---------- */
  function toCanvas(src, maxSide) {
    const w0 = src.naturalWidth || src.width, h0 = src.naturalHeight || src.height;
    const s = Math.min(1, maxSide / Math.max(w0, h0));
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(w0 * s)); c.height = Math.max(1, Math.round(h0 * s));
    const g = c.getContext('2d', { willReadFrequently: true });
    g.imageSmoothingQuality = 'high'; g.drawImage(src, 0, 0, c.width, c.height);
    return c;
  }
  function rotateCanvas(src, deg) {
    const c = document.createElement('canvas'), q = ((deg % 360) + 360) % 360;
    c.width = q % 180 ? src.height : src.width; c.height = q % 180 ? src.width : src.height;
    const g = c.getContext('2d');
    g.translate(c.width / 2, c.height / 2); g.rotate(q * Math.PI / 180); g.drawImage(src, -src.width / 2, -src.height / 2);
    return c;
  }

  /* ---------- 1) 文字の場所を探す ---------- */
  async function detect(img) {
    const ort = window.ort;
    const W0 = img.width, H0 = img.height;
    const s = Math.min(1, DET_MAX / Math.max(W0, H0));
    const W = Math.max(32, Math.round(W0 * s / 32) * 32), H = Math.max(32, Math.round(H0 * s / 32) * 32);
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const g = c.getContext('2d', { willReadFrequently: true }); g.imageSmoothingQuality = 'high'; g.drawImage(img, 0, 0, W, H);
    const px = g.getImageData(0, 0, W, H).data, n = W * H, inp = new Float32Array(3 * n);
    // PaddleOCRと同じくBGR順で正規化
    const M = [0.485, 0.456, 0.406], S = [0.229, 0.224, 0.225];
    for (let i = 0, j = 0; j < n; i += 4, j++) {
      inp[j] = (px[i + 2] / 255 - M[0]) / S[0];
      inp[n + j] = (px[i + 1] / 255 - M[1]) / S[1];
      inp[2 * n + j] = (px[i] / 255 - M[2]) / S[2];
    }
    const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', inp, [1, 3, H, W]) });
    const P = out[det.outputNames[0]].data;
    return boxesFromMap(P, W, H, W0 / W, H0 / H);
  }

  function boxesFromMap(P, W, H, rx, ry) {
    const n = W * H, lab = new Int32Array(n), boxes = [];
    const stack = new Int32Array(n);
    let id = 0;
    for (let p0 = 0; p0 < n; p0++) {
      if (lab[p0] || P[p0] <= BIN_T) continue;
      id++; let sp = 0; stack[sp++] = p0; lab[p0] = id;
      let cnt = 0, sum = 0;
      const rowMin = new Map(), rowMax = new Map();
      while (sp) {
        const p = stack[--sp], x = p % W, y = (p - x) / W;
        cnt++; sum += P[p];
        const a = rowMin.get(y); if (a === undefined || x < a) rowMin.set(y, x);
        const b = rowMax.get(y); if (b === undefined || x > b) rowMax.set(y, x);
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy; if (yy < 0 || yy >= H) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx; if (xx < 0 || xx >= W) continue;
            const q = yy * W + xx;
            if (!lab[q] && P[q] > BIN_T) { lab[q] = id; stack[sp++] = q; }
          }
        }
      }
      if (cnt < 10) continue;
      if (sum / cnt < BOX_T) continue;
      const pts = [];
      rowMin.forEach((x, y) => { pts.push([x, y], [x + 1, y], [x, y + 1], [x + 1, y + 1]); });
      rowMax.forEach((x, y) => { pts.push([x + 1, y], [x + 1, y + 1]); });
      const r = minAreaRect(hull(pts));
      if (Math.min(r.w, r.h) < MIN_SIDE) continue;
      const d = (r.w * r.h) * UNCLIP / (2 * (r.w + r.h));
      r.w += 2 * d; r.h += 2 * d;
      // 元画像の座標へ
      const c = Math.cos(r.a), s = Math.sin(r.a);
      const ux = [c * rx, s * ry], vx = [-s * rx, c * ry];
      const w = r.w * Math.hypot(ux[0], ux[1]), h = r.h * Math.hypot(vx[0], vx[1]);
      const a = Math.atan2(ux[1], ux[0]);
      boxes.push({ cx: r.cx * rx, cy: r.cy * ry, w, h, a, score: sum / cnt });
    }
    return boxes;
  }
  function hull(p) {
    p.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [], up = [];
    for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
    for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
    up.pop(); lo.pop(); return lo.concat(up);
  }
  // 最小外接長方形。幅の向きは水平に近い方（-45°〜45°）にそろえる
  function minAreaRect(h) {
    let best = null;
    for (let i = 0; i < h.length; i++) {
      const p = h[i], q = h[(i + 1) % h.length];
      const ang = Math.atan2(q[1] - p[1], q[0] - p[0]), c = Math.cos(ang), s = Math.sin(ang);
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const t of h) { const u = t[0] * c + t[1] * s, v = -t[0] * s + t[1] * c; if (u < x0) x0 = u; if (u > x1) x1 = u; if (v < y0) y0 = v; if (v > y1) y1 = v; }
      const area = (x1 - x0) * (y1 - y0);
      if (!best || area < best.area) {
        const mu = (x0 + x1) / 2, mv = (y0 + y1) / 2;
        best = { area, cx: mu * c - mv * s, cy: mu * s + mv * c, w: x1 - x0, h: y1 - y0, a: ang };
      }
    }
    if (!best) return { cx: 0, cy: 0, w: 0, h: 0, a: 0 };
    let { w, h: hh, a } = best;
    while (a > Math.PI / 4) { a -= Math.PI / 2; [w, hh] = [hh, w]; }
    while (a <= -Math.PI / 4) { a += Math.PI / 2; [w, hh] = [hh, w]; }
    return { cx: best.cx, cy: best.cy, w, h: hh, a };
  }

  /* ---------- 2) 1行ずつ読む ---------- */
  const RH = 48;
  function crop(img, b) {
    const W = Math.max(8, Math.min(3200, Math.round(RH * b.w / Math.max(1, b.h))));
    const c = document.createElement('canvas'); c.width = W; c.height = RH;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.fillStyle = '#fff'; g.fillRect(0, 0, W, RH); g.imageSmoothingQuality = 'high';
    const sx = W / b.w, sy = RH / b.h, co = Math.cos(b.a), si = Math.sin(b.a);
    const A = sx * co, C = sx * si, B = -sy * si, D = sy * co;
    g.setTransform(A, B, C, D, W / 2 - (A * b.cx + C * b.cy), RH / 2 - (B * b.cx + D * b.cy));
    g.drawImage(img, 0, 0);
    return g.getImageData(0, 0, W, RH);
  }
  async function recognize(id) {
    const ort = window.ort, W = id.width, n = W * RH, px = id.data, inp = new Float32Array(3 * n);
    for (let i = 0, j = 0; j < n; i += 4, j++) {
      inp[j] = px[i + 2] / 127.5 - 1; inp[n + j] = px[i + 1] / 127.5 - 1; inp[2 * n + j] = px[i] / 127.5 - 1;
    }
    const out = await rec.run({ [rec.inputNames[0]]: new ort.Tensor('float32', inp, [1, 3, RH, W]) });
    const t = out[rec.outputNames[0]], [, T, C] = t.dims, d = t.data;
    let s = '', last = -1, conf = 0, k = 0;
    for (let i = 0; i < T; i++) {
      let mi = 0, mv = -Infinity; const o = i * C;
      for (let j = 0; j < C; j++) { const v = d[o + j]; if (v > mv) { mv = v; mi = j; } }
      if (mi !== 0 && mi !== last) { s += mi < keys.length ? keys[mi] : ' '; conf += mv; k++; }
      last = mi;
    }
    return { text: s.trim(), conf: k ? conf / k : 0 };
  }

  /* ---------- 3) 行ごとに並べる ---------- */
  function toLines(items) {
    if (!items.length) return '';
    const angs = items.filter(b => b.w > b.h * 2).map(b => b.a).sort((x, y) => x - y);
    const am = angs.length ? angs[angs.length >> 1] : 0, co = Math.cos(-am), si = Math.sin(-am);
    for (const b of items) { b.u = b.cx * co - b.cy * si; b.v = b.cx * si + b.cy * co; }
    items.sort((p, q) => p.v - q.v);
    const rows = [];
    for (const b of items) {
      const r = rows.find(r => Math.abs(r.v - b.v) < Math.min(r.h, b.h) * 0.55);
      if (r) { r.items.push(b); r.v = (r.v * (r.items.length - 1) + b.v) / r.items.length; r.h = Math.max(r.h, b.h); }
      else rows.push({ v: b.v, h: b.h, items: [b] });
    }
    rows.sort((p, q) => p.v - q.v);
    return rows.map(r => r.items.sort((p, q) => p.u - q.u).map(b => b.text).join(' ')).join('\n');
  }

  async function readCanvas(img, onProg, base, span) {
    const boxes = await detect(img);
    const items = [];
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      if (b.h > b.w * 1.6 && b.w < 60) {
        // 縦長の小さな箱は縦書き（1文字ずつ並んだもの）とみなし、90°回して読む
        const r = await recognize(crop(img, { ...b, w: b.h, h: b.w, a: b.a + Math.PI / 2 }));
        if (r.text && r.conf > 0.5) items.push({ ...b, ...r });
        continue;
      }
      const r = await recognize(crop(img, b));
      if (r.text && r.conf > 0.45) items.push({ ...b, ...r });
      if (onProg && i % 3 === 0) onProg(`文字を読み取り中… ${Math.round((i + 1) / boxes.length * 100)}%`, base + span * (i + 1) / boxes.length);
    }
    return { items, boxes };
  }

  // 公開する関数：画像（Blob）→ 行ごとのテキスト
  async function read(blob, onProg) {
    const prog = (t, p) => onProg && onProg(t, p);
    prog('読み取りの準備中…', 0.02);
    await init(prog);
    const u = URL.createObjectURL(blob);
    const im = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => rej(new Error('画像を開けませんでした')); i.src = u; });
    URL.revokeObjectURL(u);
    let img = toCanvas(im, 3200);
    prog('文字の場所を探しています…', 0.2);
    let r = await readCanvas(img, prog, 0.25, 0.75);
    // 横向きに撮った書類（縦長の箱ばかり）なら回して読み直す
    const tall = r.boxes.filter(b => b.h > b.w * 1.5).length;
    if (r.boxes.length >= 6 && tall > r.boxes.length * 0.6) {
      const cand = [];
      for (const deg of [90, -90]) {
        prog('向きを直して読み直しています…', 0.3);
        const c = rotateCanvas(img, deg), rr = await readCanvas(c, prog, 0.3, 0.7);
        cand.push({ rr, sc: rr.items.reduce((s, x) => s + x.conf * x.text.length, 0) });
      }
      cand.sort((a, b) => b.sc - a.sc); r = cand[0].rr;
    }
    prog('読み取りが終わりました', 1);
    return toLines(r.items);
  }

  window.PPOCR = { read, init };
})();
