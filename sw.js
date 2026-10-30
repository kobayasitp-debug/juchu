// オフライン用キャッシュ。アプリを更新したら VERSION を変える
const VERSION = 'kanri-v5';
// 文字読み取り用の大きなファイルは別に保管（アプリ更新のたびに再ダウンロードしない）
const OCR_CACHE = 'kanri-ocr-v2';
const ASSETS = ['./', './index.html', './manifest.webmanifest', './icon-192.png', './icon-512.png', './apple-touch-icon.png'];
const OCR_FILES = ['ocr2/ort.wasm.min.js', 'ocr2/ppocr.js', 'ocr2/ort-wasm-simd-threaded.mjs', 'ocr2/ort-wasm-simd-threaded.wasm', 'ocr2/det.ort', 'ocr2/rec.onnx', 'ocr2/dict.txt'];

self.addEventListener('install', e => {
  e.waitUntil((async () => {
    await (await caches.open(VERSION)).addAll(ASSETS);
    // 読み取り機能は失敗してもアプリ本体は使えるようにする（設定画面から再ダウンロード可）
    try {
      const c = await caches.open(OCR_CACHE);
      for (const f of OCR_FILES) if (!(await c.match(f))) await c.add(f);
    } catch (err) {}
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== VERSION && k !== OCR_CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== location.origin) return;

  // 読み取り用ファイル：キャッシュ優先（無ければ取得して保存）
  if (url.pathname.includes('/ocr2/')) {
    e.respondWith((async () => {
      const c = await caches.open(OCR_CACHE);
      const hit = await c.match(req, { ignoreSearch: true });
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) c.put(req, res.clone());
      return res;
    })());
    return;
  }

  // アプリ本体：まずキャッシュから即表示し、ネットがあれば裏で最新版に更新（次回起動時に反映）
  const key = req.mode === 'navigate' ? './index.html' : req;
  e.respondWith(
    caches.open(VERSION).then(async cache => {
      const hit = await cache.match(key, { ignoreSearch: true });
      const net = fetch(req).then(res => {
        if (res && res.ok) cache.put(key, res.clone());
        return res;
      }).catch(() => null);
      if (hit) { e.waitUntil(net); return hit; }
      return (await net) || new Response('オフラインです', { status: 503, headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
    })
  );
});
