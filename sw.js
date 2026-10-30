// オフライン用キャッシュ。アプリを更新したら VERSION を変える
const VERSION = 'kanri-v7';
// 文字読み取り用の大きなファイルは別に保管（アプリ更新のたびに再ダウンロードしない）
const OCR_CACHE = 'kanri-ocr-v2';
const ASSETS = ['./', './index.html', './manifest.webmanifest', './icon-192.png', './icon-512.png', './apple-touch-icon.png'];
const OCR_FILES = ['ocr2/ort.wasm.min.js', 'ocr2/ppocr.js', 'ocr2/ort-wasm-simd-threaded.mjs', 'ocr2/ort-wasm-simd-threaded.wasm', 'ocr2/det.ort', 'ocr2/rec.onnx', 'ocr2/dict.txt'];

self.addEventListener('install', e => {
  e.waitUntil((async () => {
    // cache:'reload' … ブラウザの一時保存をとばして、必ずサーバーの最新ファイルを取りに行く
    const c = await caches.open(VERSION);
    await c.addAll(ASSETS.map(u => new Request(u, { cache: 'reload' })));
    // 読み取り機能は失敗してもアプリ本体は使えるようにする（設定画面から再ダウンロード可）
    try {
      const oc = await caches.open(OCR_CACHE);
      for (const f of OCR_FILES) if (!(await oc.match(f))) await oc.add(f);
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

  // アプリ本体（画面）：ネットにつながっていれば常に最新版、つながらなければ保存した版
  const isPage = req.mode === 'navigate' || url.pathname.endsWith('/index.html') || url.pathname.endsWith('/');
  if (isPage) {
    e.respondWith((async () => {
      const cache = await caches.open(VERSION);
      try {
        const res = await Promise.race([
          fetch(req, { cache: 'no-store' }),
          new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 4000))
        ]);
        if (res && res.ok) { cache.put('./index.html', res.clone()); return res; }
        throw new Error('bad');
      } catch (err) {
        return (await cache.match('./index.html')) || (await cache.match('./')) ||
          new Response('オフラインです', { status: 503, headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
      }
    })());
    return;
  }

  // そのほか（アイコンなど）：保存した版をすぐ使い、裏で最新版に更新
  e.respondWith(
    caches.open(VERSION).then(async cache => {
      const hit = await cache.match(req, { ignoreSearch: true });
      const net = fetch(req).then(res => {
        if (res && res.ok) cache.put(req, res.clone());
        return res;
      }).catch(() => null);
      if (hit) { e.waitUntil(net); return hit; }
      return (await net) || new Response('オフラインです', { status: 503, headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
    })
  );
});
