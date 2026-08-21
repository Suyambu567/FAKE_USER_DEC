/**
 * Service worker — offline support.
 *
 * The app has no backend, so once the shell and the model are cached there is
 * nothing left to fetch: it works with the phone in aeroplane mode.
 *
 * Strategy is cache-first with a versioned cache name. That is safe here only
 * because every asset is immutable within a release — a new model or a code
 * change means bumping CACHE_VERSION below, which orphans the old cache and the
 * activate handler deletes it. **If you edit anything in webapp/ and do not bump
 * this string, returning visitors keep the old version indefinitely.**
 */

const CACHE_VERSION = 'fud-v1-20260807';

const ASSETS = [
  './',
  './index.html',
  './css/styles.css',
  './js/model.js',
  './js/features.js',
  './js/app.js',
  './model/model.json',
  './manifest.webmanifest',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/apple-touch-icon.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION)
      .then((cache) => cache.addAll(ASSETS))
      // Don't make the user close every tab to pick up a new release.
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k)),
      ))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  // Only same-origin GETs. There are no cross-origin requests, but a range
  // request or a POST must never be served from cache.
  if (event.request.method !== 'GET') return;
  if (new URL(event.request.url).origin !== self.location.origin) return;

  event.respondWith(
    caches.match(event.request).then((hit) => {
      if (hit) return hit;
      return fetch(event.request)
        .then((response) => {
          // Cache successful basic responses so a deep link works offline next time.
          if (response.ok && response.type === 'basic') {
            const copy = response.clone();
            caches.open(CACHE_VERSION).then((cache) => cache.put(event.request, copy));
          }
          return response;
        })
        .catch(() => caches.match('./index.html'));  // offline navigation fallback
    }),
  );
});
