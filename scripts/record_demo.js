// Scripted demo of the viewer: rotate, zoom, pan, cut planes, corner cutaway, camera cut, window, density.
// Saves every JPEG the server streams to the page (ws/, full rate) and the browser screen via CDP screencast
// (screen/, ~17 fps in headless, used for the control panel), with timestamps and the caption timeline.
// compose_demo.py turns them into one 60 fps video.
// usage: node scripts/record_demo.js <playwright dir> <dir holding libroot/ (extra Chromium libs)> <url> <frames dir>
const { chromium } = require(process.argv[2] + '/node_modules/playwright');
const fs = require('fs');
const [scratch, url, fdir] = [process.argv[3], process.argv[4], process.argv[5]];
const LIB = scratch + '/libroot/usr/lib/' + (process.arch === 'arm64' ? 'aarch64' : 'x86_64') + '-linux-gnu';
(async () => {
  for (const d of ['ws', 'screen']) fs.mkdirSync(`${fdir}/${d}`, { recursive: true });
  const b = await chromium.launch({ args: ['--no-sandbox'], env: { ...process.env, LD_LIBRARY_PATH: LIB } });
  const ctx = await b.newContext({ viewport: { width: 1920, height: 1080 } });
  await ctx.addInitScript(() => {
    localStorage.setItem('stream-settings', JSON.stringify({ res: '1920', prev: '1', q: '95', win: '2' }));
  });
  const p = await ctx.newPage();
  const wsLog = [];                                // [arrival s, seq, w, h, bytes, frame_ms, gpu_ms]
  p.on('websocket', ws => ws.on('framereceived', f => {
    if (typeof f.payload === 'string') return;
    const n = f.payload.readUInt32LE(0), h = JSON.parse(f.payload.subarray(4, 4 + n).toString());
    fs.writeFileSync(`${fdir}/ws/${String(wsLog.length).padStart(6, '0')}.jpg`, f.payload.subarray(4 + n));
    wsLog.push([Date.now() / 1000, h.seq, h.w, h.h, h.bytes, h.frame_ms, h.gpu_ms]);
  }));
  await p.goto(url + '?noworker');                 // socket on the main thread, so Playwright sees its frames
  await p.waitForFunction(() => /fps|idle/.test(document.getElementById('fps').innerText), null, { timeout: 60000 });
  // helpers in the page's global scope (view and changed() are top-level bindings of the page script)
  // animation runs inside the page, one step per requestAnimationFrame (60 Hz), on the same code path as the mouse
  // handlers (view + changed(true)); view, changed() and basis() are top-level bindings of the page script
  await p.addScriptTag({ content: `
    const ease = t => t < 0.5 ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t);
    const set = (id, v) => { const e = document.getElementById(id); e.value = v; e.dispatchEvent(new Event('input', {bubbles: true})); };
    window.__anim = (ms, kind, a) => new Promise(done => {
      const v0 = {...view}, s0 = {};
      if (kind === 'sliders') for (const k in a) s0[k] = +document.getElementById(k).value;
      const {r, u} = basis(), px = 2 * v0.dist * Math.tan(Math.PI / 12) / document.getElementById('view').clientHeight;
      const t0 = performance.now();
      const step = now => {
        const t = Math.min(1, (now - t0) / ms), e = ease(t);
        if (kind === 'sliders') { for (const k in a) set(k, s0[k] + (a[k] - s0[k]) * e); }
        else {
          if (kind === 'zoom') view.dist = v0.dist * Math.exp(Math.log(a.dist / v0.dist) * e);
          if (kind === 'turn') { view.az = v0.az + (a.az - v0.az) * e; view.el = v0.el + (a.el - v0.el) * e; }
          if (kind === 'orbit') { view.az = v0.az - a.az * Math.sin(2 * Math.PI * e); view.el = Math.max(-89, Math.min(89, v0.el + a.el * Math.sin(Math.PI * e))); }
          if (kind === 'pan') { const dx = a.x * Math.sin(2 * Math.PI * e), dy = a.y * Math.sin(Math.PI * e);
            for (const [i, k] of ['tx', 'ty', 'tz'].entries()) view[k] = v0[k] - (r[i] * dx - u[i] * dy) * px; }
          changed(true);
        }
        if (t < 1) requestAnimationFrame(step); else done();
      };
      requestAnimationFrame(step);
    });` });
  await p.waitForTimeout(3000);

  const cdp = await ctx.newCDPSession(p);
  const stamps = [];
  cdp.on('Page.screencastFrame', f => {
    const n = stamps.length;
    fs.writeFileSync(`${fdir}/screen/${String(n).padStart(6, '0')}.jpg`, Buffer.from(f.data, 'base64'));
    stamps.push([f.metadata.timestamp, Date.now() / 1000]);
    cdp.send('Page.screencastFrameAck', { sessionId: f.sessionId }).catch(() => {});
  });
  await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 95, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });

    const wait = ms => p.waitForTimeout(ms);
  const caps = [];
  const cap = async t => caps.push([Date.now() / 1000, t]);
  const run = (ms, kind, a) => p.evaluate(([ms, kind, a]) => __anim(ms, kind, a), [ms, kind, a]);
  const sliders = (ms, to) => run(ms, 'sliders', to);
  const zoom = (ms, dist) => run(ms, 'zoom', { dist });
  const orbit = (ms, az, el) => run(ms, 'orbit', { az, el });       // degrees: a loop out and back
  const pan = (ms, x, y) => run(ms, 'pan', { x, y });               // pixels, like Shift + drag
  const turn = (ms, az, el) => run(ms, 'turn', { az, el });
  const click = async id => { await p.evaluate(id => document.getElementById(id).click(), id); await wait(200); };  // no scroll

  await cap('Whole brain · rendered on a server GPU, streamed to the browser');
  await wait(3500);
  await cap('Rotate · drag'); await orbit(5000, 130, 50); await wait(800);
  await cap('Zoom in · wheel — bricks stream in and sharpen'); await zoom(3000, 34); await wait(2500);
  await cap('Pan · Shift + drag'); await pan(3000, 180, 70); await wait(600);
  await cap('Zoom out'); await zoom(2000, 146); await click('reset'); await wait(800);

  await cap('Cut plane A–P (x)'); await sliders(3000, { cx: 0.55 }); await wait(1200);
  await cap('Second cut plane S–I (y)'); await sliders(2500, { cy: 0.45 }); await wait(800);
  await cap('Rotate with the cuts'); await orbit(4000, -120, 45); await wait(600);
  await cap('Flip · keep the other side'); await click('fx'); await wait(1800); await click('fx'); await wait(600);
  await cap('Remove the cuts'); await sliders(1800, { cx: 0, cy: 0 }); await click('reset'); await wait(600);

  await cap('Corner cutaway · three planes'); await click('mode'); await sliders(2500, { cx: 0.5, cy: 0.5, cz: 0.5 }); await wait(1000);
  await cap('The removed corner follows the camera'); await orbit(6000, 150, 60); await wait(800);
  await cap('Zoom into the cut faces'); await zoom(3500, 16); await wait(3000);
  await zoom(2500, 146); await click('mode'); await sliders(1200, { cx: 0, cy: 0, cz: 0 }); await wait(400);

  await cap('Camera cut · a plane facing the viewer'); await turn(2000, 0, 0); await click('cam');
  await sliders(1200, { camd: -0.7 }); await wait(400);
  await cap('Fly through the sections'); await sliders(5000, { camd: 0.35 }); await sliders(2500, { camd: -0.3 }); await wait(1200);
  await cap('Zoom into one section · native voxels'); await zoom(3500, 22); await wait(3000);
  await cap('Window · contrast'); await sliders(2000, { hi: 255, lo: 30 }); await wait(2000); await sliders(1500, { hi: 200, lo: 8 }); await wait(400);
  await cap('Zoom out · the cut follows rotation'); await zoom(2500, 146); await orbit(5000, 110, 45); await wait(600);
  await click('cam'); await click('reset'); await wait(600);

  await cap('Density · volume opacity'); await sliders(2000, { dens: 4.5 }); await sliders(2500, { dens: 0.3 }); await sliders(1500, { dens: 1.5 }); await wait(600);
  await cap('Fixed 4 GB GPU memory · 2–7 ms per frame on the server'); await orbit(4000, 100, 40); await wait(2500);
  await cap(''); await wait(1000);

  await cdp.send('Page.stopScreencast'); await wait(300);
  const end = await p.evaluate(() => ({ fps: document.getElementById('fps').innerText,
    mem: document.getElementById('mem').innerText, stats: document.getElementById('stats').innerText }));
  fs.writeFileSync(`${fdir}/log.json`, JSON.stringify({ screen: stamps, ws: wsLog, caps }));
  await ctx.close(); await b.close();
  const dur = stamps[stamps.length - 1][0] - stamps[0][0], t0 = stamps[0][0];
  const nws = wsLog.filter(w => w[0] >= t0).length;
  console.log(`${dur.toFixed(1)} s: screen ${stamps.length} frames (${(stamps.length / dur).toFixed(1)} fps), ` +
    `stream ${nws} frames (${(nws / dur).toFixed(1)} fps)`);
  console.log(end.fps, '|', end.stats, '|', end.mem.replace(/\n/g, ' / '));
})().catch(e => { console.error('FAIL', e.message.split('\n').slice(0, 8).join('\n')); process.exit(1); });
