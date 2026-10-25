# web

Browser client (`index.html`, served by `server/app.py`): plain HTML and JavaScript, no build step. It has one 3D view,
orbit (drag), pan (shift or right drag) and zoom (wheel).

- **Open the brain:** three planes (sagittal, coronal, axial), each switched on separately. The brain stays whole and
  only the region on the camera's side of the enabled planes is removed: one plane opens a half, two a wedge, three a
  corner, turning to face the camera. Unticking "open" crops the volume to the planes instead. The plane names come
  from the dataset (`planes` in its config), since datasets are cut along different axes.
- **Camera cut:** a plane facing the camera that follows rotation.
- **Rendering:** colour map (grayscale, sepia, hot, viridis, magma, bone, violet stain, or a custom two-colour
  gradient) with gamma and a preview bar drawn from `/lut`; window low/high, density, cut-face opacity, level of
  detail, and empty-space skipping. Every change applies on the next frame.
- **Live meter** (top right): GPU load, the server's GPU memory, system RAM and frames rendered per second, from
  `/stats` every 0.5 s, with one-minute graphs.
- **Transport:** a WebSocket at `ws(s)://<host>/ws`, opened inside a Web Worker. The worker acknowledges each frame as soon as it arrives and decodes the JPEG off the main thread, so page repaints never hold back the server's frame window.
  - The page sends its view state at most once per animation frame.
  - If the socket cannot open, the page falls back to plain `GET /render` requests.
  - `?noworker` keeps the socket on the main thread, for debugging.
- **Stream settings:** resolution at rest (256 px to 3840 px; above the screen's size the browser scales the frame down), resolution while dragging, JPEG quality and frames in flight. They are kept in `localStorage`.
- **Auto drag resolution (the default):** while dragging, the frame shrinks when frames fall behind the views sent or start queueing on the link, and grows back while the link keeps up.
  - "Queueing" means latency above the lowest latency seen on this link, so a long VPN round trip does not count as congestion.
  - The thresholds are in the `setInterval` block below `push()`.
- **Meters:** frames per second and Mbit/s over the last 3 s, and per-view latency (view sent → its frame drawn). A "sharpening…" badge shows while bricks are still loading.
- **Memory panel:** GPU pool (fixed), resident bricks, host cache, process RSS, and bricks loaded and evicted.
- **Test hook:** if `window.viewLog = {frames: [], sends: []}` exists before load, the page logs every view sent and every frame drawn into it. The headless benchmarks use this.
