"""HTTP + WebSocket front end for the bounded renderer.

GET /          the page (web/index.html)
GET /favicon.png  the project logo (docs/brand/logo.png)
GET /info      dataset, tissue box and memory numbers
GET /lut?…     colour table of a map (preview bar)
GET /stats     live RAM / GPU readings, last minute at 2 Hz (?since=<t> for newer ones only)
GET /render?…  one JPEG frame for a view given as query parameters (curl, simple clients)
GET /ws        WebSocket: the page sends its view state whenever it changes; the server renders the
               newest state and pushes frames. At most `window` frames are unacknowledged per
               viewer, so a slow link drops intermediate views instead of queueing them, and the frame
               rate is not capped at one frame per round trip. While bricks are still arriving the
               server keeps re-rendering, so the view sharpens without new input.

usage: python server/app.py data/bigbrain20_png.ome.zarr [port] [host] [--pool-gb 4] [--host-gb 8]
       [--finest 0] [--staging-mb 256] [--threads 64] [--pin-share 0.25]     (python server/app.py -h explains them)
"""
import argparse
import base64
import hashlib
import json
import os
import socket
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch

sys.path.insert(0, os.path.dirname(__file__))
import colormaps  # noqa: E402
from monitor import Monitor  # noqa: E402
from renderer import Renderer, parse  # noqa: E402

WEB = os.path.join(os.path.dirname(__file__), "..", "web")
LOGO = os.path.join(os.path.dirname(__file__), "..", "docs", "brand", "logo.png")
R = None
MON = None                                  # live RAM / GPU readings (monitor.py)
LOCK = threading.Lock()                     # one GPU renderer shared by all viewers


def memory():
    m = R.cache.memory()
    rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    m.update(gpu_alloc_gb=torch.cuda.memory_allocated() / 2**30, gpu_peak_gb=torch.cuda.max_memory_allocated() / 2**30,
             rss_gb=rss / 2**30)
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}


# ---- minimal RFC 6455 WebSocket (server side: unmasked sends, masked receives)
MAX_MSG = 1 << 20                           # client messages are small JSON; anything bigger is refused


def ws_recv(rfile):
    """Next (opcode, payload) message; fragments are joined."""
    data, op0 = b"", None
    while True:
        h = rfile.read(2)
        if len(h) < 2:
            return 8, b""
        fin, op, masked, n = h[0] & 0x80, h[0] & 0x0F, h[1] & 0x80, h[1] & 0x7F
        if n == 126:
            n = struct.unpack(">H", rfile.read(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", rfile.read(8))[0]
        if n + len(data) > MAX_MSG:
            raise ValueError("WebSocket message too large")
        mask = rfile.read(4) if masked else b"\0\0\0\0"
        payload = bytearray(rfile.read(n))
        for i in range(len(payload)):
            payload[i] ^= mask[i & 3]
        if op >= 8:                                      # control frames are never fragmented
            return op, bytes(payload)
        op0 = op if op else op0
        data += payload
        if fin:
            return op0, data


def ws_send(wfile, payload, op=2):
    n = len(payload)
    head = bytes([0x80 | op]) + (bytes([n]) if n < 126 else
                                 bytes([126]) + struct.pack(">H", n) if n < 65536 else
                                 bytes([127]) + struct.pack(">Q", n))
    wfile.write(head + payload)
    wfile.flush()


class Viewer:
    """One WebSocket client: a reader thread keeps the newest view and acks; the handler thread renders."""

    def __init__(self, handler):
        self.h = handler
        self.cond = threading.Condition()
        self.view, self.seq, self.version = None, 0, 0
        self.sent = self.acked = 0
        self.window = 2
        self.alive = True
        self.wlock = threading.Lock()

    def read_loop(self):
        try:
            while self.alive:
                op, data = ws_recv(self.h.rfile)
                if op == 8:
                    with self.wlock:                         # echo the close frame (RFC 6455 5.5.1)
                        ws_send(self.h.wfile, data[:2], op=8)
                    break
                if op == 9:
                    with self.wlock:
                        ws_send(self.h.wfile, data, op=10)
                    continue
                if op != 1:
                    continue
                msg = json.loads(data)
                with self.cond:
                    if msg.get("type") == "view":
                        self.view, self.seq = parse(msg["view"]), msg.get("seq", 0)
                        self.window = max(1, min(int(msg.get("window", 2)), 8))
                        self.version += 1
                    elif msg.get("type") == "ack":
                        self.acked = max(self.acked, int(msg["n"]))
                    self.cond.notify()
        except Exception:                                    # bad frame, bad JSON, closed socket: drop the viewer
            pass
        finally:
            with self.cond:
                self.alive = False
                self.cond.notify()

    def send_loop(self):
        done_version, refine, last, stalled = -1, False, 0.0, 0
        while True:
            with self.cond:
                while self.alive:
                    can_send = self.view is not None and self.sent - self.acked < self.window
                    want = self.version != done_version or (refine and time.time() - last > 0.03)
                    if can_send and want:
                        break
                    self.cond.wait(timeout=0.03)
                if not self.alive:
                    return
                p, seq, version = self.view, self.seq, self.version
            with LOCK:
                data, st = R.render(p)
                refine = st["misses"] > 0 or R.cache.busy()
            MON.frame(st["gpu_ms"])
            # stop re-rendering an unchanged view that no longer gains bricks (e.g. it needs more than the pool)
            stalled = stalled + 1 if version == done_version and st["uploads"] == 0 and not R.cache.loading() else 0
            refine = refine and stalled < 30
            last = time.time()
            done_version = version
            self.sent += 1
            st.update(n=self.sent, seq=seq, refining=refine, mem=memory() if self.sent % 10 == 1 else None)
            head = json.dumps(st).encode()
            try:
                with self.wlock:
                    ws_send(self.h.wfile, struct.pack("<I", len(head)) + head + data)
            except OSError:
                with self.cond:
                    self.alive = False
                return


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body, ctype, extra=None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(open(os.path.join(WEB, "index.html"), "rb").read(), "text/html; charset=utf-8")
        if u.path in ("/favicon.png", "/favicon.ico"):
            return self._send(open(LOGO, "rb").read(), "image/png")
        if u.path == "/info":
            return self._send(json.dumps(dict(R.meta, memory=memory())).encode(), "application/json")
        if u.path == "/lut":                                 # colour table of a map, for the page's preview bar
            p = parse({k: v[0] for k, v in parse_qs(u.query).items()})
            t = colormaps.lut(p["colour"], p["gamma"], p["c0"], p["c1"])
            return self._send(json.dumps(dict(names=colormaps.NAMES, rgb=(t * 255).round().astype(int).tolist())).encode(),
                              "application/json")
        if u.path == "/stats":
            since = float(parse_qs(u.query).get("since", ["0"])[0])
            snap = MON.snapshot(since)
            snap.update(pool_gb=R.cache.atlas.numel() / 2**30, host_cap_gb=R.cache.host_cap / 2**30)
            return self._send(json.dumps(snap).encode(), "application/json")
        if u.path == "/render":
            p = parse({k: v[0] for k, v in parse_qs(u.query).items()})
            with LOCK:
                data, st = R.render(p)
            MON.frame(st["gpu_ms"])
            return self._send(data, "image/jpeg", {"X-Render-ms": str(st["frame_ms"]), "X-Stats": json.dumps(st),
                                                   "Access-Control-Expose-Headers": "X-Render-ms, X-Stats"})
        if u.path == "/ws" and self.headers.get("Upgrade", "").lower() == "websocket":
            key = self.headers["Sec-WebSocket-Key"]
            accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
            self.send_response(101)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept.decode())
            self.end_headers()
            self.wfile.flush()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            v = Viewer(self)
            threading.Thread(target=v.read_loop, daemon=True).start()
            v.send_loop()
            self.close_connection = True
            return
        self.send_error(404)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Open the Mind viewer server. GPU memory is the fixed brick pool plus "
                                 "about 0.9 GB (CUDA context, kernels, frame buffers, page table); it never grows with the data.")
    ap.add_argument("root", help="pyramid directory, e.g. data/bigbrain20_png.ome.zarr")
    ap.add_argument("port", nargs="?", type=int, default=1726)
    ap.add_argument("host", nargs="?", default="0.0.0.0", help="0.0.0.0 inside Docker, so the port is reachable outside")
    ap.add_argument("--pool-gb", type=float, default=4.0, help="GPU brick pool in GB (256 KB per brick); default 4")
    ap.add_argument("--host-gb", type=float, default=8.0, help="host RAM cache of compressed bricks in GB; default 8")
    ap.add_argument("--finest", type=int, default=0, help="finest pyramid level ever used: 0 = native resolution, "
                    "1 = half (L0 is never read), 2 = a quarter ...; default 0")
    ap.add_argument("--staging-mb", type=float, default=256, help="pinned host memory for uploads in MB; default 256")
    ap.add_argument("--threads", type=int, default=64, help="brick loader threads; default 64")
    ap.add_argument("--raw", help="read level 0 straight from this uncompressed raw volume (uint8, C order k j i) "
                    "instead of the pyramid; coarser levels still come from the pyramid")
    ap.add_argument("--pin-share", type=float, default=0.25,
                    help="share of the pool for the coarse levels loaded at start and never evicted; default 0.25")
    a = ap.parse_args()
    root, port, host, host_gb = a.root, a.port, a.host, a.host_gb
    t = time.time()
    R = Renderer(root, pool_gb=a.pool_gb, host_gb=a.host_gb, finest=a.finest, threads=a.threads,
                 stage_bricks=max(1, int(a.staging_mb * 2**20) // 262144), pin_share=a.pin_share, raw=a.raw)
    with LOCK:
        R.render(parse({}))                                  # compile the kernel before the first viewer
    MON = Monitor()
    m = memory()
    print(f"{R.meta['name']}: {R.meta['levels']} levels, GPU pool {m['pool_gb']:.1f} GB ({m['pool_slots']} bricks, "
          f"{m['pinned']} pinned: L{m['pinned_from_level']}+), finest level L{m['finest_level']}, host cache {host_gb:.1f} GB, "
          f"staging {m['staging_gb'] * 1024:.0f} MB, {m['loader_threads']} loader threads; ready in {time.time() - t:.1f}s",
          flush=True)
    print(f"serving http://{host}:{port}/", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
