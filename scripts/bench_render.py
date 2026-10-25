"""Frame-time benchmark of the bounded renderer on a few views (whole brain, cuts, zoom on a cut face).
Each view is first rendered until its bricks are resident, then timed in steady state.

usage: python scripts/bench_render.py data/bigbrain20_png.ome.zarr [W H] [out_dir]
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from renderer import Renderer, parse  # noqa: E402

root = sys.argv[1]
W, H = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (1920, 1080)
out_dir = sys.argv[4] if len(sys.argv) > 4 else None
t = time.time()
R = Renderer(root)
print(f"renderer ready in {time.time() - t:.1f}s: pool {R.cache.S} slots, pinned levels >= L{R.cache.pinned_from} "
      f"({R.cache.n_pinned} bricks)")

VIEWS = {
    "overview": dict(),
    "slab cut x=0.5": dict(cx=0.5),
    "corner cutaway": dict(mode=1, cx=0.5, cy=0.5, cz=0.5),
    "zoom on cut face 1:1": dict(cz=0.5, az=180, el=0, dist=8.0),
    "zoom x4 on cut face": dict(cz=0.5, az=180, el=0, dist=2.0, tx=-5.0, ty=3.0),
    "oblique zoom, volume": dict(cx=0.45, az=-70, el=30, dist=20.0, face=0.0, dens=1.0),
    "camera cut": dict(cam=1, camd=0.0, az=60, el=20),
}
for name, v in VIEWS.items():
    p = parse(dict(v, w=W, h=H))
    t = time.time()
    n = 0
    while True:                          # warm: until no misses and nothing in flight
        data, st = R.render(p)
        n += 1
        if st["misses"] == 0 and not R.cache.busy():
            break
        time.sleep(0.005)
    warm = time.time() - t
    gpu, frame = [], []
    for _ in range(30):
        data, st = R.render(p)
        gpu.append(st["gpu_ms"])
        frame.append(st["frame_ms"])
    print(f"{name:22s} sharp after {warm:5.2f}s ({n:3d} frames) | kernel {np.median(gpu):6.2f} ms | "
          f"frame incl. JPEG {np.median(frame):6.2f} ms ({1000 / np.median(frame):5.0f} fps) | {len(data) / 1024:5.0f} KB")
    if out_dir:
        open(os.path.join(out_dir, name.replace(" ", "_").replace("=", "") + ".jpg"), "wb").write(data)
m = R.cache.memory()
print(f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB (pool {m['pool_gb']:.1f} GB), resident {m['resident']}, "
      f"host cache {m['host_cache_gb']:.2f} GB, loaded {m['loaded']}, disk {m['disk_bytes'] / 2**30:.2f} GB")
