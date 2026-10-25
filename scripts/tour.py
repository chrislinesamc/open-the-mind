"""Memory and sharpening tour: render many random zoomed cut-face views, far more bricks than the
pool holds, and log per view the time until the view is sharp and the GPU/host memory. Memory must
stay flat once the pool is full; only evictions keep growing.

usage: python scripts/tour.py data/bigbrain20_png.ome.zarr [views] [pool_gb] [host_gb] [out.csv]
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from renderer import Renderer, parse  # noqa: E402

root = sys.argv[1]
n_views = int(sys.argv[2]) if len(sys.argv) > 2 else 200
pool_gb = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
host_gb = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0
out = sys.argv[5] if len(sys.argv) > 5 else None
R = Renderer(root, pool_gb=pool_gb, host_gb=host_gb)
rng = np.random.default_rng(0)
rows = []
half = (R.hi0 - R.lo0) / 2
for i in range(n_views):
    axis = rng.integers(3)
    # look at the cut face from the removed side: x cut from -x, y cut from -y (above), z cut from -z
    v = dict(w=1600, h=1000, dist=float(rng.uniform(3, 15)), az=[-90.0, 0.0, 180.0][axis], el=89.0 if axis == 1 else 0.0,
             tx=float(rng.uniform(-0.6, 0.6) * half[0]), ty=float(rng.uniform(-0.6, 0.6) * half[1]),
             tz=float(rng.uniform(-0.6, 0.6) * half[2]))
    v[("cx", "cy", "cz")[axis]] = float(rng.uniform(0.3, 0.7))
    p = parse(v)
    t = time.time()
    frames = 0
    while True:
        data, st = R.render(p)
        frames += 1
        if (st["misses"] == 0 and not R.cache.busy()) or time.time() - t > 10:
            break
    m = R.cache.memory()
    rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    rows.append((i, time.time() - t, frames, st["frame_ms"], torch.cuda.memory_allocated() / 2**30,
                 torch.cuda.max_memory_allocated() / 2**30, m["resident"], m["evicted"], m["loaded"],
                 m["host_cache_gb"], rss, m["disk_bytes"] / 2**30))
    if i % 20 == 0 or i == n_views - 1:
        r = rows[-1]
        print(f"view {i:4d}: sharp {r[1]:5.2f}s ({r[2]:3d} frames) | GPU {r[4]:.2f} GB (peak {r[5]:.2f}) | resident {r[6]} "
              f"| evicted {r[7]} | loaded {r[8]} | host cache {r[9]:.2f} GB | RSS {r[10]:.2f} GB | disk {r[11]:.1f} GB", flush=True)
a = np.array(rows)
print(f"pool {pool_gb} GB: sharpen median {np.median(a[:, 1]):.2f}s p90 {np.percentile(a[:, 1], 90):.2f}s; "
      f"GPU allocated min/max {a[:, 4].min():.2f}/{a[:, 4].max():.2f} GB; RSS min/max {a[20:, 10].min():.2f}/{a[20:, 10].max():.2f} GB")
if out:
    np.savetxt(out, a, delimiter=",", fmt="%.4f",
               header="view,sharpen_s,frames,frame_ms,gpu_alloc_gb,gpu_peak_gb,resident,evicted,loaded,host_cache_gb,rss_gb,disk_gb")
