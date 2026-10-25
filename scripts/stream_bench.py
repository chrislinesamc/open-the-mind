"""Streaming benchmark: the same 30 s camera flight (1800 frames paced at 60 fps) through a pyramid with a fixed
GPU pool, as a viewer would drive it: orbit the whole brain, zoom into the opened corner, pan along a cut face
close up, sweep a cut plane. Frames are not held back for missing bricks, so loading has to keep up.

Per frame: frame and kernel time, bricks missing. Twice a second (server/monitor.py): the process's GPU memory,
GPU utilisation, system RAM, process RSS. Also CPU cores used and bytes read from disk. Comparing datasets of
different size with the same pool and flight shows whether memory and compute stay flat as the data grows.

usage: python scripts/stream_bench.py <pyramid> [--pool-gb 20] [--host-gb 8] [--skip 1] [--w 1920 --h 1080]
       [--frames 1800] [--fps 60] [--out result.json]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from monitor import Monitor  # noqa: E402
from renderer import Renderer, parse  # noqa: E402


def flight(i, n, R):
    """View at frame i of n: four phases of equal length."""
    q, f = divmod(4 * i, n)
    t = f / n                                       # 0..1 within the phase
    d0 = R.meta["default_dist"]
    base = dict(mode=1, cx=0.5, cy=0.5, cz=0.5)
    if q == 0:                                      # orbit the whole opened brain
        return dict(base, az=35 + 360 * t, el=25, dist=d0)
    if q == 1:                                      # zoom into the opened corner, 1/12 of the distance
        return dict(base, az=35, el=25, dist=d0 * math.exp(math.log(1 / 12) * t))
    half = (R.hi0 - R.lo0) / 2
    if q == 2:                                      # pan along the sagittal cut face, close up
        return dict(base, az=35, el=25, dist=d0 / 12, tx=float(half[0] * 0.5 * math.sin(2 * math.pi * t)),
                    ty=float(half[1] * 0.3 * (t - 0.5)))
    return dict(base, az=35, el=25, dist=d0 / 6, cx=0.3 + 0.4 * t)   # sweep the coronal plane through the tissue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--pool-gb", type=float, default=20)
    ap.add_argument("--host-gb", type=float, default=8)
    ap.add_argument("--skip", type=int, default=1, help="empty-space skipping on (1) or off (0)")
    ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--frames", type=int, default=1800); ap.add_argument("--fps", type=float, default=60)
    ap.add_argument("--out")
    a = ap.parse_args()

    R = Renderer(a.root, pool_gb=a.pool_gb, host_gb=a.host_gb)
    mon = Monitor(period=0.5, keep=100000)
    R.render(parse(dict(w=a.w, h=a.h)))                        # compile before timing
    cpu0, t0 = os.times(), time.time()
    disk0 = R.cache.stats["disk_bytes"]
    frames = []
    for i in range(a.frames):
        due = t0 + i / a.fps
        if time.time() < due:
            time.sleep(due - time.time())
        p = parse(dict(flight(i, a.frames, R), w=a.w, h=a.h, skip=a.skip, box=0))
        data, st = R.render(p)
        mon.frame(st["gpu_ms"])
        frames.append((time.time() - t0, st["frame_ms"], st["gpu_ms"], st["misses"], st["uploads"], len(data)))
    wall = time.time() - t0
    cpu1 = os.times()
    time.sleep(0.6)                                             # one more monitor sample
    s = mon.snapshot()["samples"]
    s = [x for x in s if x["t"] >= t0]
    f = np.array(frames)
    m = R.cache.memory()
    res = dict(
        pyramid=a.root, pool_gb=a.pool_gb, host_gb=a.host_gb, skip=a.skip, size=[a.w, a.h], frames=a.frames,
        target_fps=a.fps, achieved_fps=round(a.frames / wall, 1),
        frame_ms_p50=round(float(np.percentile(f[:, 1], 50)), 2), frame_ms_p95=round(float(np.percentile(f[:, 1], 95)), 2),
        kernel_ms_p50=round(float(np.percentile(f[:, 2], 50)), 2), kernel_ms_p95=round(float(np.percentile(f[:, 2], 95)), 2),
        sharp_frames_pct=round(100 * float((f[:, 3] == 0).mean()), 1),
        gpu_util_mean=round(float(np.mean([x["gpu_util"] for x in s if x["gpu_util"] is not None])), 1),
        gpu_proc_gb_max=round(max(x["gpu_proc_gb"] for x in s), 2),
        gpu_alloc_gb_max=round(max(x["gpu_alloc_gb"] for x in s), 2),
        ram_used_gb_max=round(max(x["ram_used_gb"] for x in s), 1),
        rss_gb_max=round(max(x["rss_gb"] for x in s), 2),
        cpu_cores_used=round(((cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)) / wall, 2),
        disk_read_gb=round((m["disk_bytes"] - disk0) / 2**30, 2), bricks_loaded=m["loaded"], bricks_evicted=m["evicted"],
        jpeg_kb_mean=round(float(f[:, 5].mean()) / 1024, 0))
    for k, v in res.items():
        print(f"{k:18s} {v}")
    if a.out:
        json.dump(dict(res, timeline=frames, samples=s), open(a.out, "w"))


if __name__ == "__main__":
    main()
