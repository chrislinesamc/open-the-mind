"""Build the brick pyramid (bricks.py) as an OME-Zarr 0.5 group (omezarr.py) from a raw volume or a stack of
section images (PNG, one file per section; config keys png_dir, png_pattern, png_first, invert), slab by slab.

One slab is 16 sections, one brick row. A reader thread fetches the next slab with parallel pread
while the GPU downsamples the current one and a thread pool compresses its bricks. The isotropic levels
at the end are built from the finished previous level, two brick rows at a time. The volume is never
held: memory is bounded by two slabs, whatever the number of sections.

With --mem-gb the slabs are cut into strips of image rows sized so that host RAM and GPU memory both stay
within the budget, whatever the image size; each strip holds whole bricks at every level, and a shard is
written as soon as its bricks are all in. Section images are then decoded once per strip.

usage: python pyramid/build_pyramid.py configs/bigbrain20_png.json data/bigbrain20_png.ome.zarr [--mem-gb 2]
       --mem-gb caps host RAM and GPU memory (each) at about this many GB; default: whole slabs
       --codec none stores the bricks uncompressed (to measure streaming without decompression)
       --keep-empty stores background bricks too: a dense pyramid, as large as the raw volume
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from queue import Queue

import imagecodecs
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from bricks import BRICK, encode, level_plan  # noqa: E402
from omezarr import ZarrLevelWriter, ZarrPyramid, write_group  # noqa: E402

DEV = torch.device("cuda")
POOL = ThreadPoolExecutor(64)
HOST = [None]                                       # pinned buffer for bricks coming back from the GPU (sized by build())
BASE_GB = 0.8                                       # host RAM of PyTorch + CUDA before any data (measured ~0.75 GB)
STRIP_SHARE = 5                                     # a strip's size is held about this many times at once (host or GPU)


def strip_rows(budget, row_bytes, unit, total):
    """Image rows per strip: as many as fit in `budget` bytes at `row_bytes` per row, a multiple of `unit`,
    at least one unit, and no more than `total` rounded up to the unit (all rows when budget is None).
    PyTorch rounds pinned buffers up to a power of two (a 148 MB strip takes 256 MB), so the budget is
    rounded down to one first."""
    whole = -(-total // unit) * unit
    if budget is None:
        return whole
    budget = 2 ** math.floor(math.log2(max(budget, 1)))
    return min(whole, max(unit, int(budget // row_bytes) // unit * unit))


def to_host(g):
    """GPU tensor -> numpy view of a pinned host buffer. Pinned copies run at full speed; a plain .cpu()
    into pageable memory took 1.3 s per 684 MB BigBrain slab on the GB10."""
    if HOST[0] is None or HOST[0].numel() < g.numel():
        HOST[0] = torch.empty(g.numel(), dtype=torch.uint8, pin_memory=True)
    h = HOST[0][:g.numel()].view(g.shape)
    h.copy_(g)
    return h.numpy()


def read_slabs(raw, shape, sh, q, free, chunk=64 << 20):
    """Reader thread: fill a free pinned strip (16 sections x sh rows) with parallel preadv and put
    (gz, first row, strip) for every strip of every brick row. Any error (short file, I/O) is put on the
    queue so the main thread raises it instead of waiting."""
    try:
        nk, nj, ni = shape
        sec = nj * ni
        fd = os.open(raw, os.O_RDONLY)
        for gz in range(-(-nk // BRICK[0])):
            k0, k1 = gz * BRICK[0], min((gz + 1) * BRICK[0], nk)
            for j0 in range(0, nj, sh):
                j1 = min(j0 + sh, nj)
                buf = free.get()
                a = buf.numpy()
                mv = memoryview(a.reshape(-1))
                n = (j1 - j0) * ni                   # bytes of one section in this strip, contiguous in the file

                def job(t):
                    k, o = t
                    m, dst = min(chunk, n - o), (k - k0) * sh * ni + o
                    got = 0
                    while got < m:
                        r = os.preadv(fd, [mv[dst + got:dst + m]], k * sec + j0 * ni + o + got)
                        if r == 0:
                            raise EOFError(f"{raw} ends before shape {shape}")
                        got += r
                list(POOL.map(job, [(k, o) for k in range(k0, k1) for o in range(0, n, chunk)]))
                a[k1 - k0:] = 0                      # short last slab or strip: zero the stale tail
                a[:, j1 - j0:] = 0
                q.put((gz, j0, buf))
        q.put(None)
    except Exception as e:
        q.put(e)


def read_png_slabs(cfg, sh, q, free, workers):
    """Reader thread for a stack of section images (one file per k section, e.g. BigBrain's coronal PNGs): the
    sections of a strip are decoded by `workers` threads (each holds one decoded image) and their rows j0..j0+sh
    copied into a free pinned strip. With "invert", values become 255 - v, so a white background is 0 and its
    bricks are not stored. Errors go on the queue, as in read_slabs."""
    try:
        nk, nj, ni = cfg["shape_kji"]
        first, inv = cfg.get("png_first", 0), cfg.get("invert", False)
        with ThreadPoolExecutor(workers) as dec:
            for gz in range(-(-nk // BRICK[0])):
                k0, k1 = gz * BRICK[0], min((gz + 1) * BRICK[0], nk)
                for j0 in range(0, nj, sh):
                    j1 = min(j0 + sh, nj)
                    buf = free.get()
                    a = buf.numpy()

                    def job(k):
                        path = os.path.join(cfg["png_dir"], cfg["png_pattern"].format(k + first))
                        img = imagecodecs.png_decode(open(path, "rb").read())
                        if img.shape != (nj, ni) or img.dtype != np.uint8:
                            raise ValueError(f"{path}: {img.shape} {img.dtype}, expected ({nj}, {ni}) uint8")
                        if inv:
                            np.subtract(255, img[j0:j1], out=a[k - k0, :j1 - j0])
                        else:
                            a[k - k0, :j1 - j0] = img[j0:j1]
                    list(dec.map(job, range(k0, k1)))
                    a[k1 - k0:] = 0                      # short last slab or strip: zero the stale tail
                    a[:, j1 - j0:] = 0
                    q.put((gz, j0, buf))
        q.put(None)
    except Exception as e:
        q.put(e)


def pool(x, f, valid):
    """Block mean with factors f = (fz, fy, fx) of a padded uint8 GPU array whose data occupies
    valid = (z, y, x) from the origin. A block cut by the data edge averages only its valid voxels:
    the last valid slice is copied into the padding first (exact for factors of 2)."""
    for axis, (fa, v) in enumerate(zip(f, valid)):
        if fa == 2 and v % 2 == 1 and v < x.shape[axis]:
            x.select(axis, v).copy_(x.select(axis, v - 1))
    z, y, x_ = x.shape                               # int16 holds a sum of 8 uint8 values, at half int32's memory
    s = x.view(z // f[0], f[0], y // f[1], f[1], x_ // f[2], f[2]).sum((1, 3, 5), dtype=torch.int16)
    n = f[0] * f[1] * f[2]
    return ((s + n // 2) // n).to(torch.uint8)


def fit(x, shape):
    """Crop or zero-pad a (z, y, x) GPU array to `shape` (a brick row, or a strip of one)."""
    if tuple(x.shape) == shape:
        return x.contiguous()
    out = torch.zeros(shape, dtype=torch.uint8, device=x.device)
    z, y, x_ = (min(a, b) for a, b in zip(x.shape, shape))
    out[:z, :y, :x_] = x[:z, :y, :x_]
    return out


def emit(writer, gz, x, stats, codec="zstd", keep_empty=False, gy0=0):
    """Split one padded strip of a brick row, starting at brick row gy0, into bricks, compress the non-empty
    ones (all of them with keep_empty) and write them; the shards they complete are written out."""
    gy, gx = x.shape[1] // BRICK[1], x.shape[2] // BRICK[2]
    bricks = x.view(BRICK[0], gy, BRICK[1], gx, BRICK[2]).permute(1, 3, 0, 2, 4)
    vmax = bricks.amax((2, 3, 4))
    yy, xx = torch.nonzero(vmax >= 0 if keep_empty else vmax > 0, as_tuple=True)
    sel = to_host(bricks[yy, xx].contiguous())                # only the non-empty bricks
    blobs = list(POOL.map(partial(encode, codec=codec), sel))  # copies: the pinned buffer is reused next call
    vm = vmax[yy, xx].cpu().numpy()
    for y, x_, b, m in zip(yy.tolist(), xx.tolist(), blobs, vm):
        writer.write(gz, gy0 + y, x_, b, m)
    writer.flush(gy0 + gy)
    stats["stored"] += len(blobs)
    stats["raw"] += sel.nbytes
    stats["bytes"] += sum(map(len, blobs))


def build(cfg_path, out, codec="zstd", keep_empty=False, mem_gb=None):
    cfg = json.load(open(cfg_path))
    levels = level_plan(cfg["shape_kji"], cfg["spacing_mm_kji"])
    aniso = [n for n, L in enumerate(levels) if n == 0 or L["factor"] == [1, 2, 2]]
    os.makedirs(out, exist_ok=True)
    writers = [ZarrLevelWriter(out, n, L, codec) for n, L in enumerate(levels)]
    stats = [dict(stored=0, raw=0, bytes=0) for _ in levels]
    t0 = time.time()

    nk, nj, ni = cfg["shape_kji"]
    g0 = levels[0]["grid"]
    png = "png_dir" in cfg
    budget = None if mem_gb is None else max(0.0, mem_gb - BASE_GB) * 2**30
    workers = 16 if budget is None else max(1, min(16, int(0.15 * budget // (nj * ni))))
    strip = None if budget is None else (budget - (workers * nj * ni if png else 0)) / STRIP_SHARE
    unit = BRICK[1] << (len(aniso) - 1)              # whole bricks at every anisotropic level
    sh = strip_rows(strip, BRICK[0] * g0[2] * BRICK[2], unit, nj)
    if mem_gb is not None:
        print(f"memory budget {mem_gb} GB: strips of {sh} rows ({-(-nj // sh)} per slab)"
              + (f", {workers} image decoders" if png else ""), flush=True)

    q, free = Queue(), Queue()
    for _ in range(2):                               # two pinned strips: one being read, one on its way to the GPU
        free.put(torch.empty((BRICK[0], sh, ni), dtype=torch.uint8, pin_memory=True))
    HOST[0] = torch.empty(BRICK[0] * sh * g0[2] * BRICK[2], dtype=torch.uint8, pin_memory=True)   # sized once: no regrowth
    reader = (read_png_slabs, (cfg, sh, q, free, workers)) if png else (read_slabs, (cfg["raw"], cfg["shape_kji"], sh, q, free))
    threading.Thread(target=reader[0], args=reader[1], daemon=True).start()
    while (item := q.get()) is not None:
        if isinstance(item, Exception):
            raise item
        gz, j0, buf = item
        x = fit(buf.to(DEV), (BRICK[0], min(sh, g0[1] * BRICK[1] - j0), g0[2] * BRICK[2]))
        free.put(buf)                                # blocking copy done: the reader may refill buf
        nz = min(BRICK[0], nk - gz * BRICK[0])
        for n in aniso:
            if n:
                prev, G = levels[n - 1]["shape"], levels[n]["grid"]
                v = min(max(prev[1] - (j0 >> (n - 1)), 0), x.shape[1])
                x = fit(pool(x, (1, 2, 2), (nz, v, prev[2])),
                        (BRICK[0], min(sh >> n, G[1] * BRICK[1] - (j0 >> n)), G[2] * BRICK[2]))
            emit(writers[n], gz, x, stats[n], codec, keep_empty, (j0 >> n) // BRICK[1])
        if j0 + sh >= nj and ((gz + 1) % max(1, g0[0] // 10) == 0 or gz + 1 == g0[0]):   # about 10 progress lines
            print(f"slab {gz + 1}/{g0[0]}  {time.time() - t0:6.1f}s  "
                  f"L0 stored {stats[0]['stored']} bricks, {stats[0]['bytes'] / 1e9:.2f} GB", flush=True)
    del free, x
    torch.cuda.empty_cache()

    for n in range(aniso[-1] + 1, len(levels)):      # isotropic levels from the finished level n-1
        writers[n - 1].close()
        prev = ZarrPyramid(out, levels=[n - 1], plan=levels, codec=codec)
        pz, py, px = levels[n - 1]["shape"]
        gp, G = levels[n - 1]["grid"], levels[n]["grid"]
        shp = strip_rows(strip, 2 * BRICK[0] * gp[2] * BRICK[2], 2 * BRICK[1], py)
        for gz in range(G[0]):
            for y0 in range(0, gp[1] * BRICK[1], shp):
                x = torch.from_numpy(prev.rows(n - 1, 2 * gz, 2 * gz + 2, y0 // BRICK[1], (y0 + shp) // BRICK[1])).to(DEV)
                valid = (min(max(pz - 2 * gz * BRICK[0], 0), 2 * BRICK[0]), min(max(py - y0, 0), x.shape[1]), px)
                x = fit(pool(x, (2, 2, 2), valid), (BRICK[0], min(shp // 2, G[1] * BRICK[1] - y0 // 2), G[2] * BRICK[2]))
                emit(writers[n], gz, x, stats[n], codec, keep_empty, y0 // 2 // BRICK[1])
    writers[-1].close()
    for w in writers[:aniso[-1]]:
        w.close()

    for L, s in zip(levels, stats):
        L.update(stored_bricks=s["stored"], total_bricks=int(np.prod(L["grid"])), bytes=s["bytes"],
                 ratio_stored=round(s["raw"] / max(s["bytes"], 1), 3))
    meta = dict(format="ome-zarr-0.5 brick pyramid", name=cfg["name"], source=cfg.get("source", cfg.get("raw", cfg.get("png_dir"))),
                axes=cfg["axes"], planes=cfg.get("planes"), dtype="uint8",
                brick_zyx=list(BRICK), codec="zstd-1" if codec == "zstd" else "none", levels=levels, build_seconds=round(time.time() - t0, 1))
    write_group(out, meta)
    total = sum(s["bytes"] for s in stats)
    src = np.prod(cfg["shape_kji"])
    print(f"done in {time.time() - t0:.0f}s: {total / 1e9:.2f} GB for all levels, source {src / 1e9:.1f} GB "
          f"({src / total:.1f}x smaller)")
    for n, L in enumerate(levels):
        print(f"  L{n} shape {L['shape']} grid {L['grid']} stored {L['stored_bricks']}/{L['total_bricks']} "
              f"{L['bytes'] / 1e9:.3f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("out")
    ap.add_argument("--codec", choices=["zstd", "none"], default="zstd")
    ap.add_argument("--keep-empty", action="store_true", help="store background bricks too (a dense pyramid)")
    ap.add_argument("--mem-gb", type=float, help="cap host RAM and GPU memory (each) at about this many GB; default: whole slabs")
    a = ap.parse_args()
    build(a.config, a.out, a.codec, a.keep_empty, a.mem_gb)
