"""The brick pyramid as a standard OME-Zarr 0.5 group (Zarr v3, sharded).

    zarr.json           group: OME multiscales (axes in micrometres, scale and translation per level),
                        plus attributes["pyramid"]: the level plan (shape, spacing, factor, brick grid per
                        level) and build stats
    <n>/zarr.json       level n: uint8 array, inner chunk = one 16 x 128 x 128 brick, zstd-1 (or uncompressed
                        with codec "none", to measure streaming without decompression)
    <n>/c/<z>/<y>/<x>   one shard = 1 x 16 x 16 bricks (16 x 2048 x 2048 voxels): the compressed bricks,
                        then the shard index, (offset, nbytes) uint64 per brick in C order, at the end.
                        Empty bricks are "missing" (2^64 - 1) and read as the fill value 0; shards with
                        no data are not written
    brick_max/<n>       per brick max (uint8, one chunk), for empty-space skipping; other OME-Zarr
                        readers ignore it

Every brick is a plain zstd frame of a C-order brick, so the viewer keeps a one-pread-per-brick path:
the shard indexes are read once at start into a per-brick (offset, size) table (ZarrPyramid). Any
Zarr v3 / OME-Zarr reader (zarr-python, napari, Neuroglancer) can open the group.

Voxel positions: level voxels are cell-centred block means, so level n is shifted by half a level-0
voxel less than half its own voxel: translation = (spacing_n - spacing_0) / 2.
"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import imagecodecs
import numpy as np

from bricks import BRICK, IDX, Pyramid, RawLevel0

SHARD = (16, 2048, 2048)
PER = tuple(s // b for s, b in zip(SHARD, BRICK))       # bricks per shard: (1, 16, 16)
NCH = int(np.prod(PER))
MISSING = 2**64 - 1
assert PER[0] == 1, "writers flush one brick row at a time"
WRITE_POOL = ThreadPoolExecutor(16)                     # the shards of a row are written in parallel


def shard_grid(grid):
    return [-(-g // p) for g, p in zip(grid, PER)]


def shard_path(root, n, sz, sy, sx):
    return os.path.join(root, str(n), "c", str(sz), str(sy), str(sx))


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def write_shard(path, blobs):
    """One shard file from NCH compressed bricks in C order (None = empty). Returns bytes written."""
    if all(b is None for b in blobs):
        return 0
    idx = np.full((NCH, 2), MISSING, "<u8")
    off = 0
    for i, b in enumerate(blobs):
        if b is not None:
            idx[i] = off, len(b)
            off += len(b)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.writelines(b for b in blobs if b is not None)
        f.write(idx.tobytes())
    return off + idx.nbytes


def array_meta(shape, chunks, sharded, codec="zstd"):
    zstd = [{"name": "bytes"}] + ([{"name": "zstd", "configuration": {"level": 1, "checksum": False}}]
                                  if codec == "zstd" else [])
    codecs = [{"name": "sharding_indexed", "configuration": {
        "chunk_shape": list(BRICK), "codecs": zstd,
        "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "index_location": "end"}}] if sharded else zstd
    return {"zarr_format": 3, "node_type": "array", "shape": list(shape), "data_type": "uint8",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunks)}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0, "codecs": codecs, "dimension_names": ["z", "y", "x"], "attributes": {}}


def write_level_meta(root, n, L, vmax, codec="zstd"):
    """zarr.json of level n and its brick_max array (always zstd: it is tiny)."""
    write_json(os.path.join(root, str(n), "zarr.json"), array_meta(L["shape"], SHARD, True, codec))
    write_json(os.path.join(root, "brick_max", str(n), "zarr.json"), array_meta(L["grid"], L["grid"], False))
    f = os.path.join(root, "brick_max", str(n), "c", "0", "0", "0")
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, "wb") as f:
        f.write(imagecodecs.zstd_encode(np.ascontiguousarray(vmax, np.uint8), level=1))


def write_group(root, meta):
    """Root zarr.json: OME-Zarr 0.5 multiscales from the level plan, the plan itself under "pyramid"."""
    s0 = np.array(meta["levels"][0]["spacing"]) * 1000
    datasets = []
    for n, L in enumerate(meta["levels"]):
        s = np.array(L["spacing"]) * 1000
        ct = [{"type": "scale", "scale": [round(float(v), 6) for v in s]}]
        if n:
            ct.append({"type": "translation", "translation": [round(float(v), 6) for v in (s - s0) / 2]})
        datasets.append({"path": str(n), "coordinateTransformations": ct})
    ome = {"version": "0.5", "multiscales": [{
        "name": meta["name"], "type": "mean", "datasets": datasets,
        "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"]}]}
    write_json(os.path.join(root, "brick_max", "zarr.json"),
               {"zarr_format": 3, "node_type": "group", "attributes": {}})
    write_json(os.path.join(root, "zarr.json"),
               {"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome, "pyramid": meta}})


def open_pyramid(root, raw=None, **kw):
    """Reader for a pyramid; with `raw`, level 0 is read from the uncompressed source volume instead."""
    pyr = ZarrPyramid(root, **kw)
    return RawLevel0(pyr, raw) if raw else pyr


class ZarrLevelWriter:
    """Writes one level. Bricks arrive in (z, y, x) order; a shard is one brick row deep and 16 bricks
    along y, so a shard is written out as soon as the bricks above a given y have all arrived."""

    def __init__(self, root, n, L, codec="zstd"):
        self.root, self.n, self.L, self.codec = root, n, L, codec
        self.vmax = np.zeros(L["grid"], np.uint8)
        self.row, self.blobs, self.sy_done = None, {}, 0

    def write(self, gz, gy, gx, blob, vmax):
        if gz != self.row:
            self.flush()
            self.row = gz
        self.blobs[gy, gx] = blob
        self.vmax[gz, gy, gx] = vmax

    def flush(self, gy_end=None):
        """Write the shards of the current row whose bricks have all arrived: those wholly above brick row
        gy_end, or all of them (gy_end None, or past the last row)."""
        if self.row is None:
            return
        _, sgy, sgx = shard_grid(self.L["grid"])
        sy_end = sgy if gy_end is None or gy_end >= self.L["grid"][1] else gy_end // PER[1]

        def one(s):
            sy, sx = s
            blobs = [self.blobs.pop((sy * PER[1] + y, sx * PER[2] + x), None) for y in range(PER[1]) for x in range(PER[2])]
            write_shard(shard_path(self.root, self.n, self.row, sy, sx), blobs)
        list(WRITE_POOL.map(one, [(sy, sx) for sy in range(self.sy_done, sy_end) for sx in range(sgx)]))
        self.sy_done = sy_end
        if sy_end == sgy:
            self.row, self.blobs, self.sy_done = None, {}, 0

    def close(self):
        self.flush()
        write_level_meta(self.root, self.n, self.L, self.vmax, self.codec)


class ZarrPyramid(Pyramid):
    """Random access to the bricks of an OME-Zarr pyramid written by this project. `plan` gives the
    level plan while the pyramid is still being built (before the root zarr.json exists)."""

    def __init__(self, root, levels=None, plan=None, codec=None):
        self.root = root
        meta = os.path.join(root, "zarr.json")         # absent while the pyramid is being built
        self.meta = json.load(open(meta))["attributes"]["pyramid"] if os.path.exists(meta) else None
        self.codec = codec or ("none" if self.meta and self.meta.get("codec") == "none" else "zstd")
        if plan is None:
            plan = self.meta["levels"]
        if levels is None:
            levels = range(len(plan))
        self.idx, self.vmax = {}, {}
        for n in levels:
            self.idx[n] = self._index(n, plan[n]["grid"])
            f = os.path.join(root, "brick_max", str(n), "c", "0", "0", "0")
            self.vmax[n] = np.frombuffer(imagecodecs.zstd_decode(open(f, "rb").read()),
                                         np.uint8).reshape(plan[n]["grid"])
        self.local = threading.local()

    def _index(self, n, grid):
        """Per-brick (offset in its shard, size) from the shard indexes; size 0 = empty."""
        sg = shard_grid(grid)
        idx = np.zeros([s * p for s, p in zip(sg, PER)], IDX)
        for sz, sy, sx in np.ndindex(*sg):
            p = shard_path(self.root, n, sz, sy, sx)
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                f.seek(-NCH * 16, os.SEEK_END)
                a = np.frombuffer(f.read(NCH * 16), "<u8").reshape(*PER, 2)
            sl = tuple(slice(s * q, (s + 1) * q) for s, q in zip((sz, sy, sx), PER))
            stored = a[..., 0] != MISSING
            idx["off"][sl] = np.where(stored, a[..., 0], 0)
            idx["n"][sl] = np.where(stored, a[..., 1], 0)
        return idx[:grid[0], :grid[1], :grid[2]].copy()

    def path(self, n, gz, gy, gx):
        return shard_path(self.root, n, gz // PER[0], gy // PER[1], gx // PER[2])
