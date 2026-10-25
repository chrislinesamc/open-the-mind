"""Bricks and levels of the pyramid, independent of the file layout (that is omezarr.py).

Every level uses the same brick, 16 x 128 x 128 voxels (z, y, x), so one GPU pool slot fits any
level. Level 0 is the source. While the in-plane spacing is finer than the section spacing a level
halves y and x only (anisotropic storage: e.g. 4 -> 8 -> 16 -> 32 -> 64 um in-plane while sections stay 60 um); after
that it halves all three axes. Voxel values are block means. Edge bricks are zero-padded.

Histology at a few um is noisy, so lossless codecs only reach ~1.36x on tissue (zstd-1, decodes at
~700 MB/s per core). The big saving is not storing the ~80% of bricks that are background. Bricks are
small and independent, so a viewer reads only what the screen needs.
"""
import errno
import os
import resource
import threading
from concurrent.futures import ThreadPoolExecutor

import imagecodecs
import numpy as np

BRICK = (16, 128, 128)
IDX = np.dtype([("off", "<u8"), ("n", "<u4")])        # per brick: byte offset in its file, size (0 = empty)
READ_POOL = ThreadPoolExecutor(32)                     # rows(): pread and decode release the GIL


def raise_fd_limit():
    """Raise this process's open-file soft limit to its hard limit and return it. Readers keep one descriptor
    per thread and shard file (thousands of shards, 64+ threads); a terminal's default soft limit of 1024
    runs out within seconds ("[Errno 24] Too many open files")."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY or hard > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            soft = hard
        except (ValueError, OSError):
            pass
    return soft


FD_LIMIT = raise_fd_limit()
# descriptors one thread may keep open: all reader threads together (up to ~128) stay well below the limit
FD_PER_THREAD = max(4, min(1024, (min(FD_LIMIT, 1 << 20) - 512) // 128))


def level_plan(shape, spacing, max_coarse_bricks=8):
    """Shape, spacing (mm), factor (relative to the previous level) and brick grid of every level."""
    levels = [dict(shape=list(shape), spacing=list(spacing), factor=[1, 1, 1])]
    while True:
        prev = levels[-1]
        if np.prod([-(-n // b) for n, b in zip(prev["shape"], BRICK)]) <= max_coarse_bricks:
            break
        f = [1, 2, 2] if prev["spacing"][1] < prev["spacing"][0] else [2, 2, 2]
        levels.append(dict(shape=[-(-n // a) for n, a in zip(prev["shape"], f)],
                           spacing=[s * a for s, a in zip(prev["spacing"], f)], factor=f))
    for L in levels:
        L["grid"] = [-(-n // b) for n, b in zip(L["shape"], BRICK)]
    return levels


def encode(brick, codec="zstd"):
    """Bytes of one brick: zstd level 1, or the plain voxels with codec "none"."""
    if codec == "none":
        return np.ascontiguousarray(brick).tobytes()
    return imagecodecs.zstd_encode(np.ascontiguousarray(brick), level=1)


def decode(blob):
    return np.frombuffer(imagecodecs.zstd_decode(blob), np.uint8).reshape(BRICK)


class Pyramid:
    """Random access to the bricks of a pyramid. A subclass sets meta, idx[n] (per brick offset and size,
    IDX), vmax[n] (per brick max), codec ("zstd" or "none") and local (threading.local()), and maps a brick
    to its file in path().

    Each thread gets its own file descriptors: on Lustre, preads through one shared descriptor are
    serialised by the client (~550 cold random bricks/s), while one descriptor per thread lets them run
    in parallel (~4,300 bricks/s with 128 threads)."""

    def path(self, n, gz, gy, gx):
        """File holding brick (gz, gy, gx) of level n."""
        raise NotImplementedError

    def fd(self, path):
        fds = self.local.__dict__.setdefault("fds", {})
        if path not in fds:
            if len(fds) >= FD_PER_THREAD:        # sharded pyramids have thousands of files: bound fds per thread
                self.close_fds()
            try:
                fds[path] = os.open(path, os.O_RDONLY)
            except OSError as e:                 # limit reached anyway (other code holds descriptors): free ours, retry
                if e.errno != errno.EMFILE:
                    raise
                self.close_fds()
                fds[path] = os.open(path, os.O_RDONLY)
        return fds[path]

    def close_fds(self):
        """Close this thread's descriptors."""
        fds = self.local.__dict__.setdefault("fds", {})
        for f in fds.values():
            os.close(f)
        fds.clear()

    def blob(self, n, gz, gy, gx):
        """Compressed bytes of one brick, or None if the brick is empty."""
        off, size = self.idx[n][gz, gy, gx]
        return os.pread(self.fd(self.path(n, gz, gy, gx)), int(size), int(off)) if size else None

    def decode_into(self, n, blob, out):
        """Decode a blob of level n into a flat uint8 array of one brick."""
        if self.codec == "none":
            out[:] = np.frombuffer(blob, np.uint8)
        else:
            imagecodecs.zstd_decode(blob, out=out)

    def brick(self, n, gz, gy, gx):
        b = self.blob(n, gz, gy, gx)
        out = np.zeros(BRICK, np.uint8)
        if b is not None:
            self.decode_into(n, b, out.reshape(-1))
        return out

    def rows(self, n, gz0, gz1, gy0=0, gy1=None):
        """Brick rows gz0 <= gz < gz1 of level n, only bricks gy0 <= gy < gy1 along y (all by default), as one
        padded array (z, y, x); for building and tests."""
        gz_n, gy_n, gx_n = self.idx[n].shape
        gy1 = gy_n if gy1 is None else min(gy1, gy_n)
        bz, by, bx = BRICK
        out = np.zeros(((gz1 - gz0) * bz, (gy1 - gy0) * by, gx_n * bx), np.uint8)

        def one(g):                                # each brick fills its own block of out: no locking
            gz, gy, gx = g
            b = self.blob(n, gz, gy, gx)
            if b is not None:
                tmp = np.empty(BRICK, np.uint8)
                self.decode_into(n, b, tmp.reshape(-1))
                out[(gz - gz0) * bz:(gz - gz0 + 1) * bz, (gy - gy0) * by:(gy - gy0 + 1) * by, gx * bx:(gx + 1) * bx] = tmp
        cells = [(gz, gy, gx) for gz in range(gz0, min(gz1, gz_n)) for gy in range(gy0, gy1) for gx in range(gx_n)
                 if self.idx[n]["n"][gz, gy, gx]]
        list(READ_POOL.map(one, cells))
        return out


class RawLevel0(Pyramid):
    """Level 0 cut straight out of the uncompressed raw volume (uint8, C order k, j, i), with no pyramid
    copy of it read; coarser levels, the brick maxima and which bricks hold tissue come from the pyramid.
    A level-0 blob is the brick's raw 256 KB. Each brick is 16 x 128 rows of 128 bytes scattered over
    the file, read with one pread per row, against one pread of a ~190 KB compressed brick from the
    pyramid: this mode shows what the pyramid saves."""

    def __init__(self, base, raw):
        self.base, self.raw = base, raw
        self.root, self.meta, self.idx, self.vmax, self.codec = base.root, base.meta, base.idx, base.vmax, base.codec
        self.shape = self.meta["levels"][0]["shape"]
        if os.path.getsize(raw) != int(np.prod(self.shape)):
            raise ValueError(f"{raw} is {os.path.getsize(raw)} bytes, not {np.prod(self.shape)} for shape {self.shape}")
        self.local = threading.local()

    def blob(self, n, gz, gy, gx):
        if n:
            return self.base.blob(n, gz, gy, gx)
        if not self.idx[0]["n"][gz, gy, gx]:
            return None
        nk, nj, ni = self.shape
        (k0, j0, i0), b = [g * s for g, s in zip((gz, gy, gx), BRICK)], np.zeros(BRICK, np.uint8)
        k1, j1, i1 = min(k0 + BRICK[0], nk), min(j0 + BRICK[1], nj), min(i0 + BRICK[2], ni)
        fd = self.fd(self.raw)
        for k in range(k0, k1):
            for j in range(j0, j1):
                b[k - k0, j - j0, :i1 - i0] = np.frombuffer(os.pread(fd, i1 - i0, (k * nj + j) * ni + i0), np.uint8)
        return b.tobytes()

    def decode_into(self, n, blob, out):
        if n:
            return self.base.decode_into(n, blob, out)
        out[:] = np.frombuffer(blob, np.uint8)
