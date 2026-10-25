"""Cold random brick reads: drop the level-0 shard files from the client page cache (posix_fadvise DONTNEED),
then read and decode random stored bricks with N threads, through one shared file descriptor per shard and with
one descriptor per thread and shard. This is what a first visit to a new region costs; on Lustre a shared
descriptor serialises the reads.

usage: python scripts/cold_read.py data/bigbrain20_png.ome.zarr [threads ...]
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import imagecodecs
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pyramid"))
from omezarr import open_pyramid  # noqa: E402

pyr = open_pyramid(sys.argv[1], levels=[0])
threads_list = [int(t) for t in sys.argv[2:]] or [8, 32, 64, 128]
stored = np.argwhere(pyr.idx[0]["n"] > 0)
paths = sorted({pyr.path(0, *g) for g in stored})
shared_fds = {p: os.open(p, os.O_RDONLY) for p in paths}
out = np.empty((128, 16 * 128 * 128), np.uint8)
rng = np.random.default_rng(11)
local = threading.local()
for shared in (True, False):
    for threads in threads_list:
        sel = stored[rng.choice(len(stored), 1024, replace=False)]
        for f in shared_fds.values():
            os.posix_fadvise(f, 0, 0, os.POSIX_FADV_DONTNEED)

        def job(i):
            g = sel[i]
            p = pyr.path(0, *g)
            if shared:
                f = shared_fds[p]
            else:                                  # one descriptor per thread and shard
                fds = local.__dict__.setdefault("fds", {})
                f = fds.get(p) or fds.setdefault(p, os.open(p, os.O_RDONLY))
            off, n = pyr.idx[0][tuple(g)]
            b = os.pread(f, int(n), int(off))
            imagecodecs.zstd_decode(b, out=out[i % 128])
            return len(b)
        t = time.time()
        with ThreadPoolExecutor(threads) as ex:
            total = sum(ex.map(job, range(len(sel))))
        dt = time.time() - t
        print(f"{'shared fd    ' if shared else 'fd per thread'} {threads:4d} threads: {len(sel) / dt:6.0f} bricks/s, "
              f"{total / dt / 1e6:6.0f} MB/s (cold, random)")
    local = threading.local()
