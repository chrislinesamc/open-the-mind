"""Write level 0 of a pyramid back out as the raw volume (uint8, C order, k j i). Level 0 is a lossless copy of the
source, so the 138.5 GB raw file can be rebuilt from the 18 GB level 0 instead of being copied around.

usage: python scripts/export_raw.py data/bigbrain20_png.ome.zarr out.raw [first_section end_section]
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pyramid"))
from bricks import BRICK  # noqa: E402
from omezarr import open_pyramid  # noqa: E402

root, out = sys.argv[1], sys.argv[2]
P = open_pyramid(root, levels=[0])
nk, nj, ni = P.meta["levels"][0]["shape"]
k0, k1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, nk)
bz, by, bx = BRICK
gy_n, gx_n = P.idx[0].shape[1:]
fd = os.open(out, os.O_WRONLY | os.O_CREAT)
os.ftruncate(fd, nk * nj * ni)
for gz in range(k0 // bz, -(-k1 // bz)):
    slab = np.zeros((bz, gy_n * by, gx_n * bx), np.uint8)

    def fill(gy):
        for gx in range(gx_n):
            slab[:, gy * by:(gy + 1) * by, gx * bx:(gx + 1) * bx] = P.brick(0, gz, gy, gx)
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(fill, range(gy_n)))
    a, b = max(gz * bz, k0), min((gz + 1) * bz, k1)
    data = np.ascontiguousarray(slab[a - gz * bz:b - gz * bz, :nj, :ni])
    view, off = memoryview(data).cast("B"), a * nj * ni
    while view:                                    # pwrite is capped near 2 GB per call
        n = os.pwrite(fd, view[:1 << 30], off)
        view, off = view[n:], off + n
    print(f"sections {a}-{b - 1} written", flush=True)
os.close(fd)
