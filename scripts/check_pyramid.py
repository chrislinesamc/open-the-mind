"""Spot-check a built pyramid against its raw source: random level-0 bricks must match exactly, and
the coarsest anisotropic level must match an independent block mean of the source.

usage: python scripts/check_pyramid.py configs/<dataset>.json data/<dataset>.ome.zarr [reference.npy]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pyramid"))
from bricks import BRICK  # noqa: E402
from omezarr import open_pyramid  # noqa: E402

cfg = json.load(open(sys.argv[1]))
pyr = open_pyramid(sys.argv[2])
nk, nj, ni = cfg["shape_kji"]
raw = np.memmap(cfg["raw"], np.uint8, "r", shape=(nk, nj, ni))
rng = np.random.default_rng(1)
idx = pyr.idx[0]
stored = np.argwhere(idx["n"] > 0)
empty = np.argwhere(idx["n"] == 0)
bz, by, bx = BRICK
for name, cells in (("stored", stored), ("empty", empty)):
    bad = 0
    for gz, gy, gx in cells[rng.choice(len(cells), 40, replace=False)]:
        src = np.zeros(BRICK, np.uint8)
        s = raw[gz * bz:(gz + 1) * bz, gy * by:(gy + 1) * by, gx * bx:(gx + 1) * bx]
        src[:s.shape[0], :s.shape[1], :s.shape[2]] = s
        bad += not np.array_equal(pyr.brick(0, gz, gy, gx), src)
    print(f"level 0, 40 random {name} bricks: {bad} differ from the raw source")

if len(sys.argv) > 3:      # independent 16x16 block mean of the source (poc_cube/tools/coarse_volume.py)
    ref = np.load(sys.argv[3], mmap_mode="r")
    n = next(n for n, L in enumerate(pyr.meta["levels"]) if L["shape"] == list(ref.shape))
    got = pyr.rows(n, 0, pyr.meta["levels"][n]["grid"][0])[:ref.shape[0], :ref.shape[1], :ref.shape[2]]
    d = np.abs(got.astype(int) - ref)
    print(f"level {n} vs {os.path.basename(sys.argv[3])}: max |diff| {d.max()}, mean {d.mean():.3f} "
          f"(chained 2x2 rounding vs one 16x16 mean)")
