"""Check that a pyramid built from a stack of section images holds exactly the same voxels: level 0, brick row
by brick row, against the decoded images (inverted if the config says so). With --all every section is
compared; otherwise a random sample. A few sections are also read through zarr-python, an independent reader.

usage: python scripts/check_png_pyramid.py configs/bigbrain20_png.json data/bigbrain20_png.ome.zarr
       [--all | --sections 32] [--zarr-sections 3] [--threads 16]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import imagecodecs
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pyramid"))
from bricks import BRICK  # noqa: E402
from omezarr import open_pyramid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("pyramid")
    ap.add_argument("--all", action="store_true", help="compare every section")
    ap.add_argument("--sections", type=int, default=32, help="random sections to compare without --all")
    ap.add_argument("--zarr-sections", type=int, default=3, help="sections also read through zarr-python")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    nk, nj, ni = cfg["shape_kji"]
    first, inv = cfg.get("png_first", 0), cfg.get("invert", False)
    pyr = open_pyramid(a.pyramid, levels=[0])

    def image(k):
        img = imagecodecs.png_decode(open(os.path.join(cfg["png_dir"], cfg["png_pattern"].format(k + first)), "rb").read())
        return 255 - img if inv else img

    rng = np.random.default_rng(0)
    ks = range(nk) if a.all else sorted({0, nk - 1, *rng.choice(nk, a.sections, replace=False).tolist()})
    rows = sorted({k // BRICK[0] for k in ks})
    t0, checked, bad = time.time(), 0, []
    with ThreadPoolExecutor(a.threads) as ex:
        for i, gz in enumerate(rows):
            want = [k for k in ks if k // BRICK[0] == gz]
            slab = pyr.rows(0, gz, gz + 1)
            for k, img in zip(want, ex.map(image, want)):
                got = slab[k - gz * BRICK[0], :nj, :ni]
                if not np.array_equal(got, img):
                    d = np.argwhere(got != img)
                    bad.append((k, len(d), d[0].tolist()))
                checked += 1
            if i % 50 == 0 or i == len(rows) - 1:
                print(f"{checked}/{len(ks)} sections compared in {time.time() - t0:.0f}s, {len(bad)} differ", flush=True)

    import zarr
    g = zarr.open_group(a.pyramid, mode="r")
    zk = rng.choice(nk, a.zarr_sections, replace=False).tolist()
    zbad = [k for k in zk if not np.array_equal(g["0"][k], image(k))]
    print(f"level 0 vs section images: {checked} sections ({checked * nj * ni / 1e9:.1f} G voxels), "
          f"{len(bad)} differ{': ' + str(bad[:5]) if bad else ''}")
    print(f"zarr-python, sections {zk}: {len(zk) - len(zbad)}/{len(zk)} identical")
    sys.exit(1 if bad or zbad else 0)


if __name__ == "__main__":
    main()
