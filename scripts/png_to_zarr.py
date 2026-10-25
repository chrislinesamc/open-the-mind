"""Build a pyramid from a folder of section images, one 8-bit grayscale PNG per section, with no config to
write by hand: the numbering (e.g. pm0001o.png .. pm7404o.png), the image size and whether the background is
white (then intensities are inverted so it becomes 0 and is not stored) are read from the files. Only the
section spacing and the pixel size have to be given. Writes the config next to the pyramid, then builds it
(pyramid/build_pyramid.py) within the memory budget.

usage: python scripts/png_to_zarr.py <png dir> <out.ome.zarr> --section-um 20 --pixel-um 21.17
       [--pixel-um-y 21.17] [--invert auto|yes|no] [--mem-gb 2] [--first N --count N]
       [--planes "sagittal (L-R)" "axial (S-I)" "coronal (A-P)"]
"""
import argparse
import json
import os
import re
import sys

import imagecodecs
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pyramid"))


def numbering(folder):
    """(pattern, first, count) of the PNGs in folder: one run of numbered names, e.g. ("pm{:04d}o.png", 1, 7404)."""
    names = [n for n in os.listdir(folder) if n.lower().endswith(".png")]
    if not names:
        sys.exit(f"no .png files in {folder}")
    groups = {}
    for n in names:
        m = re.fullmatch(r"(.*?)(\d+)(\D*\.png)", n, re.IGNORECASE)
        if m:
            groups.setdefault((m[1], m[3]), []).append(m[2])
    if not groups:
        sys.exit(f"no numbered .png files in {folder} (expected names like s0001.png)")
    (pre, post), digits = max(groups.items(), key=lambda g: len(g[1]))
    pad = len(digits[0]) if len({len(d) for d in digits}) == 1 else 0     # fixed width: zero-padded names
    nums = sorted(int(d) for d in digits)
    missing = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
    if missing:
        sys.exit(f"{len(missing)} sections missing from {pre}N{post} {nums[0]}..{nums[-1]}, e.g. {missing[:5]}; "
                 f"use --first/--count for a complete run")
    return pre + ("{:0%dd}" % pad if pad else "{:d}") + post, nums[0], len(nums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png_dir")
    ap.add_argument("out", help="pyramid to write, e.g. data/brain.ome.zarr")
    ap.add_argument("--section-um", type=float, required=True, help="distance between sections in micrometres")
    ap.add_argument("--pixel-um", type=float, required=True, help="pixel size in micrometres (x, and y unless --pixel-um-y)")
    ap.add_argument("--pixel-um-y", type=float)
    ap.add_argument("--invert", choices=["auto", "yes", "no"], default="auto",
                    help="255 - v so a white background becomes 0; auto: when the image border is mostly bright")
    ap.add_argument("--first", type=int, help="first section number to use (default: the lowest found)")
    ap.add_argument("--count", type=int, help="number of sections to use (default: all)")
    ap.add_argument("--planes", nargs=3, metavar=("X", "Y", "Z"), help="names of the three cut planes: across "
                    "image columns, across image rows, and across sections")
    ap.add_argument("--mem-gb", type=float, default=2, help="RAM and GPU memory budget of the build; default 2")
    ap.add_argument("--config", help="config to write (default: next to the pyramid, <name>.json)")
    a = ap.parse_args()

    pattern, first, count = numbering(a.png_dir)
    if a.first is not None:
        count -= a.first - first
        first = a.first
    if a.count is not None:
        count = min(count, a.count)
    path = lambda k: os.path.join(a.png_dir, pattern.format(k))
    if count < 1 or not os.path.exists(path(first)):
        sys.exit(f"no sections from {first}")
    img = imagecodecs.png_decode(open(path(first + count // 2), "rb").read())
    if img.ndim != 2 or img.dtype != np.uint8:
        sys.exit(f"{path(first + count // 2)} is {img.dtype} {img.shape}; 8-bit grayscale images are needed")
    nj, ni = img.shape
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    invert = a.invert == "yes" or (a.invert == "auto" and bool(np.median(border) > 127))

    name = os.path.basename(a.out.rstrip("/")).removesuffix(".zarr").removesuffix(".ome")
    cfg_path = a.config or os.path.join(os.path.dirname(a.out.rstrip("/")) or ".", name + ".json")
    px, py = a.pixel_um / 1000, (a.pixel_um_y or a.pixel_um) / 1000
    planes = a.planes or ["plane across image columns (x)", "plane across image rows (y)", "plane across sections (z)"]
    cfg = dict(name=name, png_dir=a.png_dir, png_pattern=pattern, png_first=first, invert=invert, dtype="uint8",
               shape_kji=[count, nj, ni], spacing_mm_kji=[a.section_um / 1000, py, px],
               axes=f"k = section ({pattern.format(first)} .. {pattern.format(first + count - 1)}), j = image row, "
                    f"i = image column" + ("; intensities inverted (255 - v), background is 0" if invert else ""),
               planes=dict(zip("xyz", planes)), source=os.path.abspath(a.png_dir))
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"{count} sections {pattern.format(first)} .. {pattern.format(first + count - 1)}, {ni} x {nj} pixels, "
          f"{a.section_um} um apart, {a.pixel_um} um pixels, {'inverted' if invert else 'not inverted'}; "
          f"wrote {cfg_path}", flush=True)

    from build_pyramid import build
    build(cfg_path, a.out, mem_gb=a.mem_gb)


if __name__ == "__main__":
    main()
