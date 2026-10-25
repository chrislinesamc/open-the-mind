"""Build a pyramid from a NIfTI-1 volume (.nii or .nii.gz). The file is streamed slice by slice into a raw
uint8 volume next to the pyramid, so it never has to fit in RAM; then the config is written and the pyramid
built (pyramid/build_pyramid.py) within the memory budget. The raw volume is deleted afterwards unless
--keep-raw (level 0 is lossless: scripts/export_raw.py writes it back).

- Intensities: uint8 data is kept as stored; other types are scaled linearly from their min..max (or
  --range) to 0..255, which takes a second pass over the file.
- Background: with --invert auto, a bright first slice (white background, as in histology) is inverted
  (255 - v) so the background becomes 0 and its bricks are not stored.
- Orientation (--orient coronal, the default): when the voxel axes follow the world axes, the volume is
  rewritten as a stack of coronal sections like the PNG datasets: k = posterior -> anterior, j = superior ->
  inferior (rows drawn downwards), i = left -> right. --orient none keeps the stored order (k = z, j = y, i = x).

usage: python scripts/nifti_to_zarr.py <volume.nii.gz> <out.ome.zarr> [--mem-gb 2] [--invert auto|yes|no]
       [--orient coronal|none] [--range LO HI] [--keep-raw]
"""
import argparse
import gzip
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pyramid"))

DTYPES = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}
UNIT_MM = {1: 1000.0, 2: 1.0, 3: 0.001}             # xyzt_units & 7: metre, millimetre, micrometre


def header(path):
    """Shape (x, y, z), voxel dtype, spacing in mm, data offset and the 3x3 voxel-to-world matrix."""
    with (gzip.open if path.endswith(".gz") else open)(path, "rb") as f:
        h = f.read(348)
    e = "<" if struct.unpack("<i", h[:4])[0] == 348 else ">"
    if struct.unpack(e + "i", h[:4])[0] != 348 or h[344:347] not in (b"n+1", b"ni1"):
        sys.exit(f"{path}: not a NIfTI-1 file")
    if h[344:347] == b"ni1":
        sys.exit(f"{path}: a .hdr/.img pair; convert it to a single .nii first")
    dim = struct.unpack(e + "8h", h[40:56])
    if dim[0] < 3 or any(d > 1 for d in dim[4:dim[0] + 1]):
        sys.exit(f"{path}: {dim[0]}-D data with dims {dim[1:dim[0] + 1]}; one 3-D volume is needed")
    code = struct.unpack(e + "h", h[70:72])[0]
    if code not in DTYPES:
        sys.exit(f"{path}: NIfTI datatype {code} is not supported")
    pix = np.abs(struct.unpack(e + "8f", h[76:108])[1:4])
    offset = int(struct.unpack(e + "f", h[108:112])[0])
    unit = UNIT_MM.get(h[123] & 7, 1.0)
    qform, sform = struct.unpack(e + "2h", h[252:256])
    if sform > 0:
        R = np.array(struct.unpack(e + "12f", h[280:328])).reshape(3, 4)[:, :3]
    elif qform > 0:                                  # rotation from the quaternion, times pixdim (and qfac)
        b, c, d = struct.unpack(e + "3f", h[256:268])
        a = np.sqrt(max(0.0, 1 - b * b - c * c - d * d))
        R = np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                      [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                      [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - c*c - b*b]]) * pix
        if struct.unpack(e + "f", h[76:80])[0] < 0:
            R[:, 2] *= -1
    else:
        R = np.diag(pix)
    return dim[1:4], np.dtype(e + DTYPES[code]), [round(float(p), 9) for p in pix * unit], offset, R   # float32: 0.1 -> 0.1000000015


def slices(path, shape, dtype, offset):
    """Yield the z slices of the volume, each a (y, x) array, reading the file once from the start."""
    nx, ny, nz = shape
    n = nx * ny * dtype.itemsize
    with (gzip.open if path.endswith(".gz") else open)(path, "rb") as f:
        f.read(offset)
        for z in range(nz):
            b = f.read(n)
            if len(b) < n:
                sys.exit(f"{path}: ends at slice {z} of {nz}")
            yield np.frombuffer(b, dtype).reshape(ny, nx)


def axis_map(R):
    """For each voxel axis, the world axis (0 = x left-right, 1 = y posterior-anterior, 2 = z inferior-superior)
    it runs along and its direction, or None if the voxel axes are oblique to the world axes."""
    out = []
    for a in range(3):
        col = R[:, a]
        w = int(np.argmax(np.abs(col)))
        if abs(col[w]) < 0.9 * np.linalg.norm(col):
            return None
        out.append((w, 1 if col[w] > 0 else -1))
    return out if sorted(w for w, _ in out) == [0, 1, 2] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nifti")
    ap.add_argument("out", help="pyramid to write, e.g. data/brain.ome.zarr")
    ap.add_argument("--mem-gb", type=float, default=2, help="RAM and GPU memory budget of the build; default 2")
    ap.add_argument("--invert", choices=["auto", "yes", "no"], default="auto",
                    help="255 - v so a white background becomes 0; auto: when the first slice is mostly bright")
    ap.add_argument("--orient", choices=["coronal", "none"], default="coronal")
    ap.add_argument("--range", nargs=2, type=float, metavar=("LO", "HI"), help="intensity range mapped to 0..255 "
                    "(non-uint8 data; default: the data's min..max)")
    ap.add_argument("--raw", help="raw volume to write (default: next to the pyramid, <name>.raw)")
    ap.add_argument("--keep-raw", action="store_true", help="keep the raw volume after the build (scripts/check_pyramid.py "
                    "compares the pyramid with it)")
    a = ap.parse_args()
    t0 = time.time()

    (nx, ny, nz), dtype, (sx, sy, sz), offset, R = header(a.nifti)
    print(f"{a.nifti}: {nx} x {ny} x {nz} {dtype.name}, voxels {sx:g} x {sy:g} x {sz:g} mm", flush=True)
    lo, hi = a.range or (None, None)
    if dtype != np.uint8 and lo is None:             # first pass: the data range
        lo, hi = np.inf, -np.inf
        for s in slices(a.nifti, (nx, ny, nz), dtype, offset):
            lo, hi = min(lo, float(s.min())), max(hi, float(s.max()))
        print(f"intensity range {lo:g} .. {hi:g} -> 0 .. 255", flush=True)
    to8 = (lambda s: s) if dtype == np.uint8 and a.range is None else \
        (lambda s: np.clip((s.astype(np.float32) - lo) * (255 / max(hi - lo, 1e-12)) + 0.5, 0, 255).astype(np.uint8))

    m = axis_map(R) if a.orient == "coronal" else None
    if a.orient == "coronal" and (m is None or m[0][0] != 0):
        print("voxel axes are not aligned with left-right first: keeping the stored order (--orient none)", flush=True)
        m = None
    if m:                                            # voxel x = world x; voxel y, z = world y, z in some order
        ay = [w for w, _ in m].index(1)              # the voxel axis along posterior-anterior: k
        az = [w for w, _ in m].index(2)              # the voxel axis along inferior-superior: j, flipped
        n = (nx, ny, nz)
        shape = (n[ay], n[az], nx)
        spacing = [(sx, sy, sz)[ay], (sx, sy, sz)[az], sx]
        planes = dict(x="sagittal plane (L–R, x)", y="axial plane (S–I, y)", z="coronal plane (A–P, z, section axis)")
        axes = "k = posterior -> anterior (coronal sections), j = superior -> inferior, i = left -> right; C order"
    else:
        shape, spacing = (nz, ny, nx), [sz, sy, sx]
        planes = dict(x="plane across x", y="plane across y", z="plane across z (section axis)")
        axes = "k = voxel z, j = voxel y, i = voxel x as stored; C order"

    name = os.path.basename(a.out.rstrip("/")).removesuffix(".zarr").removesuffix(".ome")
    base = os.path.dirname(a.out.rstrip("/")) or "."
    raw, cfg_path = a.raw or os.path.join(base, name + ".raw"), os.path.join(base, name + ".json")
    nk, nj, ni = shape
    invert = None if a.invert == "auto" else a.invert == "yes"
    fd = os.open(raw + ".part", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd, nk * nj * ni)
    for z, s in enumerate(slices(a.nifti, (nx, ny, nz), dtype, offset)):
        s = to8(s)
        if invert is None:
            invert = bool(np.median(s) > 127)
        if invert:
            s = 255 - s
        if not m:
            os.pwrite(fd, s.tobytes(), z * nj * ni)
            continue
        if m[0][1] < 0:                              # i runs left -> right
            s = s[:, ::-1]
        # s is indexed (voxel y, voxel x); voxel z is this slice
        kdir, jdir = m[ay][1], m[az][1]
        if ay == 1:                                  # rows of s are sections k; this slice is one row j of each
            j = nj - 1 - z if jdir > 0 else z
            for r in range(ny):
                k = r if kdir > 0 else ny - 1 - r
                os.pwrite(fd, s[r].tobytes(), (k * nj + j) * ni)
        else:                                        # this slice is a whole section k; its rows are j
            k = z if kdir > 0 else nz - 1 - z
            os.pwrite(fd, np.ascontiguousarray(s[::-1] if jdir > 0 else s).tobytes(), k * nj * ni)
        if z % 100 == 0:
            print(f"  slice {z}/{nz}  {time.time() - t0:.0f}s", flush=True)
    os.close(fd)
    os.rename(raw + ".part", raw)

    cfg = dict(name=name, raw=raw, dtype="uint8", shape_kji=list(shape), spacing_mm_kji=spacing,
               axes=axes + ("; intensities inverted (255 - v), background is 0" if invert else ""),
               planes=planes, source=os.path.abspath(a.nifti))
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"wrote {raw} ({nk} x {nj} x {ni}, {'inverted' if invert else 'not inverted'}) and {cfg_path} "
          f"in {time.time() - t0:.0f}s", flush=True)

    from build_pyramid import build
    build(cfg_path, a.out, mem_gb=a.mem_gb)
    if not a.keep_raw:
        os.remove(raw)
        print(f"removed {raw} (--keep-raw keeps it)", flush=True)


if __name__ == "__main__":
    main()
