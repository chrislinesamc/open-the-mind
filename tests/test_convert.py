"""Converters: a folder of numbered PNGs (scripts/png_to_zarr.py) and a NIfTI volume (scripts/nifti_to_zarr.py)
must give the volume they describe, in the documented orientation, with the right config.

run: python -m pytest tests/test_convert.py -q
"""
import gzip
import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "pyramid"))
from omezarr import open_pyramid  # noqa: E402


def run(script, *args):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), *map(str, args)],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def level0(zarr):
    pyr = open_pyramid(str(zarr))
    L = pyr.meta["levels"][0]
    z, y, x = L["shape"]
    return pyr.rows(0, 0, L["grid"][0])[:z, :y, :x], pyr.meta


def volume(shape, seed=0):
    """Bright background with darker 'tissue' inside an ellipsoid, like histology."""
    rng = np.random.default_rng(seed)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    inside = sum(((a - (n - 1) / 2) / (0.4 * n)) ** 2 for a, n in zip((z, y, x), shape)) < 1
    return np.where(inside, rng.integers(0, 200, shape), 255).astype(np.uint8)


def test_png_folder(tmp_path):
    """Zero-padded numbering starting at 5, white background: detected and inverted."""
    import imagecodecs
    vol = volume((20, 150, 170))
    d = tmp_path / "png"
    d.mkdir()
    for k in range(vol.shape[0]):
        (d / f"sec_{k + 5:05d}.png").write_bytes(imagecodecs.png_encode(vol[k]))
    (d / "notes.png").write_bytes(imagecodecs.png_encode(vol[0]))          # unnumbered: ignored
    run("png_to_zarr.py", d, tmp_path / "s.ome.zarr", "--section-um", 40, "--pixel-um", 10, "--mem-gb", 0.01)
    cfg = json.load(open(tmp_path / "s.json"))
    assert (cfg["png_pattern"], cfg["png_first"], cfg["invert"]) == ("sec_{:05d}.png", 5, True)
    assert cfg["shape_kji"] == [20, 150, 170] and cfg["spacing_mm_kji"] == [0.04, 0.01, 0.01]
    got, _ = level0(tmp_path / "s.ome.zarr")
    assert np.array_equal(got, 255 - vol)


def test_png_folder_with_a_gap(tmp_path):
    import imagecodecs
    d = tmp_path / "png"
    d.mkdir()
    for k in (1, 2, 4):
        (d / f"s{k}.png").write_bytes(imagecodecs.png_encode(np.zeros((8, 8), np.uint8)))
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "png_to_zarr.py"), str(d), str(tmp_path / "s.ome.zarr"),
                          "--section-um", "20", "--pixel-um", "20"], capture_output=True, text=True)
    assert out.returncode != 0 and "missing" in out.stderr


def write_nifti(path, data_zyx, srow, dtype=np.uint8, code=2):
    """A single-file NIfTI-1 (gzipped if path ends in .gz) of a C-order (z, y, x) array, x varying fastest."""
    nz, ny, nx = data_zyx.shape
    h = bytearray(348)
    struct.pack_into("<i", h, 0, 348)
    struct.pack_into("<8h", h, 40, 3, nx, ny, nz, 1, 1, 1, 1)
    struct.pack_into("<2h", h, 70, code, np.dtype(dtype).itemsize * 8)
    struct.pack_into("<8f", h, 76, 1, *np.abs(np.diag(np.array(srow)[:, :3])), 1, 1, 1, 1)
    struct.pack_into("<f", h, 108, 352)
    struct.pack_into("<2f", h, 112, 1, 0)
    h[123] = 2                                        # millimetres
    struct.pack_into("<2h", h, 252, 0, 1)            # sform
    struct.pack_into("<12f", h, 280, *np.array(srow, np.float32).ravel())
    h[344:348] = b"n+1\0"
    blob = bytes(h) + b"\0" * 4 + data_zyx.astype(dtype).tobytes()
    with (gzip.open if str(path).endswith(".gz") else open)(path, "wb") as f:
        f.write(blob)


RAS = [[0.1, 0, 0, 0], [0, 0.2, 0, 0], [0, 0, 0.3, 0]]           # voxel x, y, z = world x (L->R), y (P->A), z (I->S)


@pytest.mark.parametrize("orient", ["coronal", "none"])
def test_nifti_uint8(tmp_path, orient):
    vol = volume((30, 40, 50))                        # (z, y, x)
    write_nifti(tmp_path / "v.nii.gz", vol, RAS)
    run("nifti_to_zarr.py", tmp_path / "v.nii.gz", tmp_path / "v.ome.zarr", "--orient", orient, "--mem-gb", 0.01)
    assert not (tmp_path / "v.raw").exists(), "raw volume should be removed without --keep-raw"
    cfg = json.load(open(tmp_path / "v.json"))
    got, _ = level0(tmp_path / "v.ome.zarr")
    if orient == "coronal":                           # k = y (P->A), j = z flipped (S->I), i = x (L->R)
        want = (255 - vol).transpose(1, 0, 2)[:, ::-1]
        assert cfg["spacing_mm_kji"] == pytest.approx([0.2, 0.3, 0.1])
    else:
        want = 255 - vol
        assert cfg["spacing_mm_kji"] == pytest.approx([0.3, 0.2, 0.1])
    assert list(want.shape) == cfg["shape_kji"]
    assert np.array_equal(got, want)


def test_nifti_other_axis_order_and_int16(tmp_path):
    """Voxel y runs inferior->superior and voxel z anterior->posterior; x runs right->left. int16 is scaled from
    its min..max to 0..255, and a dark first slice means no inversion."""
    rng = np.random.default_rng(1)
    vol = rng.integers(-1000, 3000, (12, 16, 20)).astype(np.int16)
    vol[0] = -1000
    srow = [[-0.1, 0, 0, 0], [0, 0, -0.3, 0], [0, 0.2, 0, 0]]   # columns: voxel x -> -world x, y -> +z, z -> -y
    write_nifti(tmp_path / "v.nii", vol, srow, np.int16, 4)
    run("nifti_to_zarr.py", tmp_path / "v.nii", tmp_path / "v.ome.zarr", "--mem-gb", 0.01, "--keep-raw")
    assert (tmp_path / "v.raw").exists()
    got, _ = level0(tmp_path / "v.ome.zarr")
    lo, hi = vol.min(), vol.max()
    v8 = np.clip((vol.astype(np.float32) - lo) * (255 / (hi - lo)) + 0.5, 0, 255).astype(np.uint8)
    # k = posterior->anterior = voxel z reversed; j = superior->inferior = voxel y reversed; i = L->R = x reversed
    want = v8[::-1, ::-1, ::-1]
    assert np.array_equal(got, want)
