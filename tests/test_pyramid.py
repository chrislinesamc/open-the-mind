"""Pyramid builder: every stored brick must equal the source (level 0) or the block mean of the
level below (in-plane 2x2 means, then 2x2x2), and empty bricks must not be stored.

run: python -m pytest tests/test_pyramid.py -q
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pyramid"))
from bricks import BRICK  # noqa: E402
from build_pyramid import build  # noqa: E402
from omezarr import open_pyramid  # noqa: E402


def block_mean(a, f):
    """Mean over the valid voxels of each block (blocks cut by the data edge average fewer voxels)."""
    pad = [(0, -n % k) for n, k in zip(a.shape, f)]
    v = np.pad(np.ones(a.shape, np.int64), pad)
    a = np.pad(a, pad).astype(np.int64)
    z, y, x = a.shape
    blocks = lambda b: b.reshape(z // f[0], f[0], y // f[1], f[1], x // f[2], f[2]).sum((1, 3, 5))
    s, c = blocks(a), blocks(v)
    return ((s + c // 2) // c).astype(np.uint8)


def synthetic(shape, seed=0):
    """Noisy 'tissue' inside an ellipsoid, exact zeros outside, like the brain volumes."""
    rng = np.random.default_rng(seed)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c = [(n - 1) / 2 for n in shape]
    inside = ((z - c[0]) / (0.45 * shape[0])) ** 2 + ((y - c[1]) / (0.3 * shape[1])) ** 2 \
        + ((x - c[2]) / (0.3 * shape[2])) ** 2 < 1
    return np.where(inside, rng.integers(1, 256, shape), 0).astype(np.uint8)


def source(tmp_path):
    """Synthetic volume and its build config."""
    shape = (37, 701, 901)                       # odd, and not a multiple of the brick on any axis
    vol = synthetic(shape)
    vol[-1], vol[:, -1], vol[:, :, -1] = 200, 150, 100   # data on the far faces: partial edge blocks
    raw = tmp_path / "vol.raw"
    vol.tofile(raw)
    cfg = tmp_path / "cfg.json"
    json.dump(dict(name="synth", raw=str(raw), dtype="uint8", shape_kji=list(shape),
                   spacing_mm_kji=[0.016, 0.004, 0.004], axes="test"), open(cfg, "w"))
    return vol, str(cfg)


def reference_levels(vol, levels):
    ref = vol
    for n, L in enumerate(levels):
        if n:
            ref = block_mean(ref, L["factor"])
        yield ref


@pytest.mark.parametrize("codec", ["zstd", "none"])
def test_build_matches_reference(tmp_path, codec):
    vol, cfg = source(tmp_path)
    out = str(tmp_path / "synth.ome.zarr")
    build(cfg, out, codec)

    pyr = open_pyramid(out)
    levels = pyr.meta["levels"]
    assert [L["factor"] for L in levels] == [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2]]
    for n, (L, ref) in enumerate(zip(levels, reference_levels(vol, levels))):
        assert list(ref.shape) == L["shape"]
        got = pyr.rows(n, 0, L["grid"][0])
        z, y, x = ref.shape
        assert np.array_equal(got[:z, :y, :x], ref), f"level {n} differs from reference"
        assert not got[z:].any() and not got[:, y:].any() and not got[:, :, x:].any()
        # stored exactly the non-empty bricks, with the right max
        gz, gy, gx = L["grid"]
        blocks = got.reshape(gz, BRICK[0], gy, BRICK[1], gx, BRICK[2]).max((1, 3, 5))
        assert np.array_equal(pyr.idx[n]["n"] > 0, blocks > 0)
        assert np.array_equal(pyr.vmax[n], blocks)
    assert (pyr.idx[0]["n"] == 0).any(), "test volume should have empty bricks"


def test_reads_with_standard_zarr(tmp_path):
    """A standard Zarr v3 reader returns the reference volumes and the OME multiscales metadata, and the
    group validates as an OME-Zarr 0.5 image."""
    zarr = pytest.importorskip("zarr")
    vol, cfg = source(tmp_path)
    build(cfg, str(tmp_path / "s.ome.zarr"))
    pyr = open_pyramid(str(tmp_path / "s.ome.zarr"))
    levels = pyr.meta["levels"]

    g = zarr.open_group(str(tmp_path / "s.ome.zarr"), mode="r")
    ms = g.attrs["ome"]["multiscales"][0]
    assert g.attrs["ome"]["version"] == "0.5" and [a["name"] for a in ms["axes"]] == ["z", "y", "x"]
    s0 = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert s0 == [16.0, 4.0, 4.0]
    for n, (L, ref) in enumerate(zip(levels, reference_levels(vol, levels))):
        ds = ms["datasets"][n]
        assert ds["path"] == str(n)
        scale = ds["coordinateTransformations"][0]["scale"]
        assert np.allclose(scale, np.array(L["spacing"]) * 1000)
        if n:
            assert np.allclose(ds["coordinateTransformations"][1]["translation"], (np.array(scale) - s0) / 2)
        assert np.array_equal(g[str(n)][:], ref), f"zarr-python read of level {n} differs from reference"
        assert np.array_equal(g["brick_max"][str(n)][:], pyr.vmax[n])
    try:
        from ome_zarr_models.v05.image import Image
    except ImportError:
        return
    Image.from_zarr(g)                           # raises unless the group is a valid OME-Zarr 0.5 image


def test_dense_uncompressed_pyramid(tmp_path):
    """--codec none --keep-empty stores every brick as plain voxels: the same volumes, every brick present,
    readable by zarr-python."""
    zarr = pytest.importorskip("zarr")
    vol, cfg = source(tmp_path)
    build(cfg, str(tmp_path / "sparse.ome.zarr"))
    build(cfg, str(tmp_path / "dense.ome.zarr"), "none", True)
    sparse, dense = open_pyramid(str(tmp_path / "sparse.ome.zarr")), open_pyramid(str(tmp_path / "dense.ome.zarr"))
    assert dense.codec == "none"
    g = zarr.open_group(str(tmp_path / "dense.ome.zarr"), mode="r")
    for n, L in enumerate(dense.meta["levels"]):
        assert (dense.idx[n]["n"] == 16 * 128 * 128).all(), "every brick stored, uncompressed"
        assert np.array_equal(dense.rows(n, 0, L["grid"][0]), sparse.rows(n, 0, L["grid"][0]))
        assert np.array_equal(g[str(n)][:], sparse.rows(n, 0, L["grid"][0])[:L["shape"][0], :L["shape"][1], :L["shape"][2]])


@pytest.mark.parametrize("invert", [False, True])
def test_png_stack_matches_raw(tmp_path, invert):
    """A stack of section PNGs builds the same pyramid as the raw volume (inverted with "invert")."""
    import imagecodecs
    vol, cfg = source(tmp_path)
    png = tmp_path / "png"
    png.mkdir()
    for k in range(vol.shape[0]):
        (png / f"s{k + 1:04d}.png").write_bytes(imagecodecs.png_encode(255 - vol[k] if invert else vol[k]))
    c = json.load(open(cfg))
    c.pop("raw")
    c.update(png_dir=str(png), png_pattern="s{:04d}.png", png_first=1, invert=invert)
    json.dump(c, open(tmp_path / "png.json", "w"))
    build(cfg, str(tmp_path / "raw.ome.zarr"))
    build(str(tmp_path / "png.json"), str(tmp_path / "png.ome.zarr"))
    a, b = open_pyramid(str(tmp_path / "raw.ome.zarr")), open_pyramid(str(tmp_path / "png.ome.zarr"))
    for n, L in enumerate(a.meta["levels"]):
        assert np.array_equal(a.idx[n]["n"] > 0, b.idx[n]["n"] > 0) and np.array_equal(a.vmax[n], b.vmax[n])
        assert np.array_equal(a.rows(n, 0, L["grid"][0]), b.rows(n, 0, L["grid"][0]))


@pytest.mark.parametrize("kind", ["raw", "png"])
def test_memory_budget_strips_match(tmp_path, kind):
    """With a tiny --mem-gb every level is built in strips of the smallest size (raw and PNG input, edges
    not on brick boundaries): the pyramid must equal the one built from whole slabs, brick for brick."""
    import imagecodecs
    vol, cfg = source(tmp_path)
    if kind == "png":
        png = tmp_path / "png"
        png.mkdir()
        for k in range(vol.shape[0]):
            (png / f"s{k + 1:04d}.png").write_bytes(imagecodecs.png_encode(vol[k]))
        c = json.load(open(cfg))
        c.pop("raw")
        c.update(png_dir=str(png), png_pattern="s{:04d}.png", png_first=1)
        cfg = str(tmp_path / "png.json")
        json.dump(c, open(cfg, "w"))
    build(cfg, str(tmp_path / "whole.ome.zarr"))
    build(cfg, str(tmp_path / "strips.ome.zarr"), mem_gb=0.01)
    a, b = open_pyramid(str(tmp_path / "whole.ome.zarr")), open_pyramid(str(tmp_path / "strips.ome.zarr"))
    for n, L in enumerate(a.meta["levels"]):
        assert np.array_equal(a.idx[n]["n"] > 0, b.idx[n]["n"] > 0) and np.array_equal(a.vmax[n], b.vmax[n])
        assert np.array_equal(a.rows(n, 0, L["grid"][0]), b.rows(n, 0, L["grid"][0])), f"level {n} differs"


def test_reads_under_a_low_open_file_limit(tmp_path):
    """With a terminal's default soft limit (1024), 64 threads reading every brick must not run out of
    descriptors: the reader raises the soft limit to the hard one and bounds descriptors per thread."""
    import resource
    import subprocess
    vol, cfg = source(tmp_path)
    build(cfg, str(tmp_path / "s.ome.zarr"))
    code = f"""
import resource, sys
resource.setrlimit(resource.RLIMIT_NOFILE, (64, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
sys.path.insert(0, {os.path.join(os.path.dirname(__file__), "..", "pyramid")!r})
import bricks
bricks.FD_PER_THREAD = 2                      # also exercise closing and reopening
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from omezarr import open_pyramid
p = open_pyramid({str(tmp_path / "s.ome.zarr")!r})
cells = [(n, *g) for n in p.idx for g in np.argwhere(p.idx[n]["n"] > 0)]
with ThreadPoolExecutor(64) as ex:
    assert all(b is not None for b in ex.map(lambda c: p.blob(*c), cells * 5))
print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    if resource.getrlimit(resource.RLIMIT_NOFILE)[1] > 64:
        assert int(out.stdout.split()[-1]) > 64, "soft limit was not raised"
