"""Bounded renderer vs a dense reference.

The fused kernel samples bricks through the page table, looks up neighbour bricks at brick borders
and composites front to back. The reference ray-marches the dense level-0 volume with plain torch
indexing, using the same sample positions, so the two images must agree to within float rounding.
Also: empty-space skipping changes the image only slightly, misses fall back to coarse levels
without holes, and a pool much smaller than the data stays consistent under eviction.

run: python -m pytest tests/test_raycast.py -q
"""
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(HERE, "..", "server"), os.path.join(HERE, "..", "pyramid"), HERE]
from build_pyramid import build  # noqa: E402
import colormaps  # noqa: E402
from renderer import TANF, Renderer, parse  # noqa: E402
from test_pyramid import synthetic  # noqa: E402

SHAPE = (37, 700, 900)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rc")
    vol = synthetic(SHAPE, seed=3)
    vol.tofile(tmp / "vol.raw")
    json.dump(dict(name="synth", raw=str(tmp / "vol.raw"), dtype="uint8", shape_kji=list(SHAPE),
                   spacing_mm_kji=[0.016, 0.004, 0.004], axes="test"), open(tmp / "cfg.json", "w"))
    build(str(tmp / "cfg.json"), str(tmp / "s.zarr"))
    return vol, str(tmp / "s.zarr")


def settle(R, p, n=50):
    for _ in range(n):
        img, st = R.draw(p)
        if st["misses"] == 0 and not R.cache.busy():
            return img
        torch.cuda.synchronize()
    raise AssertionError("bricks never became resident")


def reference(R, vol, p):
    """Dense ray march of level 0 with the kernel's sample positions and compositing."""
    dev = R.dev
    W, H = p["w"], p["h"]
    eye, fwd, right, up, target = R.camera(p)
    lo, hi, olo, ohi = R.regions(p, eye)
    f = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)
    u = ((torch.arange(W, device=dev) + 0.5) / W * 2 - 1)[None, :].expand(H, W).reshape(-1, 1)
    v = ((torch.arange(H, device=dev) + 0.5) / H * 2 - 1)[:, None].expand(H, W).reshape(-1, 1)
    d = f(fwd) + u * f(right * TANF * W / H) - v * f(up * TANF)
    d = d / d.norm(dim=1, keepdim=True)
    d = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    e = f(eye)
    t0, t1 = (f(lo) - e) / d, (f(hi) - e) / d
    tnear = torch.minimum(t0, t1).amax(1).clamp_min(0)
    tfar = torch.maximum(t0, t1).amin(1)
    sk, sj, si = R.cache.levels[0]["spacing"]
    V = torch.from_numpy(vol).to(dev).float()
    vmin = f(-R.size / 2)
    dt = p["step"] / (d.abs() / f([si, sj, sk])).amax(1)              # one voxel along each ray
    T = torch.ones(W * H, device=dev)
    C = torch.zeros(W * H, 3, device=dev)
    t = tnear.clone()
    first = torch.ones(W * H, dtype=torch.bool, device=dev)
    active = tnear < tfar
    for _ in range(20000):
        if not active.any():
            break
        P = e + d * t[:, None]
        q = (P - vmin) / f([si, sj, sk]) - 0.5                     # x, y, z voxel coords
        i0 = q.floor()
        w = q - i0
        i0 = i0.long()
        val = torch.zeros(W * H, device=dev)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    x, y, z = i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz
                    ok = (x >= 0) & (x < SHAPE[2]) & (y >= 0) & (y < SHAPE[1]) & (z >= 0) & (z < SHAPE[0])
                    g = V[z.clamp(0, SHAPE[0] - 1), y.clamp(0, SHAPE[1] - 1), x.clamp(0, SHAPE[2] - 1)] * ok
                    wt = (w[:, 0] if dx else 1 - w[:, 0]) * (w[:, 1] if dy else 1 - w[:, 1]) * \
                         (w[:, 2] if dz else 1 - w[:, 2])
                    val += g * wt
        x = ((val - p["lo"]) / max(p["hi"] - p["lo"], 1.0)).clamp(0, 1) * active
        a = 1 - torch.exp(-p["dens"] * x * dt / 0.01)
        a = torch.where(first & (x > 0), torch.clamp(a, min=p["face"]), a)
        lut = torch.from_numpy(colormaps.lut(p["colour"], p["gamma"], p["c0"], p["c1"])).to(dev)
        fi = x * 255
        i0 = fi.long().clamp(max=254)
        col = lut[i0] + (lut[i0 + 1] - lut[i0]) * (fi - i0)[:, None]
        C += (T * a)[:, None] * col
        T = T * (1 - a)
        first &= ~active
        t = t + dt
        active &= (t < tfar) & (T > 0.02)
    return (C.clamp(max=1) * 255).to(torch.uint8).T.reshape(3, H, W)


VIEWS = [dict(), dict(cx=0.4, az=-50, el=20, colour=1), dict(colour=3, gamma=0.6, az=60),
         dict(colour=7, c0=0x203040, c1=0xffe080, az=-120, el=10), dict(cz=0.5, fz=1, az=10, el=-30, dist=4.0),
         dict(face=0.0, dens=0.6, az=120, el=40)]


@pytest.mark.parametrize("view", VIEWS)
def test_matches_dense_reference(setup, view):
    vol, root = setup
    R = Renderer(root, pool_gb=0.1, host_gb=0.1)
    p = parse(dict(view, w=160, h=120, lod=-50, skip=0, box=0, step=1.0, budget=10**6))   # level 0 everywhere
    img = settle(R, p).int()
    ref = reference(R, vol, p).int()
    diff = (img - ref).abs().float()
    assert ref.float().mean() > 5, "view should show tissue"
    assert diff.mean() < 0.3 and torch.quantile(diff.flatten(), 0.999) <= 3, \
        f"mean {diff.mean():.3f}, max {diff.max():.0f}"


def test_skipping_and_fallback(setup):
    vol, root = setup
    R = Renderer(root, pool_gb=60 * 262144 / 2**30, host_gb=0.1)      # pins only the coarse levels
    assert R.cache.pinned_from > 0
    p = parse(dict(w=160, h=120, lod=-50, box=0, az=-50, el=20, cx=0.4))
    coarse, st = R.draw(p)                       # nothing fine is resident yet: coarse fallback
    assert st["misses"] > 0 and coarse.float().mean() > 5
    skip = settle(R, p).int()
    noskip = settle(R, dict(p, skip=0)).int()
    assert (skip - noskip).abs().float().mean() < 1.0


def test_small_pool_stays_consistent(setup):
    vol, root = setup
    R = Renderer(root, pool_gb=40 * 262144 / 2**30, host_gb=0.01)     # 40 slots for 62 stored bricks
    c = R.cache
    pinned_owner = c.owner[:c.n_pinned].clone()
    rng = np.random.default_rng(0)
    for i in range(60):
        p = parse(dict(w=96, h=72, lod=-50, box=0, az=float(rng.uniform(-180, 180)),
                       el=float(rng.uniform(-60, 60)), dist=float(rng.uniform(2, 8)),
                       cx=float(rng.uniform(0, 0.6))))
        R.draw(p)
    torch.cuda.synchronize()
    page, owner = c.page.cpu(), c.owner.cpu()
    held = owner[:c.next_free]
    assert c.S == 40 and c.next_free <= c.S and c.stats["evicted"] > 0
    assert (page[held[held >= 0]] == torch.arange(c.next_free)[held >= 0].int()).all()
    assert int((page >= 0).sum()) == int((held >= 0).sum())
    assert c.n_pinned > 0 and torch.equal(c.owner[:c.n_pinned].cpu(), pinned_owner.cpu()), "pinned bricks were evicted"


def test_finest_level_cap(setup):
    vol, root = setup
    R = Renderer(root, pool_gb=0.1, host_gb=0.1, finest=1)            # level 0 is never used or loaded
    p = parse(dict(w=160, h=120, lod=-50, box=0, az=-50, el=20, cx=0.4, dist=4.0))
    img = settle(R, p)
    torch.cuda.synchronize()
    c = R.cache
    assert img.float().mean() > 5, "view should show tissue from level 1"
    assert not (c.page[:int(c.off[1])] >= 0).any(), "a level-0 brick was loaded"
    assert (c.page[int(c.off[1]):int(c.off[2])] >= 0).any(), "level 1 should be resident"


@pytest.mark.parametrize("axis", "xyz")
def test_open_one_plane_matches_crop(setup, axis):
    """Opening with one plane removes the half on the camera's side: the same image as cropping to the
    other half (which test_matches_dense_reference checks against the dense reference)."""
    vol, root = setup
    R = Renderer(root, pool_gb=0.1, host_gb=0.1)
    base = dict(w=160, h=120, lod=-50, skip=0, box=0, budget=10**6, az=35, el=25, auto=0, ox=0, oy=0, oz=0)
    base["o" + axis] = 1
    opened = settle(R, parse(dict(base, mode=1, **{"c" + axis: 0.45}))).int()
    cropped = settle(R, parse(dict(base, mode=0, **{"c" + axis: 0.55, "f" + axis: 1}))).int()
    whole = settle(R, parse(dict(base, mode=0, ox=0, oy=0, oz=0))).int()
    diff = (opened - cropped).abs().float()
    assert (opened - whole).abs().float().mean() > 1, "opening should change the view"
    assert diff.mean() < 0.3 and torch.quantile(diff.flatten(), 0.999) <= 3, f"mean {diff.mean():.3f}"


def test_open_three_planes_keeps_the_rest(setup):
    """With all three planes on, only rays through the removed corner change; the rest of the brain stays."""
    vol, root = setup
    R = Renderer(root, pool_gb=0.1, host_gb=0.1)
    base = dict(w=160, h=120, lod=-50, skip=0, box=0, budget=10**6, az=35, el=25, cx=0.5, cy=0.5, cz=0.5)
    opened = settle(R, parse(dict(base, mode=1))).int()
    whole = settle(R, parse(dict(base, mode=1, ox=0, oy=0, oz=0))).int()
    changed = (opened - whole).abs().amax(0) > 3
    frac = changed.float().mean().item()
    assert 0.01 < frac < 0.5, f"{frac:.3f} of the pixels changed"


def test_raw_level0_matches_pyramid(setup):
    """Level 0 read straight from the raw volume gives the same bricks, so the same image, as the pyramid."""
    vol, root = setup
    raw = os.path.join(os.path.dirname(root), "vol.raw")
    p = parse(dict(w=160, h=120, lod=-50, box=0, az=-50, el=20, mode=1, cx=0.5, cy=0.5, cz=0.5))
    R = Renderer(root, pool_gb=0.1, host_gb=0.1)
    Rr = Renderer(root, pool_gb=0.1, host_gb=0.1, raw=raw)
    pyr, rp = R.cache.pyr, Rr.cache.pyr
    for g in np.argwhere(pyr.idx[0]["n"] > 0):
        assert np.array_equal(pyr.brick(0, *g), rp.brick(0, *g)), f"brick {g} differs"
    assert torch.equal(settle(R, p), settle(Rr, p))
