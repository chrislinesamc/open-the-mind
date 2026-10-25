"""Fused ray-casting kernel over the GPU brick pool (Triton), one thread per ray.

For every sample along a ray the kernel
  1. picks a pyramid level from the pixel footprint at that distance (one level coarser for every
     `lvl_steps` samples the ray has taken, so long rays through faint tissue stay cheap),
  2. skips the whole brick if its max is below the window (empty-space skipping, no fetch needed),
  3. looks the brick up in the page table; if it is not resident it flags a request for it and falls
     back to the nearest coarser resident level (coarse levels are pinned, so this ends),
  4. interpolates the uint8 voxels trilinearly, looking up the neighbour brick at brick borders,
  5. composites front to back (emission-absorption) and stops when the ray is opaque.
Steps are one voxel along the ray (times `step`), measured at the chosen level and in the ray direction.
The first sample after a cut face is capped opaque, so cut faces show the data itself (as in the POC).

World frame (mm), centred on the volume: x = i, y = j, z = k. Level voxel (k, j, i) covers
[v * spacing, (v + 1) * spacing) from the volume's min corner; samples are cell-centred.
Page table entries: slot >= 0 resident, -1 stored on disk but not resident, -2 empty brick.
"""
import os
import sysconfig

import triton
import triton.language as tl

# Triton compiles a small C launcher against Python.h. The container's Python 3.12 has no headers,
# so use the copy unpacked from the deadsnakes python3.12-dev packages into data/ (git-ignored).
if not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")):
    _h = os.path.join(os.path.dirname(__file__), "..", "data", "py312-headers", "usr", "include")
    os.environ["CPATH"] = os.pathsep.join(filter(None, [_h, os.path.join(_h, "python3.12"), os.environ.get("CPATH")]))

BZ, BY, BX = 16, 128, 128
BVOX = BZ * BY * BX


@triton.jit
def _level(lvf, lvi, lv, pz, py, px, vminz, vminy, vminx):
    """Spacing, grid, page offset and the brick holding point p at level lv."""
    sz = tl.load(lvf + lv * 4 + 0)
    sy = tl.load(lvf + lv * 4 + 1)
    sx = tl.load(lvf + lv * 4 + 2)
    gz = tl.load(lvi + lv * 4 + 0)
    gy = tl.load(lvi + lv * 4 + 1)
    gx = tl.load(lvi + lv * 4 + 2)
    off = tl.load(lvi + lv * 4 + 3)
    bz = tl.floor((pz - vminz) / (sz * 16)).to(tl.int32)
    by = tl.floor((py - vminy) / (sy * 128)).to(tl.int32)
    bx = tl.floor((px - vminx) / (sx * 128)).to(tl.int32)
    ing = (bz >= 0) & (bz < gz) & (by >= 0) & (by < gy) & (bx >= 0) & (bx < gx)
    pidx = off + (bz * gy + by) * gx + bx
    return sz, sy, sx, gz, gy, gx, off, bz, by, bx, ing, pidx


@triton.jit
def _corner(atlas, page, cz, cy, cx, bz, by, bx, smain, gz, gy, gx, off, ok):
    """Voxel (cz, cy, cx) of the resolved level; the neighbour brick is looked up at brick borders,
    and a neighbour that is not resident is replaced by clamping into the main brick."""
    cbz = cz >> 4
    cby = cy >> 7
    cbx = cx >> 7
    same = (cbz == bz) & (cby == by) & (cbx == bx)
    cing = (cbz >= 0) & (cbz < gz) & (cby >= 0) & (cby < gy) & (cbx >= 0) & (cbx < gx)
    s = tl.load(page + off + (cbz * gy + cby) * gx + cbx, mask=ok & cing & (same == 0), other=-2)
    s = tl.where(same, smain, s)
    clamp = s == -1
    cz = tl.where(clamp, tl.minimum(tl.maximum(cz, bz * 16), bz * 16 + 15), cz)
    cy = tl.where(clamp, tl.minimum(tl.maximum(cy, by * 128), by * 128 + 127), cy)
    cx = tl.where(clamp, tl.minimum(tl.maximum(cx, bx * 128), bx * 128 + 127), cx)
    s = tl.where(clamp, smain, s)
    addr = s.to(tl.int64) * 262144 + ((cz & 15) * 16384 + (cy & 127) * 128 + (cx & 127))
    return tl.load(atlas + addr, mask=ok & (s >= 0), other=0).to(tl.float32)


# integer arguments that change between frames must not trigger a recompile (Triton specialises
# ints that equal 1 or are multiples of 16)
@triton.jit(do_not_specialize=["W", "H", "NL", "use_oct", "thr", "frame", "max_steps", "lvl_steps", "lv_min"])
def raycast_kernel(out, nsteps, atlas, page, bmax, req, used, submax, lvf, lvi,
                   W, H, NL,
                   ex, ey, ez, fx, fy, fz, rx, ry, rz, ux, uy, uz,
                   vminx, vminy, vminz,
                   blox, bloy, bloz, bhix, bhiy, bhiz,
                   olox, oloy, oloz, ohix, ohiy, ohiz, use_oct,
                   camc, pixk, lod_bias, lv_min,
                   win_lo, win_inv, dens, face, lut, step, thr, frame, max_steps, lvl_steps,
                   TW: tl.constexpr, TH: tl.constexpr):
    pid = tl.program_id(0)
    ntx = tl.cdiv(W, TW)
    lane = tl.arange(0, TW * TH)
    col = (pid % ntx) * TW + lane % TW
    row = (pid // ntx) * TH + lane // TW
    valid = (col < W) & (row < H)

    # ray direction
    u = (col.to(tl.float32) + 0.5) / W * 2 - 1
    v = (row.to(tl.float32) + 0.5) / H * 2 - 1
    dx = fx + u * rx - v * ux
    dy = fy + u * ry - v * uy
    dz = fz + u * rz - v * uz
    inv_n = 1.0 / tl.sqrt(dx * dx + dy * dy + dz * dz)
    dx = dx * inv_n
    dy = dy * inv_n
    dz = dz * inv_n
    dx = tl.where(tl.abs(dx) < 1e-9, 1e-9, dx)
    dy = tl.where(tl.abs(dy) < 1e-9, 1e-9, dy)
    dz = tl.where(tl.abs(dz) < 1e-9, 1e-9, dz)
    ivx = 1.0 / dx
    ivy = 1.0 / dy
    ivz = 1.0 / dz

    # clip box -> [tnear, tfar]; removed octant -> [oin, oout); camera cut raises tnear
    t0 = (blox - ex) * ivx
    t1 = (bhix - ex) * ivx
    tnear = tl.minimum(t0, t1)
    tfar = tl.maximum(t0, t1)
    t0 = (bloy - ey) * ivy
    t1 = (bhiy - ey) * ivy
    tnear = tl.maximum(tnear, tl.minimum(t0, t1))
    tfar = tl.minimum(tfar, tl.maximum(t0, t1))
    t0 = (bloz - ez) * ivz
    t1 = (bhiz - ez) * ivz
    tnear = tl.maximum(tnear, tl.minimum(t0, t1))
    tfar = tl.minimum(tfar, tl.maximum(t0, t1))
    tnear = tl.maximum(tnear, 0.0)
    ddotf = dx * fx + dy * fy + dz * fz
    tnear = tl.maximum(tnear, (camc - (ex * fx + ey * fy + ez * fz)) / ddotf)

    t0 = (olox - ex) * ivx
    t1 = (ohix - ex) * ivx
    oin = tl.minimum(t0, t1)
    oout = tl.maximum(t0, t1)
    t0 = (oloy - ey) * ivy
    t1 = (ohiy - ey) * ivy
    oin = tl.maximum(oin, tl.minimum(t0, t1))
    oout = tl.minimum(oout, tl.maximum(t0, t1))
    t0 = (oloz - ez) * ivz
    t1 = (ohiz - ez) * ivz
    oin = tl.maximum(oin, tl.minimum(t0, t1))
    oout = tl.minimum(oout, tl.maximum(t0, t1))
    has_oct = (oin < oout) & (use_oct != 0)

    t = tnear
    active = valid & (tnear < tfar)
    T = tl.full([TW * TH], 1.0, tl.float32)
    cr = tl.zeros([TW * TH], tl.float32)
    cg = tl.zeros([TW * TH], tl.float32)
    cb = tl.zeros([TW * TH], tl.float32)
    first = tl.full([TW * TH], 1, tl.int32)
    steps = tl.zeros([TW * TH], tl.int32)
    prev_s = tl.full([TW * TH], -3, tl.int32)

    while tl.max(active.to(tl.int32), 0) > 0:
        t = tl.where(has_oct & (t >= oin) & (t < oout), oout + 1e-6, t)
        active = active & (t < tfar)
        pz = ez + dz * t
        py = ey + dy * t
        px = ex + dx * t

        # 1. level from the pixel footprint; a ray that is still transparent after every `lvl_steps`
        #    samples goes one level coarser (bounds the cost of long rays through faint tissue)
        lv = tl.floor(tl.log2(tl.maximum(t * pixk, 1e-12)) + lod_bias).to(tl.int32) + steps // lvl_steps
        lv = tl.minimum(tl.maximum(lv, lv_min), NL - 1)                 # lv_min: finest level ever used
        sz, sy, sx, gz, gy, gx, off, bz, by, bx, ing, pidx = _level(lvf, lvi, lv, pz, py, px,
                                                                    vminz, vminy, vminx)
        # step: `step` voxels along the ray at this level (sections are 15x thicker than in-plane pixels,
        # so a ray along the section axis takes 15x longer steps than one in the plane)
        dt = step / tl.maximum(tl.maximum(tl.abs(dx) / sx, tl.abs(dy) / sy), tl.abs(dz) / sz)

        # 2. empty-space skipping: jump to the brick exit
        mx = tl.load(bmax + pidx, mask=active & ing, other=0).to(tl.int32)
        empty = mx < thr
        lzb = vminz + bz.to(tl.float32) * (sz * 16)
        lyb = vminy + by.to(tl.float32) * (sy * 128)
        lxb = vminx + bx.to(tl.float32) * (sx * 128)
        tex = (tl.where(dx > 0, lxb + sx * 128, lxb) - px) * ivx
        tex = tl.minimum(tex, (tl.where(dy > 0, lyb + sy * 128, lyb) - py) * ivy)
        tex = tl.minimum(tex, (tl.where(dz > 0, lzb + sz * 16, lzb) - pz) * ivz)
        tskip = t + tl.maximum(tex, 0.0) + 0.05 * sy

        # 3. page lookup, miss request, fallback to coarser resident levels
        dense = active & ing & (empty == 0)
        s = tl.load(page + pidx, mask=dense, other=-2)
        miss = dense & (s == -1)
        tl.store(req + pidx, tl.full([TW * TH], 1, tl.uint8), mask=miss)
        need = miss & (lv < NL - 1)
        lr = lv
        while tl.max(need.to(tl.int32), 0) > 0:
            lr = tl.where(need, lr + 1, lr)
            sz2, sy2, sx2, gz2, gy2, gx2, off2, bz2, by2, bx2, ing2, pidx2 = _level(
                lvf, lvi, lr, pz, py, px, vminz, vminy, vminx)
            s2 = tl.load(page + pidx2, mask=need & ing2, other=-2)
            sz = tl.where(need, sz2, sz)
            sy = tl.where(need, sy2, sy)
            sx = tl.where(need, sx2, sx)
            gz = tl.where(need, gz2, gz)
            gy = tl.where(need, gy2, gy)
            gx = tl.where(need, gx2, gx)
            off = tl.where(need, off2, off)
            bz = tl.where(need, bz2, bz)
            by = tl.where(need, by2, by)
            bx = tl.where(need, bx2, bx)
            s = tl.where(need, s2, s)
            need = need & (s2 == -1) & (lr < NL - 1)
        tl.store(used + s, tl.zeros([TW * TH], tl.int32) + frame, mask=dense & (s >= 0) & (s != prev_s))
        prev_s = tl.where(dense, s, prev_s)

        # second-level skip: the 8 x 16 x 16 sub-block of the resident brick holding p is empty
        vz = tl.floor((pz - vminz) / sz).to(tl.int32)
        vy = tl.floor((py - vminy) / sy).to(tl.int32)
        vx = tl.floor((px - vminx) / sx).to(tl.int32)
        sbz = tl.minimum(tl.maximum(vz - bz * 16, 0), 15) >> 3       # brick-local, so float rounding at a
        sby = tl.minimum(tl.maximum(vy - by * 128, 0), 127) >> 4     # brick face cannot pick the neighbour's
        sbx = tl.minimum(tl.maximum(vx - bx * 128, 0), 127) >> 4     # sub-block
        smx = tl.load(submax + s.to(tl.int64) * 128 + (sbz * 8 + sby) * 8 + sbx, mask=dense & (s >= 0), other=255)
        subempty = dense & (s >= 0) & (smx.to(tl.int32) < thr)
        lzs = vminz + (bz * 16 + sbz * 8).to(tl.float32) * sz
        lys = vminy + (by * 128 + sby * 16).to(tl.float32) * sy
        lxs = vminx + (bx * 128 + sbx * 16).to(tl.float32) * sx
        tes = (tl.where(dx > 0, lxs + sx * 16, lxs) - px) * ivx
        tes = tl.minimum(tes, (tl.where(dy > 0, lys + sy * 16, lys) - py) * ivy)
        tes = tl.minimum(tes, (tl.where(dz > 0, lzs + sz * 8, lzs) - pz) * ivz)
        tskip = tl.where(subempty, t + tl.maximum(tes, 0.0) + 0.05 * sy, tskip)
        empty = empty | subempty
        dense = dense & (subempty == 0)

        # 4. trilinear interpolation at the resolved level
        qz = (pz - vminz) / sz - 0.5
        qy = (py - vminy) / sy - 0.5
        qx = (px - vminx) / sx - 0.5
        z0f = tl.floor(qz)
        y0f = tl.floor(qy)
        x0f = tl.floor(qx)
        wz = qz - z0f
        wy = qy - y0f
        wx = qx - x0f
        z0 = z0f.to(tl.int32)
        y0 = y0f.to(tl.int32)
        x0 = x0f.to(tl.int32)
        # fast path: every sample of this warp has all 8 corners inside its own brick
        inner = (z0 >= bz * 16) & (z0 < bz * 16 + 15) & (y0 >= by * 128) & (y0 < by * 128 + 127) \
            & (x0 >= bx * 128) & (x0 < bx * 128 + 127) & (s >= 0)
        if tl.min((inner | (dense == 0)).to(tl.int32), 0) == 1:
            base = atlas + s.to(tl.int64) * 262144 + ((z0 & 15) * 16384 + (y0 & 127) * 128 + (x0 & 127))
            c000 = tl.load(base, mask=dense, other=0).to(tl.float32)
            c001 = tl.load(base + 1, mask=dense, other=0).to(tl.float32)
            c010 = tl.load(base + 128, mask=dense, other=0).to(tl.float32)
            c011 = tl.load(base + 129, mask=dense, other=0).to(tl.float32)
            c100 = tl.load(base + 16384, mask=dense, other=0).to(tl.float32)
            c101 = tl.load(base + 16385, mask=dense, other=0).to(tl.float32)
            c110 = tl.load(base + 16512, mask=dense, other=0).to(tl.float32)
            c111 = tl.load(base + 16513, mask=dense, other=0).to(tl.float32)
        else:
            c000 = _corner(atlas, page, z0, y0, x0, bz, by, bx, s, gz, gy, gx, off, dense)
            c001 = _corner(atlas, page, z0, y0, x0 + 1, bz, by, bx, s, gz, gy, gx, off, dense)
            c010 = _corner(atlas, page, z0, y0 + 1, x0, bz, by, bx, s, gz, gy, gx, off, dense)
            c011 = _corner(atlas, page, z0, y0 + 1, x0 + 1, bz, by, bx, s, gz, gy, gx, off, dense)
            c100 = _corner(atlas, page, z0 + 1, y0, x0, bz, by, bx, s, gz, gy, gx, off, dense)
            c101 = _corner(atlas, page, z0 + 1, y0, x0 + 1, bz, by, bx, s, gz, gy, gx, off, dense)
            c110 = _corner(atlas, page, z0 + 1, y0 + 1, x0, bz, by, bx, s, gz, gy, gx, off, dense)
            c111 = _corner(atlas, page, z0 + 1, y0 + 1, x0 + 1, bz, by, bx, s, gz, gy, gx, off, dense)
        c00 = c000 + (c001 - c000) * wx
        c01 = c010 + (c011 - c010) * wx
        c10 = c100 + (c101 - c100) * wx
        c11 = c110 + (c111 - c110) * wx
        c0 = c00 + (c01 - c00) * wy
        c1 = c10 + (c11 - c10) * wy
        val = c0 + (c1 - c0) * wz

        # 5. front-to-back compositing, opaque cap on the first sample after a cut face
        x = tl.minimum(tl.maximum((val - win_lo) * win_inv, 0.0), 1.0)
        x = tl.where(dense, x, 0.0)
        a = 1.0 - tl.exp(-dens * x * dt / 0.01)
        a = tl.where((first == 1) & (x > 0), tl.maximum(a, face), a)
        # colour: 256-entry RGB table (colormaps.py), linear between entries
        fi = x * 255.0
        i0 = tl.minimum(fi.to(tl.int32), 254)
        fr = fi - i0.to(tl.float32)
        l0 = lut + i0 * 3
        w = tl.where(x > 0, T * a, 0.0)
        r0 = tl.load(l0)
        g0 = tl.load(l0 + 1)
        b0 = tl.load(l0 + 2)
        cr += w * (r0 + (tl.load(l0 + 3) - r0) * fr)
        cg += w * (g0 + (tl.load(l0 + 4) - g0) * fr)
        cb += w * (b0 + (tl.load(l0 + 5) - b0) * fr)
        T = T * (1.0 - a)
        first = tl.where(active, 0, first)
        t = tl.where(empty, tskip, t + dt)
        steps += 1
        active = active & (t < tfar) & (T > 0.02) & (steps < max_steps)

    pix = row * W + col
    hw = W * H
    tl.store(nsteps + pix, steps, mask=valid)                 # samples per ray, for profiling
    tl.store(out + pix, (tl.minimum(cr, 1.0) * 255).to(tl.uint8), mask=valid)
    tl.store(out + hw + pix, (tl.minimum(cg, 1.0) * 255).to(tl.uint8), mask=valid)
    tl.store(out + 2 * hw + pix, (tl.minimum(cb, 1.0) * 255).to(tl.uint8), mask=valid)
