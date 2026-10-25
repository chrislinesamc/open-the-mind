"""Frame renderer: camera and cut state -> one fused kernel launch over the brick pool -> outline
overlay -> JPEG on the GPU (nvJPEG). Same view parameters as the POC page.
"""
import math
import time

import numpy as np
import torch
from torchvision.io import encode_jpeg

import colormaps
from brick_cache import BrickCache
from raycast import raycast_kernel

TANF = math.tan(math.radians(15))          # half of the 30 degree vertical field of view

DEFAULTS = dict(w=960, h=720, az=35.0, el=25.0, dist=0.0, tx=0.0, ty=0.0, tz=0.0,
                cx=0.0, cy=0.0, cz=0.0, fx=0, fy=0, fz=0, ox=1, oy=1, oz=1, cam=0, camd=0.0, mode=0, auto=1,
                lo=8.0, hi=200.0, dens=1.5, face=1.0, step=1.0, lod=0.0, q=85, box=1, skip=1, budget=64, colour=0,
                gamma=1.0, c0=0x000000, c1=0xFFFFFF)                # colour map (colormaps.py), gamma, custom colours


def parse(q):
    """Query dict (strings or numbers) -> typed view parameters."""
    p = {k: type(dv)(float(q[k])) if k in q else dv for k, dv in DEFAULTS.items()}
    p["w"], p["h"] = max(16, min(p["w"], 3840)), max(16, min(p["h"], 2160))
    return p


class Renderer:
    def __init__(self, root, pool_gb=4.0, host_gb=8.0, **cache_opts):
        self.cache = BrickCache(root, pool_gb=pool_gb, host_gb=host_gb, **cache_opts)
        L0 = self.cache.levels[0]
        nk, nj, ni = L0["shape"]
        sk, sj, si = L0["spacing"]
        self.size = np.array([ni * si, nj * sj, nk * sk])          # x, y, z extent in mm
        self.s0 = min(si, sj)
        self.dev = self.cache.atlas.device
        self.luts = {}
        # tissue box: bricks of level 1 that hold any data (~1 mm). Cuts, outline and the home view use
        # it instead of the volume box, which is mostly background.
        n = min(1, self.cache.NL - 1)
        L = self.cache.levels[n]
        occ = np.argwhere(self.cache.pyr.vmax[n] > 0)[:, ::-1]                 # x, y, z brick indices
        bmm = np.array([128 * L["spacing"][2], 128 * L["spacing"][1], 16 * L["spacing"][0]])
        self.lo0 = np.maximum(-self.size / 2 + occ.min(0) * bmm, -self.size / 2)
        self.hi0 = np.minimum(-self.size / 2 + (occ.max(0) + 1) * bmm, self.size / 2)
        self.radius = float(np.linalg.norm(self.hi0 - self.lo0)) / 2
        self.centre = (self.lo0 + self.hi0) / 2
        self.meta = dict(name=self.cache.pyr.meta["name"], shape_kji=L0["shape"], spacing_mm_kji=L0["spacing"],
                         size_mm_xyz=self.size.tolist(), tissue_box_mm=[self.lo0.tolist(), self.hi0.tolist()],
                         levels=len(self.cache.levels), default_dist=self.radius / math.sin(math.radians(15)) * 1.05,
                         planes=self.cache.pyr.meta.get("planes"))         # slider names for x, y, z, if the dataset has them

    def camera(self, p):
        az, el = math.radians(p["az"]), math.radians(p["el"])
        dist = p["dist"] or self.meta["default_dist"]
        target = self.centre + np.array([p["tx"], p["ty"], p["tz"]])      # pan is relative to the tissue centre
        eye = target + dist * np.array([math.cos(el) * math.sin(az), -math.sin(el), math.cos(el) * math.cos(az)])
        fwd = (target - eye) / np.linalg.norm(target - eye)
        right = np.cross(fwd, [0.0, -1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return eye, fwd, right, up, target

    def regions(self, p, eye):
        """Clip box and removed box. Cut positions are fractions of the tissue box; ox/oy/oz switch
        each plane on. Mode 0 crops the volume to the planes. Mode 1 opens it: the brain stays whole
        and only the region on the camera's side of every enabled plane is removed, so one plane
        removes a half, two a wedge, three a corner, exposing the sagittal, coronal and axial faces."""
        c = np.array([p["cx"], p["cy"], p["cz"]]) * (self.hi0 - self.lo0)
        on = np.array([p["o" + k] for k in "xyz"]) != 0
        lo, hi = self.lo0.copy(), self.hi0.copy()
        olo, ohi = np.full(3, 1.0), np.full(3, -1.0)            # empty: nothing removed
        if p["mode"] == 0:
            for a, k in enumerate("xyz"):
                if not on[a]:
                    continue
                if p["f" + k]:
                    hi[a] = self.hi0[a] - c[a]
                else:
                    lo[a] = self.lo0[a] + c[a]
        elif on.any():
            plane = self.lo0 + c
            if p["auto"]:
                side = np.where(eye > plane, 1.0, -1.0)
            else:
                side = np.array([-1.0 if p["f" + k] else 1.0 for k in "xyz"])
            m = 1.0                                              # mm beyond the tissue box: no sliver at its faces
            olo = np.where(on & (side > 0), plane, self.lo0 - m)
            ohi = np.where(on & (side < 0), plane, self.hi0 + m)
        return lo, hi, olo, ohi

    @torch.inference_mode()
    def draw(self, p):
        """One frame as a (3, H, W) uint8 GPU image; brick uploads before, miss collection after."""
        W, H = p["w"], p["h"]
        cache = self.cache
        uploads = cache.apply_uploads()
        eye, fwd, right, up, target = self.camera(p)
        lo, hi, olo, ohi = self.regions(p, eye)
        camc = float((target + fwd * p["camd"] * self.radius) @ fwd) if p["cam"] else -1e30
        r = right * TANF * W / H
        u = up * TANF
        out = torch.empty((3, H, W), dtype=torch.uint8, device=self.dev)
        self.nsteps = torch.empty((H, W), dtype=torch.int32, device=self.dev)
        TW, TH = 8, 4
        grid = (triton_cdiv(W, TW) * triton_cdiv(H, TH),)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        raycast_kernel[grid](
            out, self.nsteps, cache.atlas, cache.page, cache.bmax, cache.req, cache.used, cache.submax, cache.lvf, cache.lvi,
            W, H, cache.NL, *eye, *fwd, *r, *u, *(-self.size / 2),
            *lo, *hi, *olo, *ohi, int(p["mode"] == 1), camc,
            2 * TANF / H / self.s0, p["lod"], cache.finest,
            p["lo"], 1.0 / max(p["hi"] - p["lo"], 1.0), p["dens"], p["face"], self.lut(p),
            p["step"],
            int(math.floor(p["lo"])) + 1 if p["skip"] else 0, cache.frame, 20000, max(p["budget"], 1),
            TW=TW, TH=TH, num_warps=1)
        e1.record()
        if p["box"]:
            self.outline(out, eye, fwd, right, up, W, H, (self.lo0, self.hi0), (90, 90, 90))
            if p["mode"] == 0:
                self.outline(out, eye, fwd, right, up, W, H, (lo, hi), (230, 140, 40))
            else:
                self.outline(out, eye, fwd, right, up, W, H, (np.maximum(olo, self.lo0), np.minimum(ohi, self.hi0)),
                             (230, 140, 40))
        misses = cache.collect()                      # syncs, so the kernel time below is ready
        cache.frame += 1
        return out, dict(gpu_ms=round(e0.elapsed_time(e1), 2), misses=misses, uploads=uploads, w=W, h=H)

    def lut(self, p):
        """Colour table of the view on the GPU; a few recent ones are kept, so changing maps costs nothing."""
        key = (int(p["colour"]), round(p["gamma"], 3), int(p["c0"]), int(p["c1"]))
        if key not in self.luts:
            if len(self.luts) > 32:
                self.luts.clear()
            self.luts[key] = torch.from_numpy(colormaps.lut(*key)).reshape(-1).to(self.dev)
        return self.luts[key]

    def render(self, p):
        """One frame as JPEG bytes (nvJPEG on the GPU) and its stats."""
        t = time.perf_counter()
        out, stats = self.draw(p)
        data = encode_jpeg(out, quality=int(max(10, min(p["q"], 95)))).cpu().numpy().tobytes()
        stats.update(bytes=len(data), frame_ms=round((time.perf_counter() - t) * 1000, 2))
        return data, stats

    def outline(self, img, eye, fwd, right, up, W, H, box, colour):
        """Thin box outline drawn into the (3, H, W) GPU image."""
        a, b = box
        if np.any(a > b):
            return
        corners = np.array([[(a, b)[i & 1][0], (a, b)[i >> 1 & 1][1], (a, b)[i >> 2 & 1][2]] for i in range(8)])
        rel = corners - eye
        z = rel @ fwd
        sx = (rel @ right) / (np.maximum(z, 1e-6) * TANF * W / H)
        sy = (rel @ up) / (np.maximum(z, 1e-6) * TANF)
        px = (sx + 1) * W / 2
        py = (1 - sy) * H / 2
        segs = []
        for i in range(8):
            for bit in (1, 2, 4):
                j = i | bit
                if j != i and z[i] > 1e-3 and z[j] > 1e-3:
                    n = int(min(max(abs(px[j] - px[i]), abs(py[j] - py[i])), 8000)) + 2
                    tt = np.linspace(0, 1, n)
                    segs.append(np.stack([px[i] + (px[j] - px[i]) * tt, py[i] + (py[j] - py[i]) * tt], 1))
        if not segs:
            return
        pts = torch.from_numpy(np.concatenate(segs)).to(self.dev)
        x, y = pts[:, 0].long(), pts[:, 1].long()
        keep = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        x, y = x[keep], y[keep]
        for ch, v in enumerate(colour):
            img[ch, y, x] = v


def triton_cdiv(a, b):
    return (a + b - 1) // b
