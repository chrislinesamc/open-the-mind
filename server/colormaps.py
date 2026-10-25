"""Colour maps for the ray caster: a 256-entry RGB lookup table (values 0..1), indexed by the windowed voxel value
and interpolated linearly in the kernel. Opacity does not depend on the map, only on the window and density.

Grayscale and sepia are the old per-channel power curves; the others are given by colour stops (viridis and magma
approximate matplotlib's). "gamma" bends the value before the map (< 1 brightens faint tissue); "custom" runs from
colour c0 to c1 (0xRRGGBB).
"""
import numpy as np

NAMES = ["grayscale", "sepia", "hot", "viridis", "magma", "bone", "violet stain", "custom"]

STOPS = {
    "viridis": [(0.0, (0.267, 0.005, 0.329)), (0.125, (0.283, 0.141, 0.458)), (0.25, (0.230, 0.322, 0.546)),
                (0.375, (0.172, 0.448, 0.558)), (0.5, (0.128, 0.567, 0.551)), (0.625, (0.153, 0.678, 0.506)),
                (0.75, (0.369, 0.789, 0.383)), (0.875, (0.678, 0.864, 0.190)), (1.0, (0.993, 0.906, 0.144))],
    "magma": [(0.0, (0.001, 0.000, 0.014)), (0.125, (0.113, 0.065, 0.277)), (0.25, (0.317, 0.071, 0.485)),
              (0.375, (0.512, 0.128, 0.505)), (0.5, (0.716, 0.215, 0.475)), (0.625, (0.904, 0.310, 0.388)),
              (0.75, (0.987, 0.536, 0.382)), (0.875, (0.996, 0.769, 0.534)), (1.0, (0.987, 0.991, 0.750))],
    "bone": [(0.0, (0.0, 0.0, 0.0)), (0.375, (0.319, 0.319, 0.444)), (0.75, (0.652, 0.777, 0.777)), (1.0, (1.0, 1.0, 1.0))],
    "violet stain": [(0.0, (0.0, 0.0, 0.0)), (0.5, (0.36, 0.18, 0.55)), (1.0, (0.95, 0.85, 1.0))],
}


def _stops(stops, v):
    pos = [p for p, _ in stops]
    return np.stack([np.interp(v, pos, [c[i] for _, c in stops]) for i in range(3)], 1)


def _rgb(hex_colour):
    h = int(hex_colour) & 0xFFFFFF
    return np.array([(h >> 16) & 255, (h >> 8) & 255, h & 255]) / 255.0


def lut(index=0, gamma=1.0, c0=0x000000, c1=0xFFFFFF):
    """(256, 3) float32 table for map NAMES[index]."""
    name = NAMES[int(index)] if 0 <= int(index) < len(NAMES) else NAMES[0]
    v = np.linspace(0.0, 1.0, 256) ** max(float(gamma), 0.05)
    if name == "grayscale":
        t = np.stack([v ** 0.95] * 3, 1)
    elif name == "sepia":
        t = np.stack([v ** 0.75, v ** 0.95, v ** 1.25], 1)
    elif name == "hot":
        t = np.clip(np.stack([3 * v, 3 * v - 1, 3 * v - 2], 1), 0, 1)
    elif name == "custom":
        t = _rgb(c0)[None] + (_rgb(c1) - _rgb(c0))[None] * v[:, None]
    else:
        t = _stops(STOPS[name], v)
    return np.ascontiguousarray(t, np.float32)
