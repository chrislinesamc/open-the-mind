# pyramid

Out-of-core brick pyramid builder. Pyramids are standard OME-Zarr 0.5 groups (`omezarr.py`); the brick and level
logic that does not depend on the file layout is in `bricks.py`.

```
CUDA_VISIBLE_DEVICES=0 python pyramid/build_pyramid.py configs/bigbrain20_png.json data/bigbrain20_png.ome.zarr [--mem-gb 2]
python scripts/check_png_pyramid.py configs/bigbrain20_png.json data/bigbrain20_png.ome.zarr   # PNG input
python scripts/check_pyramid.py configs/<dataset>.json data/<dataset>.ome.zarr                 # raw input
```

`scripts/png_to_zarr.py` and `scripts/nifti_to_zarr.py` write the config for a folder of PNGs or a NIfTI file and
call the builder, so most datasets need no hand-written config.

## Memory

The volume is never held. The builder reads one slab (16 sections, one brick row) at a time while the previous one is
on the GPU, so memory does not depend on the number of sections. It does depend on the section size: without a
budget, a slab is 16 whole images (about 600 MB for BigBrain's 6572 × 5711 sections, 4 GB host RAM in all).

`--mem-gb N` caps host RAM and GPU memory (each) at about N GB, whatever the image size:

- Each slab is cut into strips of image rows. A strip holds whole bricks at every level, so strips give exactly the
  same pyramid as whole slabs (`tests/test_pyramid.py::test_memory_budget_strips_match`).
- The isotropic levels, built from the finished level below, are read back in strips too.
- A shard is written as soon as all its bricks are in, so no level waits for a whole row.
- Section images are decoded by a few threads, once per strip (more strips, more decoding).
- Pinned host buffers are sized just under a power of two, because PyTorch rounds them up to one.

Measured on an RTX 5060 laptop GPU with `--mem-gb 2`:

| Input | Peak RAM | Peak GPU memory | Time |
|---|---|---|---|
| 64 BigBrain sections, 6572 × 5711 (4.1 GB RAM, 2.45 GB GPU without the budget) | 1.6 GB | 0.67 GB | 15 s |
| 7,404 BigBrain thumbnails, 500 × 434 | 0.92 GB | 0.16 GB | 12 s |
| BigBrain 100 µm NIfTI, 1392 × 1541 × 1209 (with the conversion) | 1.07 GB | 0.30 GB | 28 s |

## Format: OME-Zarr 0.5 (Zarr v3, sharded)

| Path | Content |
|---|---|
| `zarr.json` | the group: OME `multiscales` (axes in micrometres, scale and translation per level), and under `attributes.pyramid` the level plan (shape, spacing, factor, brick grid) and build stats |
| `<n>/zarr.json` | level n: a uint8 array whose chunk is one 16 × 128 × 128 brick, zstd level 1, inside 16 × 2048 × 2048 shards |
| `<n>/c/<z>/<y>/<x>` | one shard: up to 256 compressed bricks, then the shard index, (offset, size) per brick. Empty bricks are "missing" (2⁶⁴ − 1) and read as 0; shards without data are not written |
| `brick_max/<n>` | per brick: max voxel value (uint8), used for empty-space skipping; other OME-Zarr readers ignore it |

- Any Zarr v3 / OME-Zarr reader opens it (zarr-python, napari, Neuroglancer); `tests/test_pyramid.py` reads it with
  zarr-python and validates it with `ome-zarr-models`.
- The viewer reads the shard indexes once at start into a per-brick (offset, size) table, then fetches any brick with
  one `pread`.
- Coarse levels carry a translation of half the difference in voxel size, so their block means sit over the voxels
  they average.

- Every level uses one brick shape, 16 × 128 × 128 voxels (z, y, x), so one GPU pool slot fits a brick of any level.
- While the in-plane pixel is finer than the section spacing, a level halves y and x only, so storage stays anisotropic (for 4 µm pixels and 60 µm sections: 4 → 8 → 16 → 32 → 64 µm in-plane). After that, every level halves all three axes.
- Voxel values are block means over the valid voxels. A block cut by the data edge, such as the last section, is not darkened by the padding.
- **Reading on Lustre:** open one file descriptor per reader thread (`bricks.Pyramid` does this). Preads through one shared descriptor are serialised by the Lustre client: about 550–640 cold random bricks/s, however many threads. With one descriptor per thread it is 2,500–4,300 bricks/s at 128 threads (`scripts/cold_read.py`). Striping the files over 13 storage targets made no measurable difference.

## Why zstd-1 bricks

We measured codecs on tissue bricks from a 138.5 GB whole-brain volume (16000 × 16000 × 541, 4 × 4 × 60 µm):

| Codec | Ratio on tissue | Decode, MB/s per core | Note |
|---|---|---|---|
| zstd-1 | 1.35 | ~700 | chosen: lossless, fast |
| zstd-9 | 1.37 | ~600 | slow to encode |
| blosc2 + zstd | 1.33–1.39 | ~450 | |
| JPEG XL lossless | 1.26–1.35 | 13–20 | too slow to decode |
| JPEG q90 | 2.06 | ~140 | lossy, PSNR 37 dB |

Nissl tissue at 4 µm is noisy, so no lossless codec gets much past 1.36×, and lossy JPEG only reaches 2× at acceptable
quality. Most of the saving comes from not storing the ~81% of level-0 bricks that are background.

## Example result

For a 138.5 GB whole-brain volume (16000 × 16000 × 541, 4 × 4 × 60 µm):

- Build time: 143 s for the 138.5 GB raw file. Reads use parallel `pread` at about 2 GB/s; the old `memmap` path ran at 140 MB/s.
- Peak memory: 2 pinned slabs (8 GB host) and about 12 GB GPU.

| Level | Voxel (µm) | Shape (k, j, i) | Stored bricks | Size |
|---|---|---|---|---|
| L0 | 4 × 4 × 60 | 541 × 16000 × 16000 | 100,575 / 531,250 | 18.16 GB |
| L1 | 8 × 8 × 60 | 541 × 8000 × 8000 | 26,301 / 134,946 | 4.49 GB |
| L2 | 16 × 16 × 60 | 541 × 4000 × 4000 | 7,180 / 34,816 | 1.11 GB |
| L3 | 32 × 32 × 60 | 541 × 2000 × 2000 | 2,088 / 8,704 | 0.27 GB |
| L4 | 64 × 64 × 60 | 541 × 1000 × 1000 | 704 / 2,176 | 0.07 GB |
| L5–L7 | 128–512 | … | 184 | 0.01 GB |

The full pyramid is 24.1 GB for all levels, against 138.5 GB for the raw level 0 (5.7× smaller). Level 0 matches the raw
file exactly (`tests/test_pyramid.py`, `scripts/check_pyramid.py`).
