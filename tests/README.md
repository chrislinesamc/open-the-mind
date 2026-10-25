# tests

```
python -m pytest tests/ -q
```

| Test | Checks |
|---|---|
| `test_pyramid.py::test_build_matches_reference` | Every brick equals the source (L0) or the block mean of the level below; empty bricks are not stored; per-brick max is right (zstd and uncompressed) |
| `test_pyramid.py::test_reads_with_standard_zarr` | zarr-python reads every level, and the group validates as OME-Zarr 0.5 |
| `test_pyramid.py::test_dense_uncompressed_pyramid` | `--codec none --keep-empty` stores every brick as plain voxels with the same contents |
| `test_pyramid.py::test_png_stack_matches_raw` | A stack of section PNGs (inverted or not) builds the same pyramid as the raw volume |
| `test_pyramid.py::test_memory_budget_strips_match` | A build with a tiny `--mem-gb` (strips of rows at every level, raw and PNG input) equals the whole-slab build brick for brick |
| `test_convert.py::test_png_folder` | `png_to_zarr.py` detects zero-padded numbering, image size and a white background, and the pyramid equals the inverted images |
| `test_convert.py::test_png_folder_with_a_gap` | A missing section number is reported instead of building a wrong volume |
| `test_convert.py::test_nifti_uint8` | `nifti_to_zarr.py` on uint8 data: coronal reorientation (and `--orient none`), spacing, inversion, raw file removed |
| `test_convert.py::test_nifti_other_axis_order_and_int16` | Other axis orders and directions, int16 scaled to 0–255, no inversion of a dark background |
| `test_pyramid.py::test_reads_under_a_low_open_file_limit` | 64 reader threads work under a 64-file soft limit |
| `test_raycast.py::test_matches_dense_reference` | The fused kernel (page table, brick borders, compositing, colour maps) matches a plain dense ray march of level 0 on six views |
| `test_raycast.py::test_skipping_and_fallback` | Misses fall back to coarse levels without holes; empty-space skipping barely changes the image |
| `test_raycast.py::test_small_pool_stays_consistent` | A 40-slot pool under heavy eviction keeps the page table and the slot owners consistent |
| `test_raycast.py::test_finest_level_cap` | `--finest 1` never loads level 0 |
| `test_raycast.py::test_open_one_plane_matches_crop`, `test_open_three_planes_keeps_the_rest` | Opening with one plane equals cropping away that half; opening a corner leaves the rest unchanged |
| `test_raycast.py::test_raw_level0_matches_pyramid` | `--raw` (level 0 from the raw volume) gives the same bricks and image as the pyramid |
