# data (git-ignored)

Datasets and built pyramids. Nothing in here is committed.

| Path | What |
|---|---|
| `bigbrain_100um.nii.gz`, `.json`, `.ome.zarr/` | `test/test.sh nifti`: the whole brain at 100 µm, its config and pyramid (0.6 + 0.7 GB) |
| `bigbrain_thumbs/`, `.json`, `.ome.zarr/` | `test/test.sh png`: the 7,404 coronal PNG thumbnails, config and pyramid (0.43 + 0.49 GB) |
| `bigbrain_slab/`, `.json`, `.ome.zarr/` | `test/test.sh slab`: 64 full-resolution sections, config and pyramid (0.9 + 0.9 GB) |
| `bigbrain20_png/` | BigBrain's 7,404 coronal 20 µm PNG sections from the FTP (`scripts/fetch_ftp_sections.py`, ~60 GB) |
| `bigbrain20_png.ome.zarr/` | BigBrain pyramid built from those sections (`configs/bigbrain20_png.json`) |
| `py312-headers/` | Python 3.12 headers for Triton, used only if the system Python has no `Python.h` |
