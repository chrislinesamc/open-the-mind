# configs

Dataset descriptions (path, shape, spacing, orientation), the input of `pyramid/build_pyramid.py`.
`scripts/png_to_zarr.py` and `scripts/nifti_to_zarr.py` write one next to the pyramid they build, so these are
examples.

- `example_raw.json`: template for a raw `uint8` volume (one C-order file, shape `(sections, rows, columns)`). Set `raw`, `shape_kji`, `spacing_mm_kji`, and the slider names in `planes`. Paths are relative to the project root, where all commands run.
- `bigbrain20_png.json`: BigBrain 2015 at 20 µm, built straight from the FTP's 7,404 coronal PNG sections (`scripts/fetch_ftp_sections.py`): `png_dir`, `png_pattern`, `invert`, 20 µm between sections and 21.17 µm (1200 dpi) in-plane, and the slider names in `planes`.
- Pool size, host-cache size and the finest level are flags of `server/app.py` (`--pool-gb 4`, `--host-gb 8`, `--finest 0` by default; see `python server/app.py -h`).
