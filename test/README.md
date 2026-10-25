# test

One-command trial on public BigBrain data (see "Quick start" in the main README):

```
test/test.sh [nifti|png|slab]
```

It prints the machine's specs, checks that it can run the viewer (NVIDIA GPU of compute capability 7.5+, a driver
for CUDA 12.6+, 2 GB free RAM and GPU memory, disk space, Python, C compiler, headers), installs PyTorch for this
driver and CPU into `.venv`, downloads the data, converts it to a Zarr pyramid within `MEM_GB` (2 GB) and starts the
viewer. Finished steps are skipped on a re-run. Settings are environment variables listed at the top of the script.

| File | What |
|---|---|
| `test.sh` | The trial |
| `fetch_sample_sections.py` | Downloads a short run of full-resolution BigBrain sections (the `slab` dataset), resumable |

To start again from nothing: `rm -rf .venv data/bigbrain_* data/py312-headers`.
