# scripts

| Script | Use |
|---|---|
| `png_to_zarr.py` | Builds a pyramid from a folder of numbered 8-bit PNG sections: detects the numbering, image size and a white background, writes the config, builds within `--mem-gb` |
| `nifti_to_zarr.py` | Builds a pyramid from a `.nii`/`.nii.gz` volume: streams it to a raw uint8 volume (scaling other types to 0–255, inverting a white background, reorienting to coronal sections), writes the config, builds within `--mem-gb`, then removes the raw file unless `--keep-raw` |
| `check_pyramid.py` | Spot-check a built pyramid against the raw source |
| `bench_render.py` | Steady-state frame time and time-to-sharp on a set of views |
| `tour.py` | Hundreds of random zoomed views: memory must stay flat while bricks are evicted |
| `cold_read.py` | Cold random brick reads (page cache dropped) vs thread count: the first-visit cost |
| `stream_bench.py` | The same 30 s camera flight at 60 fps on any pyramid and pool: fps, frame and kernel time, sharp frames, GPU memory and load, RAM, CPU, disk read |
| `check_png_pyramid.py` | Proves a pyramid built from section images is lossless: every level-0 voxel against the decoded (inverted) images, plus sections read through zarr-python |
| `fetch_ftp_sections.py` | Downloads a directory of section images (default: BigBrain's 7,404 coronal 20 µm PNGs) with parallel connections, resumable |
| `export_raw.py` | Writes level 0 of a pyramid back out as the raw volume (a lossless copy), about 3 min for 138.5 GB |
| `record_demo.js` | Scripted demo in headless Chromium (rotate, zoom, pan, cuts, cutaway, camera cut, window, density). Saves every streamed JPEG and a screencast of the page |
| `compose_demo.py` | Turns a `record_demo.js` recording into a 1920×1080 60 fps H.264 video with captions and an fps meter |

`record_demo.js` needs Playwright (tested with 1.47) and `compose_demo.py` needs ffmpeg (e.g. from the
`imageio-ffmpeg` wheel); neither is in `requirements.txt`.
