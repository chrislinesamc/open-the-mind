# server

Bounded-memory GPU renderer and frame streaming.

```
source .venv/bin/activate
python server/app.py data/bigbrain20_png.ome.zarr 1729 0.0.0.0 [--pool-gb 4] [--host-gb 8] [--finest 0] [--raw <file.raw>]
```

Then open `http://<host>:1729`. Each server is one process with its own fixed pool; several can share a GPU.

| File | Role |
|---|---|
| `raycast.py` | Fused Triton kernel, one thread per ray |
| `brick_cache.py` | Fixed GPU brick pool, page table, LRU eviction, priority loader, host cache of compressed bricks |
| `renderer.py` | Camera and cut state → kernel → box outline → JPEG on the GPU (nvJPEG) |
| `colormaps.py` | Colour maps as 256-entry RGB tables (grayscale, sepia, hot, viridis, magma, bone, violet stain, custom), with gamma |
| `monitor.py` | Live readings for the page: GPU memory of the process and GPU load (NVML), system RAM, RSS, frames per second |
| `app.py` | HTTP + WebSocket server: `/`, `/info`, `/stats`, `/lut?<map>`, `/render?<view>`, `/ws` |

## How a frame is made

1. `apply_uploads` copies bricks that finished loading into free or least-recently-used pool slots and updates the page table. This runs between frames, so a frame never reads a half-written slot.
2. The kernel marches each ray. For every sample it:
   - picks a level from the pixel footprint;
   - skips empty bricks (brick max below the window) and empty 8×16×16 sub-blocks;
   - looks the brick up in the page table, falling back to coarser resident levels on a miss and flagging the brick it wanted;
   - interpolates trilinearly, with neighbour lookups at brick borders;
   - composites front to back and stops when the ray is opaque.
   Steps are one voxel along the ray, so with 4 µm pixels and 60 µm sections, rays along the section axis take 15× longer steps than in-plane rays. A ray that is still transparent after 64 samples goes one level coarser (`budget`).
3. `collect` reads the miss flags and queues them, newest frame first and coarse first. Loader threads read and zstd-decode the bricks straight into pinned staging memory. When staging is full they block on a condition variable until uploads free slots. They must not poll: 64 threads polling every 1 ms held the GIL often enough to stretch render frames from 5 ms to about 43 ms for seconds after each zoom or pan.

The coarse levels (for example L3 and up, 2,976 bricks, on a 138.5 GB volume) are pinned, so every ray always finds data.

## Memory (fixed, independent of the data size)

Nothing is loaded whole. The GPU holds only the bricks the current views need, in a pool of fixed size.

| Item | Size | Flag |
|---|---|---|
| GPU brick pool | default 4 GB = 16,384 bricks of 256 KB | `--pool-gb` |
| Coarse levels kept in the pool, never evicted | the coarsest levels that fit in a share of the pool (L3+ = 2,976 bricks on a 138.5 GB volume) | `--pin-share` (default 0.25) |
| Finest level ever used | default L0 (native). `--finest 1` stops at half resolution and never reads L0, the largest level | `--finest` |
| GPU page table and per-brick max | 5 bytes per brick of the pyramid (3.6 MB for a 138.5 GB volume) | |
| GPU sub-block max | 128 bytes per pool slot | |
| GPU overhead (CUDA context, kernels, frame buffers) | about 0.9 GB | |
| Host cache of compressed bricks | default 8 GB, LRU | `--host-gb` |
| Pinned staging for uploads | default 256 MB | `--staging-mb` |
| Loader threads | default 64 | `--threads` |
| Level 0 straight from the uncompressed raw volume instead of the pyramid (to compare) | off | `--raw` |

Readers keep one file descriptor per thread and shard file, so the server raises its open-file soft limit to the hard
limit at start (`bricks.raise_fd_limit`); a terminal's default of 1024 runs out within seconds otherwise.

Example for a small GPU and little RAM: `python server/app.py data/bigbrain20_png.ome.zarr 1729 0.0.0.0 --pool-gb 1 --host-gb 2 --finest 1`.
With a 1 GB pool the process used about 1.9 GB of GPU memory in total.

## Measured (A100, 1920 × 1080)

On a 138.5 GB whole-brain volume (16000 × 16000 × 541, 4 × 4 × 60 µm):

The kernel takes 0.5–4.4 ms per frame on typical views. With nvJPEG encoding, a frame takes 2.3–6.4 ms
(`scripts/bench_render.py`). Hard views with the camera inside the tissue take up to about 19 ms. Over 300 random
zoomed views the GPU memory stayed at 4.01 GB while 32,000 bricks were evicted (`scripts/tour.py`).

## Measured (GB10, 1920 × 1080, `scripts/stream_bench.py`)

The same 30 s camera flight, paced at 60 fps, with the pool fixed while the data grows:

| Data | Pool | fps | GPU memory | Kernel (median) | CPU cores | Evicted |
|---|---|---|---|---|---|---|
| whole brain, 4 × 4 × 60 µm (26 GB of tissue) | 20 GB | 60.0 | 20.58 GB | 1.2 ms | 0.4 | 0 |
| BigBrain 20 µm (107 GB of tissue) | 20 GB | 60.0 | 20.58 GB | 1.8 ms | 0.9 | 0 |
| BigBrain dense, uncompressed, skipping off (278 GB) | 20 GB | 59.4 | 20.58 GB | 4.5 ms | 1.0 | 0 |
| BigBrain 20 µm | 4 GB | 60.0 | 4.58 GB | 1.8 ms | 0.9 | 29,807 |
| BigBrain dense, skipping off (278 GB) | 4 GB | 59.8 | 4.58 GB | 3.3 ms | 0.9 | 30,932 |

GPU memory is always the pool plus 0.58 GB, whatever the data size; a 4 GB pool performs like a 20 GB one.
During continuous motion 38–64 % of frames are fully sharp (the rest use a coarser level for a frame or two while
bricks arrive).

## WebSocket protocol

- The client sends `{"type": "view", "seq", "window", "view": {…}}` whenever its view changes, and `{"type": "ack", "n"}` for each frame it receives.
- The server keeps at most `window` frames unacknowledged. It always renders the newest view, so a slow link drops views instead of queueing them.
- Each frame is one binary message: a 4-byte header length, a JSON header (timings, misses, memory every 10th frame), then the JPEG.

## Python headers

Triton builds a small C launcher against `Python.h`. If the system Python has no headers, `raycast.py` points
`CPATH` at `data/py312-headers`, a copy unpacked from the `python3.12-dev` packages (`apt-get download
libpython3.12-dev python3.12-dev`, then `dpkg-deb -x`; no root needed). Unpack the package for the machine's
architecture: on the GB10 (aarch64) Triton needs `aarch64-linux-gnu/python3.12/pyconfig.h`.
