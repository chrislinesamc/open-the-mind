<p align="center">
  <img src="docs/brand/logo.png" alt="Open the Mind logo: serial brain sections stacking into a volume" width="180">
</p>

<h1 align="center">Open the Mind</h1>

<p align="center">
  Streaming out-of-core visualisation of high-resolution 3D volumes.
</p>

<p align="center">
  <img src="docs/brand/banner.png" alt="Whole brain rendered on the server GPU, with the octant facing the camera cut away" width="720">
</p>

---

## The problem

Whole-brain histology at cellular resolution runs from terabytes to petabytes. Today's tools either show it as 2D
slices or load the whole volume into memory before rendering. Neither lets a researcher see the whole brain in 3D,
cut it open anywhere and zoom into the cut, from an ordinary browser.

This project builds a viewer for that. The browser shows one 3D view of the whole brain, and the user can:

- **cut it**: with three sliders that move x/y/z planes, or with a corner cutaway that turns to face the camera
  as the brain rotates;
- **zoom in**: the cut faces sharpen from the coarse overview down to the native resolution.

Rendering runs on the server GPU and the browser only receives frames. Memory stays fixed whatever the size of the
data: the unit of work and of memory is what the screen needs, never the whole volume.

## Goals

1. Build a multi-resolution brick pyramid from a registered 3D volume without loading it whole.
2. Render it on the server GPU through a fixed-size brick cache, with volume rendering, x/y/z cut planes and a
   camera-facing cutaway.
3. Stream frames to a browser page with sliders and orbit, pan and zoom, and keep memory flat as the resolution grows.

## Requirements

- Basic Python and NumPy. Some PyTorch or CUDA helps.
- Basic HTML and JavaScript for the browser page, or willingness to learn.
- No neuroscience needed.

## Possible approaches

These are starting points, not a fixed plan.

| Part | Idea | Notes |
|---|---|---|
| Pyramid | Read the volume slab by slab and write compressed, sharded bricks. Downsample 2×2×1 in-plane until the pixel size matches the section spacing, then 2×2×2 | OME-Zarr 0.5 (sharded) or Neuroglancer precomputed. On Lustre, read with parallel `pread`; `np.memmap` reads do not scale |
| Renderer | GPU ray casting: front-to-back compositing, early ray termination, cut planes as kernel arguments | A one-cube prototype renders a 960×720 frame in 25–130 ms on an A100 |
| Brick cache | A fixed GPU brick pool with a page table and least-recently-used eviction. Rays request missing bricks; the coarse level fills in while fine bricks load | Ray-guided virtual memory (GigaVoxels 2009; Hadwiger et al. 2012) |
| Streaming | JPEG frames over HTTP or WebSocket: low resolution while dragging, one sharp frame when the view stops | The A100 has no hardware video encoder |
| Browser page | One 3D view, three cut sliders, the camera cutaway, orbit, pan and zoom | No slice panels |

**Evaluation idea:** measure frame time and round-trip latency at 1080p, the time for a cut face to sharpen after a
zoom, and bytes per frame. Peak GPU and host memory must stay the same from the coarsest level to the finest.

## Data

The starting dataset is brain 222: a whole brain cut into sagittal sections and registered in 3D. It is one `uint8`
volume of 16000 × 16000 × 541 voxels, 4 × 4 µm in-plane and 60 µm between sections, 138.5 GB raw. It lives on the
lab server, and access is through the organisers. Read the raw file directly as C-order `(541, 16000, 16000)`,
because its `.nhdr` header names the wrong `.raw` file.

A public alternative is the [BigBrain](https://bigbrainproject.org) release (Amunts et al., *Science* 2013), with
3D volumes at <https://ftp.bigbrainproject.org/bigbrain-ftp/>. Keep data files in `data/`; they are not committed.

## Repository layout

```
configs/      dataset and cache settings
data/         local data (not committed)
docs/         design notes; docs/brand holds the logo and banner
pyramid/      out-of-core pyramid builder
scripts/      command-line utilities
server/       GPU renderer, brick cache, frame streaming
tests/        correctness and memory-bound checks
web/          browser page
```

## Credits and licence

The logo is an original drawing. The banner is a GPU render of brain 222. A licence for the code has not been
chosen yet.
