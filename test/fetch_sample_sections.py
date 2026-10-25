"""Download a short run of BigBrain 20 um coronal PNG sections at full resolution (default: 64 from the middle
of the brain, about 0.9 GB at 13 MB each). Resumable: files already complete are skipped; each file is
written to .part and renamed only when its size matches the server's Content-Length.

usage: python test/fetch_sample_sections.py [out dir] [--first 3700] [--count 64] [--workers 8]
then:  python scripts/png_to_zarr.py data/bigbrain_slab data/bigbrain_slab.ome.zarr --section-um 20 --pixel-um 21.17
"""
import argparse
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/2D_Final_Sections/Coronal/Png/Full_Resolution/"
PATTERN = "pm{:04d}o.png"
LAST = 7404                                     # sections are pm0001 .. pm7404
UA = {"User-Agent": "open-the-mind/0.1"}


def fetch(url, path):
    """One file, retried; returns the bytes downloaded (0 if it was already there)."""
    if os.path.exists(path):                    # final names are only given after the size check passed
        return 0
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                size = int(r.headers["Content-Length"])
                tmp, got = path + ".part", 0
                with open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        got += len(chunk)
            if got != size:
                raise IOError(f"{os.path.basename(path)}: {got} of {size} bytes")
            os.rename(tmp, path)
            return got
        except Exception as e:
            if attempt == 5:
                raise
            print(f"  retry {os.path.basename(path)}: {e}", flush=True)
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="data/bigbrain_slab")
    ap.add_argument("--first", type=int, default=3700, help="first section number (1..7404); default 3700")
    ap.add_argument("--count", type=int, default=64, help="number of sections; default 64 (about 0.9 GB)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--url", default=URL)
    a = ap.parse_args()
    if not 1 <= a.first <= a.first + a.count - 1 <= LAST:
        ap.error(f"sections must lie in 1..{LAST}")
    os.makedirs(a.out, exist_ok=True)
    names = [PATTERN.format(k) for k in range(a.first, a.first + a.count)]
    print(f"{len(names)} sections, {names[0]} .. {names[-1]} -> {a.out}", flush=True)

    t0, total = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, n in enumerate(ex.map(lambda nm: fetch(a.url + nm, os.path.join(a.out, nm)), names), 1):
            total += n
            if i % 8 == 0 or i == len(names):
                print(f"  {i}/{len(names)}  {total / 2**20:.0f} MB  {time.time() - t0:.0f} s", flush=True)
    print(f"next: python scripts/png_to_zarr.py {a.out} {a.out}.ome.zarr --section-um 20 --pixel-um 21.17")


if __name__ == "__main__":
    main()
