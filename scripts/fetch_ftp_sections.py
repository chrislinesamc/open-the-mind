"""Download a directory of section images (e.g. BigBrain 20 um coronal PNGs from the BigBrain FTP's web index)
with parallel connections. Resumable: files already complete are skipped; each file is written to .part and
renamed only when its size matches the server's Content-Length.

usage: python scripts/fetch_ftp_sections.py [out dir] [--workers 16] [--url <index url>] [--ext .png]
default url: BigBrain Release 2015, 2D_Final_Sections/Coronal/Png/Full_Resolution (7404 sections, ~60 GB)
"""
import argparse
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/2D_Final_Sections/Coronal/Png/Full_Resolution/"
UA = {"User-Agent": "open-the-mind/0.1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="data/bigbrain20_png")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--ext", default=".png")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index = urllib.request.urlopen(urllib.request.Request(a.url, headers=UA), timeout=180).read().decode()
    names = sorted(set(re.findall(r'href="([^"/?]+' + re.escape(a.ext) + ')"', index)))
    print(f"{len(names)} files listed at {a.url}", flush=True)

    lock, done, nbytes, fails, skipped = threading.Lock(), [0], [0], [], [0]
    t0 = time.time()

    def one(name):
        path = os.path.join(a.out, name)
        if os.path.exists(path):                  # final names are only given after the size check passed
            with lock:
                skipped[0] += 1
            return
        for attempt in range(6):
            try:
                with urllib.request.urlopen(urllib.request.Request(a.url + name, headers=UA), timeout=300) as r:
                    size = int(r.headers["Content-Length"])
                    if os.path.exists(path) and os.path.getsize(path) == size:
                        with lock:
                            skipped[0] += 1
                        return
                    tmp = path + ".part"
                    got = 0
                    with open(tmp, "wb") as f:
                        while chunk := r.read(1 << 20):
                            f.write(chunk)
                            got += len(chunk)
                            with lock:
                                nbytes[0] += len(chunk)
                if got != size:
                    raise IOError(f"{name}: {got} of {size} bytes")
                os.rename(tmp, path)
                with lock:
                    done[0] += 1
                return
            except Exception as e:
                if attempt == 5:
                    with lock:
                        fails.append((name, str(e)))
                    return
                time.sleep(2 ** attempt)

    def report():
        while True:
            time.sleep(30)
            with lock:
                n, b, s = done[0], nbytes[0], skipped[0]
            dt = time.time() - t0
            rate = b / dt
            left = len(names) - n - s
            per_file = b / max(n, 1)
            print(f"{n + s:,}/{len(names):,} files ({s} already here) in {dt / 60:.1f} min: {rate / 1e6:.2f} MB/s, "
                  f"{b / 1e9:.2f} GB, {len(fails)} failed, ETA {left * per_file / max(rate, 1) / 3600:.1f} h", flush=True)
    threading.Thread(target=report, daemon=True).start()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, names))
    print(f"done: {done[0]:,} downloaded, {skipped[0]:,} already here, {nbytes[0] / 1e9:.2f} GB in "
          f"{(time.time() - t0) / 3600:.2f} h, {len(fails)} failed (run again to retry): {fails[:5]}", flush=True)


if __name__ == "__main__":
    main()
