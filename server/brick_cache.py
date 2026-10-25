"""Bounded brick memory: a fixed GPU brick pool with a page table and LRU eviction, fed by a
fixed-size host cache of compressed bricks. Neither grows with the data.

Each frame the kernel flags the bricks it wanted but did not find (req). collect() queues them in a
priority queue: newest frame first, coarse levels first within a frame, so after a fast zoom the
current view loads before stale ones (requests older than MAX_AGE frames are dropped; they come back
if still needed). Loader threads read the compressed brick (host cache or disk) and decode it
straight into a pinned staging slot. apply_uploads() runs between frames: it copies
ready bricks into free or least-recently-used pool slots and updates the page table, so a frame
never sees a half-written slot. The coarsest levels are loaded at start and pinned, so every ray
always finds some resident level.
"""
import heapq
import os
import sys
import threading
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pyramid"))
from bricks import BRICK  # noqa: E402
from omezarr import open_pyramid  # noqa: E402

BVOX = BRICK[0] * BRICK[1] * BRICK[2]
MISSING, EMPTY = -1, -2
PINNED = 2**31 - 1
MAX_AGE = 120                       # frames; older queued requests are dropped


class BrickCache:
    def __init__(self, root, pool_gb=4.0, host_gb=8.0, stage_bricks=1024, threads=64, finest=0, pin_share=0.25,
                 device="cuda", raw=None):
        self.pyr = open_pyramid(root, raw=raw)
        self.levels = self.pyr.meta["levels"]
        self.NL = len(self.levels)
        self.finest = min(max(int(finest), 0), self.NL - 1)      # rays never use (or load) a finer level
        self.threads = threads
        dev = torch.device(device)
        sizes = [int(np.prod(L["grid"])) for L in self.levels]
        self.off = np.cumsum([0] + sizes)
        stored = np.concatenate([self.pyr.idx[n]["n"].reshape(-1) > 0 for n in range(self.NL)])
        self.page = torch.where(torch.from_numpy(stored), MISSING, EMPTY).to(torch.int32).to(dev)
        self.bmax = torch.from_numpy(np.concatenate([self.pyr.vmax[n].reshape(-1)
                                                     for n in range(self.NL)])).to(dev)
        self.req = torch.zeros(len(stored), dtype=torch.uint8, device=dev)
        self.lvf = torch.tensor([L["spacing"] + [0.0] for L in self.levels], dtype=torch.float32, device=dev)
        self.lvi = torch.tensor([L["grid"] + [int(self.off[n])] for n, L in enumerate(self.levels)],
                                dtype=torch.int32, device=dev)

        self.S = int(pool_gb * 2**30) // BVOX                     # pool slots, fixed
        self.atlas = torch.zeros(self.S * BVOX, dtype=torch.uint8, device=dev)
        self.used = torch.zeros(self.S, dtype=torch.int32, device=dev)
        self.submax = torch.zeros(self.S * 128, dtype=torch.uint8, device=dev)   # max of 8x16x16 sub-blocks
        self.owner = torch.full((self.S,), -1, dtype=torch.int64, device=dev)
        self.next_free = 0
        self.frame = 1

        self.host = OrderedDict()                                 # page index -> compressed bytes
        self.host_bytes, self.host_cap = 0, int(host_gb * 2**30)
        self.stage = torch.empty((stage_bricks, BVOX), dtype=torch.uint8).pin_memory()
        self.stage_np = self.stage.numpy()
        self.free_stage = list(range(stage_bricks))
        self.pending_release = []                                 # (cuda event, staging slots)
        self.ready = []                                           # (page index, staging slot)
        self.inflight = set()                                     # loading, or decoded and waiting for upload
        self.queue, self.queued = [], {}                          # heap of (-frame, -level, page index); newest frame
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.stats = dict(loaded=0, disk_bytes=0, host_hits=0, evicted=0, dropped=0)
        self.pin_coarse(pin_share)
        for _ in range(threads):
            threading.Thread(target=self._worker, daemon=True).start()

    # -- page index <-> (level, gz, gy, gx)
    def where(self, pidx):
        n = int(np.searchsorted(self.off, pidx, side="right") - 1)
        return (n, *np.unravel_index(pidx - self.off[n], self.levels[n]["grid"]))

    def blob(self, pidx):
        with self.lock:
            b = self.host.get(pidx)
            if b is not None:
                self.host.move_to_end(pidx)
                self.stats["host_hits"] += 1
                return b
        b = self.pyr.blob(*self.where(pidx))
        with self.lock:
            self.stats["disk_bytes"] += len(b)
            self.host[pidx] = b
            self.host_bytes += len(b)
            while self.host_bytes > self.host_cap:
                _, old = self.host.popitem(last=False)
                self.host_bytes -= len(old)
        return b

    def pin_coarse(self, share=0.25):
        """Load and pin the coarsest levels whose stored bricks fit in a share of the pool (never finer than `finest`)."""
        budget, n = int(self.S * share), self.NL
        while n > self.finest and sum(int((self.pyr.idx[m]["n"] > 0).sum()) for m in range(n - 1, self.NL)) <= budget:
            n -= 1
        self.pinned_from = n
        pidx = np.nonzero((self.page[int(self.off[n]):].cpu().numpy() == MISSING))[0] + int(self.off[n])
        for i in range(0, len(pidx), len(self.free_stage)):
            chunk = pidx[i:i + len(self.free_stage)]
            for j, p in enumerate(chunk):
                self.pyr.decode_into(self.where(int(p))[0], self.blob(int(p)), self.stage_np[j])
            slots = torch.arange(self.next_free, self.next_free + len(chunk), device=self.atlas.device)
            self.next_free += len(chunk)
            self.atlas.view(self.S, BVOX)[slots] = self.stage[:len(chunk)].to(self.atlas.device)
            self.update_submax(slots)
            p = torch.from_numpy(chunk).to(self.atlas.device)
            self.page[p] = slots.to(torch.int32)
            self.owner[slots] = p
            self.used[slots] = PINNED                          # informational; eviction skips [0, n_pinned)
        self.n_pinned = self.next_free

    # -- after a frame: queue the missed bricks
    def collect(self):
        idx = torch.nonzero(self.req).flatten()
        if len(idx) == 0:
            return 0
        self.req.zero_()
        idx = idx.cpu().numpy()
        lvl = np.searchsorted(self.off, idx, side="right") - 1
        with self.cv:
            for p, n in zip(idx.tolist(), lvl.tolist()):
                if p not in self.inflight and self.queued.get(p) != self.frame:
                    self.queued[p] = self.frame
                    heapq.heappush(self.queue, (-self.frame, -n, p))
            if len(self.queue) > 2 * len(self.queued) + 4096:   # superseded and stale entries pile up: rebuild
                self.queued = {p: f for p, f in self.queued.items() if self.frame - f <= MAX_AGE}
                lv = np.searchsorted(self.off, list(self.queued), side="right") - 1 if self.queued else []
                self.queue = [(-f, -int(n), p) for (p, f), n in zip(self.queued.items(), lv)]
                heapq.heapify(self.queue)
            self.cv.notify_all()
        return len(idx)

    def _worker(self):
        while True:
            with self.cv:
                while True:
                    while not self.queue:
                        self.cv.wait()
                    negf, _, p = heapq.heappop(self.queue)
                    if self.queued.get(p) != -negf:          # superseded by a newer request for p
                        continue
                    del self.queued[p]
                    if self.frame + negf > MAX_AGE:
                        self.stats["dropped"] += 1
                        continue
                    self.inflight.add(p)
                    break
            self._load(p)

    def _load(self, pidx):
        k = None
        try:
            b = self.blob(pidx)
            with self.cv:                            # block, not poll: 64 polling threads starve the render thread of the GIL
                while not self.free_stage:
                    self.cv.wait()
                k = self.free_stage.pop()
            self.pyr.decode_into(self.where(pidx)[0], b, self.stage_np[k])
            with self.lock:
                self.ready.append((pidx, k))
        except Exception as e:                       # never lose an inflight entry or a staging slot
            print("brick load failed", pidx, e, flush=True)
            with self.lock:
                self.inflight.discard(pidx)
                if k is not None:
                    self.free_stage.append(k)
                    self.cv.notify_all()

    # -- before a frame: move ready bricks into the pool
    def apply_uploads(self, max_n=256):
        with self.lock:
            done = [ev_k for ev_k in self.pending_release if ev_k[0].query()]
            for ev, ks in done:
                self.free_stage.extend(ks)
            if done:
                self.cv.notify_all()
            self.pending_release = [x for x in self.pending_release if x not in done]
            batch, self.ready = self.ready[:max_n], self.ready[max_n:]
        if not batch:
            return 0
        dev = self.atlas.device
        n_new = min(len(batch), self.S - self.next_free)
        slots = list(range(self.next_free, self.next_free + n_new))
        self.next_free += n_new
        self.used[self.next_free - n_new:self.next_free] = self.frame    # taken: not eviction candidates
        if len(batch) > n_new:                       # evict least recently used, never this or last frame
            need = len(batch) - n_new
            # pinned coarse bricks occupy slots [0, n_pinned) and are never candidates (the kernel's LRU
            # stamp overwrites their PINNED mark, so exclude them by position)
            cand = torch.topk(self.used[self.n_pinned:], min(need, self.S - self.n_pinned), largest=False)
            ok = cand.indices[cand.values < self.frame - 1] + self.n_pinned
            slots += ok.tolist()
            if len(ok) < need:                       # pool full of bricks in use: retry next frame
                with self.lock:
                    self.ready = batch[len(slots):] + self.ready
                batch = batch[:len(slots)]
            old = self.owner[ok]
            self.page[old[old >= 0]] = MISSING
            self.stats["evicted"] += len(ok)
        if not batch:
            return 0
        st = torch.tensor(slots, device=dev)
        pidx = torch.tensor([p for p, _ in batch], device=dev)
        ks = [k for _, k in batch]
        av = self.atlas.view(self.S, BVOX)
        for s, k in zip(slots, ks):
            av[s].copy_(self.stage[k], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.update_submax(st)
        self.page[pidx] = st.to(torch.int32)
        self.owner[st] = pidx
        self.used[st] = self.frame
        with self.lock:
            self.pending_release.append((ev, ks))
            for p, _ in batch:
                self.inflight.discard(p)
        self.stats["loaded"] += len(batch)
        return len(batch)

    def update_submax(self, slots):
        """Second level of empty-space skipping: max of each 8 x 16 x 16 sub-block of the given slots."""
        b = self.atlas.view(self.S, 2, 8, 8, 16, 8, 16)[slots]
        self.submax.view(self.S, 128)[slots] = b.amax(dim=(2, 4, 6)).reshape(len(slots), 128)

    def loading(self):
        """Bricks queued or being read/decoded (not counting decoded ones waiting for a pool slot)."""
        with self.lock:
            return bool(self.queued) or len(self.inflight) > len(self.ready)

    def busy(self):
        with self.lock:
            return bool(self.inflight or self.ready or self.queued)

    def memory(self):
        return dict(pool_gb=self.atlas.numel() / 2**30, pool_slots=self.S, resident=self.next_free,
                    pinned=self.n_pinned, host_cache_gb=self.host_bytes / 2**30, host_cap_gb=self.host_cap / 2**30,
                    staging_gb=self.stage.numel() / 2**30, inflight=len(self.inflight), queued=len(self.queued),
                    ready=len(self.ready), finest_level=self.finest, pinned_from_level=self.pinned_from,
                    loader_threads=self.threads, **self.stats)
