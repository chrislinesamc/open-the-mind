"""Live machine readings for the page: a thread samples RAM, GPU and the server's own frame rate twice a
second and keeps the last minute. It never takes the render lock, so sampling cannot delay a frame.

On the GB10 the GPU and CPU share one memory, so NVML has no device memory total ("N/A"); what is
shown is the server process's GPU allocation (NVML per process), the whole machine's RAM (which the GPU
allocation is part of) and the server process's CPU-side resident set.
"""
import collections
import os
import threading
import time

import torch

try:
    import pynvml
except ImportError:                                          # readings degrade to the torch numbers
    pynvml = None

GB = 2**30


class Monitor:
    def __init__(self, period=0.5, keep=120):
        self.period, self.hist = period, collections.deque(maxlen=keep)
        self.frames, self.gpu_ms = collections.deque(maxlen=4096), 0.0
        self.peak = dict(ram_used_gb=0.0, gpu_proc_gb=0.0, rss_gb=0.0)
        self.lock = threading.Lock()
        self.h = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            except pynvml.NVMLError:
                self.h = None
        self.page = os.sysconf("SC_PAGE_SIZE")
        threading.Thread(target=self._run, daemon=True).start()

    def frame(self, gpu_ms):
        """Called after every rendered frame (from any viewer)."""
        with self.lock:
            self.frames.append(time.time())
            self.gpu_ms = gpu_ms

    def _gpu(self):
        if self.h is None:
            return None, None, None, torch.cuda.memory_reserved() / GB
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.h).gpu
            power = pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000
            temp = pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU)
            mine = [p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self.h)
                    if p.pid == os.getpid() and p.usedGpuMemory is not None]
            proc = mine[0] / GB if mine else torch.cuda.memory_reserved() / GB
            return util, power, temp, proc
        except pynvml.NVMLError:
            return None, None, None, torch.cuda.memory_reserved() / GB

    def _sample(self):
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                mem[k] = int(v.split()[0]) * 1024
        rss = int(open("/proc/self/statm").read().split()[1]) * self.page
        util, power, temp, proc = self._gpu()
        now = time.time()
        with self.lock:
            fps = sum(1 for t in self.frames if now - t < 1.0)
            gpu_ms = self.gpu_ms
        s = dict(t=round(now, 2), gpu_util=util, power_w=power and round(power, 1), temp_c=temp,
                 gpu_proc_gb=round(proc, 3), gpu_alloc_gb=round(torch.cuda.memory_allocated() / GB, 3),
                 ram_used_gb=round((mem["MemTotal"] - mem["MemAvailable"]) / GB, 3),
                 ram_total_gb=round(mem["MemTotal"] / GB, 3), rss_gb=round(rss / GB, 3),
                 server_fps=fps, gpu_ms=gpu_ms)
        for k in self.peak:
            self.peak[k] = max(self.peak[k], s[k])
        return s

    def _run(self):
        while True:
            s = self._sample()
            with self.lock:
                self.hist.append(s)
            time.sleep(self.period)

    def snapshot(self, since=0.0):
        """Samples newer than `since` (all of the last minute for 0) and the peaks since start."""
        with self.lock:
            return dict(samples=[s for s in self.hist if s["t"] > since], peak=dict(self.peak),
                        period=self.period)
