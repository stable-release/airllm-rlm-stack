import threading

import torch


class StreamStats:
    """Always-on counters for the weight-streaming pipeline.

    Tracks how many bytes moved disk -> CPU and CPU -> GPU and how long each took, so
    effective bandwidth can be reported per generation. Loads happen on both the main and
    the prefetch thread, hence the lock. Overhead is a few counter increments per layer,
    which is nothing next to a multi-hundred-MB shard read.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with getattr(self, '_lock', threading.Lock()):
            self.disk_bytes = 0
            self.disk_seconds = 0.0
            self.gpu_bytes = 0
            self.gpu_seconds = 0.0
            self.layers_loaded = 0

    def add_disk(self, nbytes, seconds):
        with self._lock:
            self.disk_bytes += nbytes
            self.disk_seconds += seconds
            self.layers_loaded += 1

    def add_gpu(self, nbytes, seconds):
        with self._lock:
            self.gpu_bytes += nbytes
            self.gpu_seconds += seconds

    def snapshot(self):
        with self._lock:
            return {
                'layers_loaded': self.layers_loaded,
                'disk_bytes': self.disk_bytes,
                'disk_seconds': round(self.disk_seconds, 3),
                'disk_gb_per_s': round(self.disk_bytes / self.disk_seconds / 1e9, 3) if self.disk_seconds else 0.0,
                'gpu_bytes': self.gpu_bytes,
                'gpu_seconds': round(self.gpu_seconds, 3),
                'gpu_gb_per_s': round(self.gpu_bytes / self.gpu_seconds / 1e9, 3) if self.gpu_seconds else 0.0,
            }

    def delta_since(self, before):
        after = self.snapshot()
        out = {k: after[k] - before[k] for k in
               ('layers_loaded', 'disk_bytes', 'disk_seconds', 'gpu_bytes', 'gpu_seconds')}
        out['disk_gb_per_s'] = round(out['disk_bytes'] / out['disk_seconds'] / 1e9, 3) if out['disk_seconds'] else 0.0
        out['gpu_gb_per_s'] = round(out['gpu_bytes'] / out['gpu_seconds'] / 1e9, 3) if out['gpu_seconds'] else 0.0
        out['disk_seconds'] = round(out['disk_seconds'], 3)
        out['gpu_seconds'] = round(out['gpu_seconds'], 3)
        return out


class LayeredProfiler:
    def __init__(self, print_memory=False):
        self.profiling_time_dict = {}
        self.print_memory = print_memory
        self.min_free_mem = 1024*1024*1024*1024


    def add_profiling_time(self, item, time):

        if not item in self.profiling_time_dict:
            self.profiling_time_dict[item] = []

        self.profiling_time_dict[item].append(time)

        if self.print_memory:
            free_mem = torch.cuda.mem_get_info()[0]
            self.min_free_mem = min(self.min_free_mem, free_mem)
            print(f"free vmem @{item}: {free_mem/1024/1024/1024:.02f}GB, min free: {self.min_free_mem/1024/1024/1024:.02f}GB")

    def clear_profiling_time(self):
        for item in self.profiling_time_dict.keys():
            self.profiling_time_dict[item] = []

    def print_profiling_time(self):
        for item in self.profiling_time_dict.keys():
            print(f"total time for {item}: {sum(self.profiling_time_dict[item])}")

