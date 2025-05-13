import platform
import subprocess
import threading
import time
import re
import json


class GPUMonitor:
    def __init__(self, log_file="gpu_stats.json", sample_interval=1, update_interval=10):
        self.log_file = log_file
        self.sample_interval = sample_interval
        self.update_interval = update_interval
        self.samples = []
        self._stop_event = threading.Event()

        self._monitor_fn = self._monitor_jetson

    def start(self):
        if self._monitor_fn:
            self.thread = threading.Thread(target=self._monitor_fn, daemon=True)
            self.thread.start()

    def stop(self):
        self._stop_event.set()
        if hasattr(self, "thread"):
            self.thread.join()

    def _monitor_jetson(self):
        def get_tegrastats_sample():
            try:
                result = subprocess.check_output("tegrastats --interval 1000 --count 1", shell=True).decode()
                return result.strip()
            except subprocess.CalledProcessError:
                return ""

        def parse(output):
            mem_used = mem_total = gpu_util = total_power_mw = None

            ram_match = re.search(r'RAM (\d+)/(\d+)MB', output)
            if ram_match:
                mem_used = int(ram_match.group(1))
                mem_total = int(ram_match.group(2))

            gpu_match = re.search(r'GR3D_FREQ (\d+)%', output)
            if gpu_match:
                gpu_util = int(gpu_match.group(1))

            power_matches = re.findall(r'(\d+)mW/\d+mW', output)
            if power_matches:
                total_power_mw = sum(int(p) for p in power_matches)

            return {
                'gpu_util_percent': gpu_util or 0,
                'gpu_mem_used_MB': mem_used,
                'gpu_mem_total_MB': mem_total,
                'total_power_mw': total_power_mw or 0
            }

        last_update = time.time()

        while not self._stop_event.is_set():
            raw = get_tegrastats_sample()
            parsed = parse(raw)
            self.samples.append(parsed)

            if time.time() - last_update >= self.update_interval:
                avg = self._calculate_averages()
                with open(self.log_file, "w") as f:
                    json.dump(avg, f, indent=2)
                last_update = time.time()

            time.sleep(self.sample_interval)

    def _calculate_averages(self):
        if not self.samples:
            return {}

        keys = self.samples[0].keys()
        avg = {}
        for key in keys:
            values = [s[key] for s in self.samples if s[key] is not None]
            avg[key] = sum(values) / len(values) if values else None
        return avg
