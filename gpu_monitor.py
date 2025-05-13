import subprocess
import os
import signal
import re

class GPUMonitor:
    def __init__(self, logfile='test.json'):
        self.logfile = logfile
        self.process = None

    def start(self):
        if self.process is not None:
            raise RuntimeError("tegrastats is already running.")

        self.process = subprocess.Popen(
            ['tegrastats', '--logfile', self.logfile],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setpgrp  # Detach process group
        )
        print(f"Started tegrastats with PID {self.process.pid}")

    def stop(self):
        if self.process is None:
            raise RuntimeError("tegrastats is not running.")

        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait()
            print("tegrastats stopped.")
        except ProcessLookupError:
            print("tegrastats process already terminated.")
        finally:
            self.process = None

        stats = self._parse_tegrastats_log(self.logfile)
        print("Averaged Stats:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        return stats

    def _parse_tegrastats_log(self, filepath):
        gpu_usages = []
        gpu_temps = []
        cpu_temps = []
        ram_usages = []
        power_usages = []

        is_agx = False

        with open(filepath, 'r') as f:
            for line in f:
                if re.search(r'VIN_|VDD_GPU_SOC|VDD_CPU_CV', line):
                    is_agx = True

                # GPU usage
                match_gpu = re.search(r'GR3D_FREQ\s+(\d+)%', line)
                if match_gpu:
                    gpu_usages.append(int(match_gpu.group(1)))

                # GPU temp
                match_gpu_temp = re.search(r'GPU@([\d\.]+)C', line)
                if match_gpu_temp:
                    gpu_temps.append(float(match_gpu_temp.group(1)))

                # CPU temp
                match_cpu_temp = re.search(r'CPU@([\d\.]+)C', line)
                if match_cpu_temp:
                    cpu_temps.append(float(match_cpu_temp.group(1)))

                # RAM usage
                match_ram = re.search(r'RAM\s+(\d+)/\d+MB', line)
                if match_ram:
                    ram_usages.append(int(match_ram.group(1)))

                if is_agx:
                    matches = re.findall(r'(VDD|VIN)[\w_]*\s+(\d+)mW', line)
                    total_power = sum(int(val) for _, val in matches)
                    if total_power > 0:
                        power_usages.append(total_power)
                else:
                    match_vddin = re.search(r'VDD_IN\s+(\d+)mW', line)
                    if match_vddin:
                        power_usages.append(int(match_vddin.group(1)))

        def avg(lst):
            lst=lst[5:]
            return sum(lst) / len(lst) if lst else 0

        return {
            'avg_gpu_usage_percent': round(avg(gpu_usages), 2),
            'avg_gpu_temperature_C': round(avg(gpu_temps), 2),
            'avg_cpu_temperature_C': round(avg(cpu_temps), 2),
            'avg_total_power_W': round(avg(power_usages)/1000, 2)
        }
