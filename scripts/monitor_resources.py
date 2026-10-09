"""Sample host memory, cgroup OOM counters and GPU use until the driver exits."""
import csv
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

pid, destination = int(sys.argv[1]), Path(sys.argv[2])
with destination.open('w', buffering=1) as output:
    writer = csv.writer(output, delimiter='\t')
    writer.writerow(['utc', 'host_available_kib', 'gpu_memory_used_mib', 'gpu_util_percent', 'cgroup_memory_bytes', 'oom_events'])
    while Path(f'/proc/{pid}').exists():
        status = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        if status == 'Z':
            break
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
        values = gpu.stdout.splitlines()[int(os.environ.get('GPU_ID', '0'))].split(',') if gpu.returncode == 0 else ['NA', 'NA']
        cgroup = Path('/sys/fs/cgroup')
        current = (cgroup / 'memory.current').read_text().strip() if (cgroup / 'memory.current').exists() else 'NA'
        events = (cgroup / 'memory.events').read_text().replace('\n', ';') if (cgroup / 'memory.events').exists() else 'NA'
        writer.writerow([datetime.datetime.now(datetime.timezone.utc).isoformat(), memory['MemAvailable'].split()[0], *[v.strip() for v in values], current, events])
        time.sleep(30)
