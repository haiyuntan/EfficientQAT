"""Download official Alpaca JSON; lm-eval prepares and caches all MMLU subjects."""
import json
import urllib.request
from pathlib import Path
root = Path(__file__).resolve().parents[2]
out = root / 'data/local_datasets/alpaca/alpaca_data.json'
out.parent.mkdir(parents=True, exist_ok=True)
if not out.exists():
    with urllib.request.urlopen('https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json', timeout=60) as response:
        rows = json.load(response)
    assert len(rows) > 50000 and all(set(('instruction','input','output')) <= row.keys() for row in rows)
    out.write_text(json.dumps(rows))
print('Alpaca:', len(json.loads(out.read_text())), out, flush=True)
from lm_eval.tasks import TaskManager, get_task_dict
import time
manager = TaskManager()
import lm_eval
subjects = sorted(p.stem for p in (Path(lm_eval.__file__).parent / 'tasks/mmlu/default').glob('mmlu_*.yaml'))
assert len(subjects) == 57, f"Expected 57 default MMLU subjects, got {len(subjects)}"
for index, subject in enumerate(subjects, 1):
    for attempt in range(1, 6):
        try:
            get_task_dict([subject], manager)
            print(f"MMLU prepared {index}/57: {subject}", flush=True)
            break
        except (ConnectionError, OSError, ValueError) as error:
            print(f"Retry {attempt}/5 {subject}: {error}", flush=True)
            if attempt == 5:
                raise
            time.sleep(2 * attempt)
get_task_dict(['mmlu'], manager)
print('MMLU data prepared', flush=True)
