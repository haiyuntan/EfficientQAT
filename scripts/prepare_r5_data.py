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
get_task_dict(['mmlu'], TaskManager())
print('MMLU data prepared', flush=True)
