"""Validate all R0-R5 local datasets before launching GPU experiments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_datasets import load_dataset
for path, name in (("wikitext", "wikitext-2-raw-v1"), ("allenai/c4", None),
                   ("togethercomputer/RedPajama-Data-1T-Sample", None),
                   ("tatsu-lab/alpaca", None), ("piqa", None),
                   ("allenai/ai2_arc", "ARC-Easy"), ("allenai/ai2_arc", "ARC-Challenge"),
                   ("hellaswag", None), ("winogrande", "winogrande_xl")):
    data = load_dataset(path, name)
    assert all(len(split) > 0 for split in data.values()), path
    print(path, name, {key: len(value) for key, value in data.items()}, flush=True)
from lm_eval.tasks import TaskManager, get_task_dict
import lm_eval
subjects = sorted(p.stem for p in (Path(lm_eval.__file__).parent / "tasks/mmlu/default").glob("mmlu_*.yaml"))
assert len(subjects) == 57
manager = TaskManager()
for index, subject in enumerate(subjects, 1):
    get_task_dict([subject], manager)
    print(f"MMLU local validated {index}/57: {subject}", flush=True)
get_task_dict(["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "mmlu"], manager)
print("All R0-R5 local datasets validated; no Hub access", flush=True)
