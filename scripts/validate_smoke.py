"""Fail the smoke gate when trained checkpoints, update audits, or MMLU are incomplete."""
import json
import math
import sys
from pathlib import Path
root, logs = map(Path, sys.argv[1:])
for run in ('R2', 'R4', 'R5'):
    folder = root / f'{run}_smoke' / 'e2e_qp'
    for name in ('config.json', 'tokenizer_config.json', 'trainer_state.json', 'model.safetensors', 'update_audit.json'):
        assert (folder / name).is_file(), (run, name)
    audit = json.loads((folder / 'update_audit.json').read_text())
    assert audit['frozen_unchanged'] and audit['changed_scales'] > 0
    metrics = json.loads((folder / 'train_results.json').read_text())
    assert math.isfinite(metrics['train_loss'])
import re
for run in ('R2', 'R3', 'R4', 'R5'):
    text = (logs / f'{run}_saved_eval.log').read_text()
    match = re.search(r'wikitext2 perplexity:\s*([0-9.]+)', text)
    assert match and math.isfinite(float(match[1])) and float(match[1]) > 0
    results = json.loads((logs / f'{run}_saved_eval' / 'eval_results.json').read_text())
    for task in ('piqa', 'arc_easy', 'arc_challenge', 'hellaswag', 'winogrande'):
        assert math.isfinite(results['results'][task]['acc,none'])
for run in ('R0', 'R3', 'R4', 'R5'):
    results = json.loads((logs / f'{run}_MMLU' / 'eval_results.json').read_text())
    assert math.isfinite(results['results']['mmlu']['acc,none'])
print('Validated saved models, finite losses, actual scale updates, frozen tensors, and reloaded 5-shot MMLU.')
