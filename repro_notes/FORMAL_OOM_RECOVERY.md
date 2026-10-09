# Formal R0–R5 recovery, 2026-10-09

Run: `r0_r5_20261008_v7_full`; branch: `reproduce/llama2-7b-w2g64`.
Pre-fix snapshot: `dfb172f`.

Smoke artifact validation rerun passed: saved checkpoints, finite losses,
changed scales, unchanged frozen tensors, checkpoint evaluation, and 5-shot MMLU.
Smoke results are limited-data pipeline checks, not formal performance results.

Completed formal baselines:
- R0: WikiText2 PPL 5.47; five-task average 64.81%.
- R1: WikiText2 PPL 6.86; five-task average 60.15%.

R2 stopped after capturing FP activations, before block 0 training. No CUDA OOM
traceback or FAILED record was saved. The original driver disappeared. Host RAM
is 120 GiB with no swap. Kernel logs are inaccessible, so the historical OOM kill
cannot be directly confirmed. The allocation site and memory requirement strongly
indicate host-memory exhaustion: two FP16 activation banks each contain
(4096 + 64) * 2048 * 4096 * 2 bytes = 65 GiB, totaling 130 GiB, before model
weights and other buffers. The quantized bank allocation follows the final log line.

Fix: use the existing `--off_load_to_disk` implementation for formal Block-AP,
keeping all training hyperparameters unchanged. The filesystem has sufficient space.
Resume reuses R0/R1 only with DONE records and reruns both numeric acceptance gates.
The interrupted R2 console log is archived before restarting R2. Original Git commit
metadata is retained and recovery commits are recorded separately.

Resource telemetry records available host memory, GPU memory and cgroup OOM counts.
Analysis export excludes weights, datasets and caches. Network push status must be
reported separately from local experiment completion.
