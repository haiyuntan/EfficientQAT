# R0–R5 experiment reports

Read logs/*/status.tsv first, then evaluation JSON, train_results.json and update_audit.json. SMOKE_OK marks a successful smoke gate; ALL_DONE marks a completed run. Failed and historical runs are retained. Model weights, datasets and caches stay on the experiment server. Console logs over 2 MiB contain only the last 2 MiB; manifest.json records truncation and omitted large text artifacts.
