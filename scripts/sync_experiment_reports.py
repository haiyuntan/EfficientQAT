"""Publish small analysis artifacts; never copy weights, datasets or caches."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
DEST = REPO / "experiment_reports"
ALLOWED = {".json", ".log", ".tsv", ".txt", ".md", ".yaml", ".yml"}
MARKERS = {"SMOKE_OK", "ALL_DONE"}
MAX_BYTES = 2 * 1024 * 1024


def git(*args):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    command = ["git"]
    if env.get("REPORT_GITHUB_TOKEN"):
        # Supply credentials from process memory; never write them into Git config.
        helper = '!f() { if [ "$1" = get ]; then printf "username=%s\\npassword=%s\\n" haiyuntan "$REPORT_GITHUB_TOKEN"; fi; }; f'
        command += ["-c", "credential.helper=", "-c", "credential.helper=" + helper]
    return subprocess.run([*command, *args], cwd=REPO, env=env, check=True, timeout=55)


def snapshot(local_only=False):
    manifest = []
    for category in ("logs", "outputs"):
        source = PROJECT / category / "r0_r5"
        if not source.exists():
            continue
        for file in sorted(source.rglob("*")):
            if not file.is_file() or file.is_symlink():
                continue
            relative = file.relative_to(source)
            if any(part.startswith("checkpoint-") or part in {"smoke_data", "blockap_cache"} for part in relative.parts):
                continue
            if file.suffix not in ALLOWED and file.name not in MARKERS:
                continue
            # All large JSON/text artifacts stay local; console logs retain their tail.
            size = file.stat().st_size
            truncated = size > MAX_BYTES
            if truncated and file.suffix != ".log":
                manifest.append({"path": f"{category}/{relative}", "bytes": size, "copied": False})
                continue
            with file.open("rb") as stream:
                if truncated:
                    stream.seek(-MAX_BYTES, 2)
                content = stream.read().decode("utf-8", errors="replace")
            content = re.sub(r"gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|hf_[A-Za-z0-9]{20,}", "[REDACTED]", content)
            target = DEST / category / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            manifest.append({"path": f"{category}/{relative}", "bytes": size, "copied": True, "tail_only": truncated})
    DEST.mkdir(exist_ok=True)
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (DEST / "README.md").write_text("# R0–R5 experiment reports\n\nRead logs/*/status.tsv first, then evaluation JSON, train_results.json and update_audit.json. SMOKE_OK marks a successful smoke gate; ALL_DONE marks a completed run. Failed and historical runs are retained. Model weights, datasets and caches stay on the experiment server. Console logs over 2 MiB contain only the last 2 MiB; manifest.json records truncation and omitted large text artifacts.\n")
    git("add", "experiment_reports")
    changes = subprocess.run(["git", "diff", "--cached", "--quiet", "--", "experiment_reports"], cwd=REPO)
    if changes.returncode == 1:
        git("-c", "user.name=haiyuntan", "-c", "user.email=haiyuntan@users.noreply.github.com", "commit", "-m", "Update R0-R5 experiment analysis artifacts", "--", "experiment_reports")
    elif changes.returncode != 0:
        raise RuntimeError("Cannot inspect staged reports")
    if local_only:
        print("Reports committed locally; push deferred", flush=True)
        return
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    if not branch:
        raise RuntimeError("Cannot synchronize a detached HEAD")
    git("-c", "http.proxy=" + os.environ.get("REPORT_GIT_PROXY", "http://99.72.0.200:3138"), "push", "origin", f"HEAD:refs/heads/{branch}")
    print("Reports synchronized to origin/" + branch, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--after-run", help="Wait for this formal run's ALL_DONE marker, then push once")
    parser.add_argument("--local-only", action="store_true", help="Commit reports without a network push")
    args = parser.parse_args()
    if args.after_run:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", args.after_run) or args.after_run in {".", ".."}:
            parser.error("Invalid run ID")
        marker = PROJECT / "logs/r0_r5" / args.after_run / "ALL_DONE"
        print("Waiting for formal R0-R5 completion: " + str(marker), flush=True)
        while not marker.is_file():
            time.sleep(60)
    with (REPO / ".git/report-sync.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            snapshot(args.local_only)
        except Exception as error:
            print(type(error).__name__ + ": " + str(error), flush=True)
            raise SystemExit(1)
