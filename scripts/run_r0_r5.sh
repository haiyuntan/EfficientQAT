#!/usr/bin/env bash
# Sequential EfficientQAT R0-R5 runner. Default mode is the inexpensive smoke check.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$REPO_ROOT/.." && pwd)"
MODE="${1:-smoke}"
RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_ID="${GPU_ID:-0}"
RESUME_FROM="${RESUME_FROM:-}"
if [[ -n "$RESUME_FROM" && ( "$MODE" != "smoke" || "$RESUME_FROM" != "R2_E2E_QP" ) ]]; then
  echo "Only smoke RESUME_FROM=R2_E2E_QP is supported" >&2
  exit 2
fi
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Llama-2-7b-hf}"
OFFICIAL_DIR="${OFFICIAL_DIR:-$PROJECT_ROOT/outputs/official_w2g64}"
RUN_ROOT="$PROJECT_ROOT/outputs/r0_r5/$RUN_ID"
LOG_ROOT="$PROJECT_ROOT/logs/r0_r5/$RUN_ID"
PYTHON="${PYTHON:-python}"
TASKS="piqa,arc_easy,arc_challenge,hellaswag,winogrande"

usage() {
  echo "Usage: $0 [smoke|full|launch] [RUN_ID]"
  echo "  smoke: limited R0/R1 eval plus tiny Block-AP/E2E-QP runs for R2-R4 (default)"
  echo "  full:  run R0, R1, R2, R3, R4 sequentially in the foreground"
  echo "  launch: start full mode detached; returns the driver PID"
}

case "$MODE" in
  smoke|full|launch|auto) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ || "$RUN_ID" == "." || "$RUN_ID" == ".." ]]; then
  echo "RUN_ID may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$RUN_ROOT"
if [[ "$MODE" == "auto" ]]; then
  nohup bash "$SCRIPT_DIR/smoke_then_full.sh" "$RUN_ID" >"$LOG_ROOT/auto.log" 2>&1 </dev/null &
  echo "$!" >"$LOG_ROOT/auto.pid"
  echo "Started smoke then full: PID=$! logs=$LOG_ROOT"
  exit 0
fi
if [[ -z "$RESUME_FROM" && ( -s "$LOG_ROOT/status.tsv" || -e "$LOG_ROOT/ALL_DONE" ) ]]; then
  echo "Run already exists; choose a new RUN_ID" >&2
  exit 1
fi

if [[ "$MODE" == "launch" ]]; then
  DRIVER_LOG="$LOG_ROOT/driver.log"
  if [[ -e "$LOG_ROOT/driver.pid" ]] && kill -0 "$(<"$LOG_ROOT/driver.pid")" 2>/dev/null; then
    echo "A driver for RUN_ID=$RUN_ID is already active (PID $(<"$LOG_ROOT/driver.pid"))." >&2
    exit 1
  fi
  if [[ -z "$RESUME_FROM" && ( -s "$LOG_ROOT/status.tsv" || -e "$LOG_ROOT/ALL_DONE" ) ]]; then
    echo "RUN_ID=$RUN_ID already has a recorded run; choose a new RUN_ID to avoid overwriting results." >&2
    exit 1
  fi
  nohup bash "$SCRIPT_DIR/run_r0_r5.sh" full "$RUN_ID" >"$DRIVER_LOG" 2>&1 </dev/null &
  DRIVER_PID=$!
  printf '%s\n' "$DRIVER_PID" >"$LOG_ROOT/driver.pid"
  echo "Started full R0-R5 run: RUN_ID=$RUN_ID PID=$DRIVER_PID"
  echo "Driver log: $DRIVER_LOG"
  exit 0
fi

exec 9>"$PROJECT_ROOT/outputs/r0_r5/gpu-$GPU_ID.lock"
flock -n 9 || { echo "Another pipeline owns GPU $GPU_ID" >&2; exit 1; }


source "$PROJECT_ROOT/env/activate.sh"
cd "$REPO_ROOT"

if [[ ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/model-00001-of-00002.safetensors" || ! -f "$MODEL_DIR/model-00002-of-00002.safetensors" ]]; then
  echo "Base model is missing or incomplete: $MODEL_DIR" >&2
  exit 1
fi
if [[ ! -f "$OFFICIAL_DIR/config.json" || ! -f "$OFFICIAL_DIR/model.safetensors" ]]; then
  echo "Official W2g64 checkpoint is missing or incomplete: $OFFICIAL_DIR" >&2
  exit 1
fi
if ! nvidia-smi -L | grep -q "GPU $GPU_ID:"; then
  echo "GPU $GPU_ID is not visible to nvidia-smi." >&2
  exit 1
fi

export HF_HOME="$PROJECT_ROOT/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export LOCAL_DATASETS_DIR="$PROJECT_ROOT/data/local_datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"

printf 'run_id\t%s\nmode\t%s\nmodel_dir\t%s\nofficial_dir\t%s\ngpu_id\t%s\n' \
  "$RUN_ID" "$MODE" "$MODEL_DIR" "$OFFICIAL_DIR" "$GPU_ID" >"$LOG_ROOT/run_metadata.tsv"
nvidia-smi >"$LOG_ROOT/nvidia-smi.txt"
git -C "$REPO_ROOT" rev-parse HEAD >"$LOG_ROOT/efficientqat-commit.txt"

CURRENT_STEP="preflight"
on_exit() {
  rc=$?
  if (( rc != 0 )); then
    printf '%s\tFAILED\t%s\n' "$CURRENT_STEP" "$(date -u +%FT%TZ)" >>"$LOG_ROOT/status.tsv"
    echo "Stopped at $CURRENT_STEP (exit $rc). See $LOG_ROOT." >&2
  fi
}
trap on_exit EXIT

run_logged() {
  local step="$1"
  local logfile="$2"
  shift 2
  CURRENT_STEP="$step"
  printf '%s\tRUNNING\t%s\n' "$step" "$(date -u +%FT%TZ)" >>"$LOG_ROOT/status.tsv"
  echo "===== $step started $(date -u +%FT%TZ) ====="
  set +e
  "$@" 2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  set -e
  if (( rc != 0 )); then
    return "$rc"
  fi
  printf '%s\tDONE\t%s\n' "$step" "$(date -u +%FT%TZ)" >>"$LOG_ROOT/status.tsv"
  echo "===== $step completed $(date -u +%FT%TZ) ====="
}

source "$SCRIPT_DIR/r5_helpers.sh"

if [[ "$MODE" == "smoke" ]]; then
  SMOKE_NET="Llama-2-smoke-$RUN_ID"
  SMOKE_FAMILY="$SMOKE_NET"
  SMOKE_SEQUENCE_LENGTH=128
  SMOKE_TRAIN_SIZE=2
  SMOKE_VAL_SIZE=2
  SMOKE_BLOCK_CACHE="$RUN_ROOT/smoke_data/blockap_cache"
  SMOKE_E2E_CACHE="$REPO_ROOT/cache/e2e_dataloader_${SMOKE_FAMILY}_redpajama_${SMOKE_SEQUENCE_LENGTH}.cache"

  if [[ -z "$RESUME_FROM" ]]; then
  run_logged R0_FP16_limited_evaluation "$LOG_ROOT/R0_FP16_smoke.log" \
    "$PYTHON" main_block_ap.py --model "$MODEL_DIR" --net Llama-2 \
    --wbits 16 --group_size 64 --output_dir "$LOG_ROOT/R0_FP16_smoke" \
    --eval_ppl --ppl_datasets wikitext2 --ppl_max_samples 1 --ppl_seqlen "$SMOKE_SEQUENCE_LENGTH" \
    --eval_tasks "$TASKS" --eval_limit 1 --eval_batch_size 1
  run_logged R1_official_W2g64_limited_evaluation "$LOG_ROOT/R1_official_smoke.log" \
    "$PYTHON" main_block_ap.py --resume_quant "$OFFICIAL_DIR" --net Llama-2 \
    --wbits 2 --group_size 64 --output_dir "$LOG_ROOT/R1_official_smoke" \
    --eval_ppl --ppl_datasets wikitext2 --ppl_max_samples 1 --ppl_seqlen "$SMOKE_SEQUENCE_LENGTH" \
    --eval_tasks "$TASKS" --eval_limit 1 --eval_batch_size 1
  run_logged prepare_tiny_local_datasets "$LOG_ROOT/prepare_smoke_data.log" \
    "$PYTHON" "$SCRIPT_DIR/prepare_smoke_data.py" \
    --model-dir "$MODEL_DIR" --redpajama-file "$PROJECT_ROOT/data/local_datasets/redpajama/data/train-00000-of-00011.parquet" --blockap-cache-dir "$SMOKE_BLOCK_CACHE" \
    --blockap-net "$SMOKE_NET" --e2e-cache-file "$SMOKE_E2E_CACHE" \
    --sequence-length "$SMOKE_SEQUENCE_LENGTH" --train-size "$SMOKE_TRAIN_SIZE" --val-size "$SMOKE_VAL_SIZE"

  else
    test -s "$LOG_ROOT/R0_FP16_smoke/eval_results.json"
    test -s "$LOG_ROOT/R1_official_smoke/eval_results.json"
    test -s "$RUN_ROOT/R2_smoke/block_ap/model.safetensors"
    test -s "$SMOKE_E2E_CACHE"
    echo "Resuming existing smoke run at R2 E2E-QP"
  fi

  smoke_block_ap() {
    local exp="$1" bits="$2" group="$3" weight_lr="$4"
    local out="$RUN_ROOT/${exp}_smoke/block_ap"
    mkdir -p "$out"
    run_logged "${exp}_BlockAP_smoke" "$LOG_ROOT/${exp}_BlockAP_smoke.console.log" \
      "$PYTHON" main_block_ap.py --model "$MODEL_DIR" --net "$SMOKE_NET" \
      --wbits "$bits" --group_size "$group" --calib_dataset redpajama \
      --train_size "$SMOKE_TRAIN_SIZE" --val_size "$SMOKE_VAL_SIZE" \
      --training_seqlen "$SMOKE_SEQUENCE_LENGTH" --batch_size 2 --epochs 1 \
      --quant_lr 1e-4 --weight_lr "$weight_lr" --real_quant \
      --cache_dir "$SMOKE_BLOCK_CACHE" --output_dir "$LOG_ROOT/${exp}_BlockAP_smoke" \
      --save_quant_dir "$out"
  }
  smoke_e2e_qp() {
    local exp="$1" bits="$2" group="$3" learning_rate="$4"
    local quant_model="$RUN_ROOT/${exp}_smoke/block_ap"
    local out="$RUN_ROOT/${exp}_smoke/e2e_qp"
    mkdir -p "$out"
    run_logged "${exp}_E2E_QP_smoke" "$LOG_ROOT/${exp}_E2E_QP_smoke.console.log" \
      "$PYTHON" main_e2e_qp.py --quant_model_path "$quant_model" \
      --model_family "$SMOKE_FAMILY" --wbits "$bits" --group_size "$group" \
      --learning_rate "$learning_rate" --dataset redpajama --dataset_format pt \
      --pt_context_len "$SMOKE_SEQUENCE_LENGTH" --max_train_samples 1 \
      --eval_dataset_size "$SMOKE_VAL_SIZE" --max_eval_samples 1 \
      --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 1 --max_steps 1 --num_train_epochs 1 \
      --logging_steps 1 --save_strategy no --evaluation_strategy no \
      --preprocessing_num_workers 1 --bf16 --data_seed 42 --max_grad_norm 0.3 \
      --output_dir "$out" --do_train True --do_eval False --audit_updates True --report_to none
  }

  if [[ -z "$RESUME_FROM" ]]; then
    smoke_block_ap R2 4 128 1e-5
  fi
  smoke_e2e_qp R2 4 128 1e-5
  smoke_block_ap R3 2 64 2e-5
  mkdir -p "$RUN_ROOT/R4_smoke"
  ln -s ../R3_smoke/block_ap "$RUN_ROOT/R4_smoke/block_ap"
  smoke_e2e_qp R4 2 64 2e-5
  run_logged R2_checkpoint_reload "$LOG_ROOT/R2_reload.log" "$PYTHON" "$SCRIPT_DIR/smoke_model.py" --kind efficientqat --model-dir "$RUN_ROOT/R2_smoke/e2e_qp" --wbits 4 --group-size 128
  r5_train smoke
  smoke_checkpoint_evaluation
  mmlu_comparison smoke
  validate_smoke
  printf 'SMOKE_OK\t%s\n' "$(date -u +%FT%TZ)" >"$LOG_ROOT/SMOKE_OK"
  echo "R0-R5 smoke pipeline passed with limited eval and tiny local training data."
  exit 0
fi

if [[ -z "$RESUME_FROM" && ( -s "$LOG_ROOT/status.tsv" || -e "$LOG_ROOT/ALL_DONE" ) ]]; then
  echo "RUN_ID=$RUN_ID already has a recorded run; choose a new RUN_ID to avoid overwriting results." >&2
  exit 1
fi

block_ap() {
  local exp="$1" bits="$2" group="$3" weight_lr="$4"
  local out="$RUN_ROOT/$exp/block_ap"
  local logdir="$LOG_ROOT/$exp/block_ap"
  mkdir -p "$out" "$logdir"
  run_logged "${exp}_BlockAP" "$LOG_ROOT/${exp}_BlockAP.console.log" \
    "$PYTHON" main_block_ap.py \
    --model "$MODEL_DIR" --net Llama-2 \
    --wbits "$bits" --group_size "$group" \
    --calib_dataset redpajama --train_size 4096 --val_size 64 \
    --training_seqlen 2048 --batch_size 2 --epochs 2 \
    --quant_lr 1e-4 --weight_lr "$weight_lr" --real_quant \
    --cache_dir "$REPO_ROOT/cache" --output_dir "$logdir" \
    --save_quant_dir "$out" --eval_ppl --ppl_seqlen 2048 \
    --eval_tasks "$TASKS" --eval_batch_size 16
}

e2e_qp() {
  local exp="$1" bits="$2" group="$3" learning_rate="$4" source_checkpoint="$5"
  local out="$RUN_ROOT/$exp/e2e_qp"
  local logdir="$LOG_ROOT/$exp/e2e_qp"
  mkdir -p "$out" "$logdir"
  run_logged "${exp}_E2E_QP" "$LOG_ROOT/${exp}_E2E_QP.console.log" \
    "$PYTHON" main_e2e_qp.py \
    --quant_model_path "$source_checkpoint" --model_family Llama-2 \
    --wbits "$bits" --group_size "$group" --learning_rate "$learning_rate" \
    --dataset redpajama --dataset_format pt --pt_context_len 4096 \
    --max_train_samples 4096 --eval_dataset_size 64 --max_eval_samples 64 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 --num_train_epochs 1 \
    --logging_steps 10 --save_strategy epoch --evaluation_strategy steps \
    --eval_steps 64 --bf16 --data_seed 42 --max_grad_norm 0.3 \
    --eval_tasks "$TASKS" --preprocessing_num_workers 8 --do_ppl_eval \
    --output_dir "$out" --do_train True --report_to none
}

check_eval_gate() {
  local step="$1" logfile="$2" expected_ppl="$3" expected_acc="$4"
  local ppl_tol="${R0_R1_PPL_TOLERANCE:-1.0}"
  local acc_tol="${R0_R1_ACC_TOLERANCE:-5.0}"
  "$PYTHON" - "$step" "$logfile" "$expected_ppl" "$expected_acc" "$ppl_tol" "$acc_tol" <<'PY'
import re
import sys

step, path, expected_ppl, expected_acc, ppl_tol, acc_tol = sys.argv[1:]
text = open(path, encoding="utf-8", errors="replace").read()
ppl = re.findall(r"wikitext2 perplexity:\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
acc = re.findall(r"Average Acc:\s*([0-9]+(?:\.[0-9]+)?)%", text, re.I)
if not ppl or not acc:
    raise SystemExit(f"{step}: required WikiText2 PPL / Average Acc not found; stopping before QAT")
ppl, acc = float(ppl[-1]), float(acc[-1])
ep, ea, tp, ta = map(float, (expected_ppl, expected_acc, ppl_tol, acc_tol))
print(f"{step} gate: WikiText2 PPL={ppl:.2f} (reference {ep:.2f} ± {tp:.2f}); "
      f"Average Acc={acc:.2f}% (reference {ea:.2f}% ± {ta:.2f} pp)")
if abs(ppl - ep) > tp or abs(acc - ea) > ta:
    raise SystemExit(f"{step}: outside configured tolerance; stopping before QAT")
PY
}

# R0: FP16 baseline. This gate prevents expensive quantization if the baseline is broken.
run_logged R0_FP16_evaluation "$LOG_ROOT/R0_FP16.console.log" \
  "$PYTHON" main_block_ap.py --model "$MODEL_DIR" --net Llama-2 \
  --wbits 16 --group_size 64 --output_dir "$LOG_ROOT/R0_FP16" \
  --eval_ppl --ppl_seqlen 2048 --eval_tasks "$TASKS" --eval_batch_size 16
check_eval_gate R0 "$LOG_ROOT/R0_FP16.console.log" 5.47 64.86

# R1: official W2g64 checkpoint sanity gate.
run_logged R1_official_W2g64_evaluation "$LOG_ROOT/R1_official_W2g64.console.log" \
  "$PYTHON" main_block_ap.py --resume_quant "$OFFICIAL_DIR" --net Llama-2 \
  --wbits 2 --group_size 64 --output_dir "$LOG_ROOT/R1_official_W2g64" \
  --eval_ppl --ppl_seqlen 2048 --eval_tasks "$TASKS" --eval_batch_size 16
check_eval_gate R1 "$LOG_ROOT/R1_official_W2g64.console.log" 6.86 60.14

# R2: W4g128 Block-AP + E2E-QP.
block_ap R2 4 128 1e-5
e2e_qp R2 4 128 1e-5 "$RUN_ROOT/R2/block_ap"

# R3: W2g64 Block-AP only; preserve this checkpoint as its own result.
block_ap R3 2 64 2e-5

# R4: independent W2g64 Block-AP + E2E-QP core reproduction.
e2e_qp R4 2 64 2e-5 "$RUN_ROOT/R3/block_ap"
r5_train full
mmlu_comparison full

CURRENT_STEP="all_experiments"
printf 'ALL_R0_R5_DONE\t%s\n' "$(date -u +%FT%TZ)" >"$LOG_ROOT/ALL_DONE"
echo "All six experiments completed. Results: $RUN_ROOT; logs/status: $LOG_ROOT"
