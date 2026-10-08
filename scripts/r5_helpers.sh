#!/usr/bin/env bash
r5_train() {
  local mode="$1" suffix="" steps=10000 batch=16 source_len=384 target_len=128
  local extras=(--save_strategy steps --save_steps 2000 --save_total_limit 2 --evaluation_strategy steps --eval_steps 2000 --group_by_length)
  if [[ "$mode" == smoke ]]; then
    suffix=_smoke; steps=2; batch=1; source_len=96; target_len=32
    extras=(--save_strategy no --evaluation_strategy no --max_train_samples 4 --audit_updates True)
  fi
  export ALPACA_DATA_FILE="$PROJECT_ROOT/data/local_datasets/alpaca/alpaca_data.json"
  [[ -s "$ALPACA_DATA_FILE" ]] || { echo "Missing Alpaca: $ALPACA_DATA_FILE" >&2; return 1; }
  run_logged R5_Alpaca_E2E "$LOG_ROOT/R5_Alpaca.console.log" \
    "$PYTHON" main_e2e_qp.py --quant_model_path "$RUN_ROOT/R3${suffix}/block_ap" \
    --model_family Llama-2 --wbits 2 --group_size 64 --learning_rate 2e-5 \
    --dataset alpaca --dataset_format alpaca --source_max_len "$source_len" --target_max_len "$target_len" \
    --per_device_train_batch_size "$batch" --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 \
    --max_steps "$steps" --eval_dataset_size 16 --logging_steps 1 --bf16 --data_seed 42 --max_grad_norm 0.3 \
    --output_dir "$RUN_ROOT/R5${suffix}/e2e_qp" --do_train True --report_to none "${extras[@]}"
}
mmlu_comparison() {
  local mode="$1" suffix="" extra=()
  if [[ "$mode" == smoke ]]; then suffix=_smoke; extra=(--eval_limit 1); fi
  # Identical 5-shot prompts for FP16, Block-AP, generic E2E, and Alpaca E2E.
  local exp path
  for exp in R0 R3 R4 R5; do
    case "$exp" in
      R0) path="$MODEL_DIR" ;;
      R3) path="$RUN_ROOT/R3${suffix}/block_ap" ;;
      *) path="$RUN_ROOT/${exp}${suffix}/e2e_qp" ;;
    esac
    local load=(--resume_quant "$path" --wbits 2)
    if [[ "$exp" == R0 ]]; then load=(--model "$path" --wbits 16); fi
    run_logged "${exp}_MMLU_reload" "$LOG_ROOT/${exp}_MMLU.console.log" \
      "$PYTHON" main_block_ap.py "${load[@]}" --net Llama-2 --group_size 64 \
      --output_dir "$LOG_ROOT/${exp}_MMLU" --eval_tasks mmlu --num_fewshot 5 --eval_batch_size 1 "${extra[@]}"
  done
}
validate_smoke() {
  run_logged smoke_artifact_validation "$LOG_ROOT/validation.log" "$PYTHON" "$SCRIPT_DIR/validate_smoke.py" "$RUN_ROOT" "$LOG_ROOT"
}
