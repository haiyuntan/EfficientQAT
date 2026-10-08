#!/usr/bin/env python
"""One-batch model-load/forward smoke test for the local Llama-2 checkpoints."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Python puts this file's directory first on sys.path; project imports live one
# directory above it (e.g. quantize.int_linear_real).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("fp16", "efficientqat"), required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--wbits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if args.kind == "fp16":
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        from accelerate import dispatch_model, infer_auto_device_map
        from quantize.int_linear_real import load_quantized_model

        model, tokenizer = load_quantized_model(
            args.model_dir, args.wbits, args.group_size
        )
        first_layer_class = model.model.layers[0].__class__.__name__
        device_map = infer_auto_device_map(
            model,
            max_memory={index: "70GiB" for index in range(torch.cuda.device_count())},
            no_split_module_classes=[first_layer_class],
        )
        model = dispatch_model(model, device_map=device_map)

    model.eval()
    batch = tokenizer("A short model smoke test.", return_tensors="pt")
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        logits = model(**batch).logits
    if logits.ndim != 3 or not torch.isfinite(logits[:, -1, :]).all().item():
        raise RuntimeError("Forward pass produced invalid logits")
    print(
        f"SMOKE_OK kind={args.kind} shape={tuple(logits.shape)} "
        f"last_token_logit_std={logits[:, -1, :].float().std().item():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
