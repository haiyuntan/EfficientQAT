#!/usr/bin/env python
"""Prepare deterministic tiny local caches so smoke mode does not fetch corpora."""

import argparse
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def make_ids(tokenizer, text: str, length: int) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise RuntimeError("Tokenizer produced no tokens for smoke text")
    return (ids * ((length + len(ids) - 1) // len(ids)))[:length]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--blockap-cache-dir", required=True)
    parser.add_argument("--blockap-net", required=True)
    parser.add_argument("--e2e-cache-file", required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=2)
    parser.add_argument("--val-size", type=int, default=2)
    parser.add_argument("--redpajama-file", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    import pyarrow.parquet as pq
    texts = []
    for batch in pq.ParquetFile(args.redpajama_file).iter_batches(batch_size=32, columns=['text']):
        for text in batch.column(0).to_pylist():
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) >= args.sequence_length:
                texts.append(ids[:args.sequence_length])
            if len(texts) >= args.train_size + args.val_size:
                break
        if len(texts) >= args.train_size + args.val_size:
            break
    if len(texts) < args.train_size + args.val_size:
        raise RuntimeError("Insufficient real calibration samples")
    splits = {'train': texts[:args.train_size], 'val': texts[args.train_size:]}
    block_cache = Path(args.blockap_cache_dir)
    block_cache.mkdir(parents=True, exist_ok=True)
    for split, count in (("train", args.train_size), ("val", args.val_size)):
        examples = []
        for index in range(count):
            ids = torch.tensor(
                [splits[split][index]],
                dtype=torch.long,
            )
            labels = ids.clone()
            labels[:, :-1] = -100
            examples.append((ids, labels))
        cache_file = block_cache / (
            f"dataloader_{args.blockap_net}_redpajama_{args.train_size}_"
            f"{args.val_size}_{args.sequence_length}_{split}.cache"
        )
        torch.save(examples, cache_file)

    e2e_cache = Path(args.e2e_cache_file)
    e2e_cache.parent.mkdir(parents=True, exist_ok=True)

    def as_dataset(count: int, split: str) -> Dataset:
        rows = []
        for index in range(count):
            ids = splits['val' if split == 'validation' else 'train'][index]
            rows.append({"input_ids": ids, "attention_mask": [1] * args.sequence_length, "labels": ids.copy()})
        return Dataset.from_list(rows)

    smoke_dataset = DatasetDict(
        {
            "train": as_dataset(args.train_size, "train"),
            "validation": as_dataset(args.val_size, "validation"),
        }
    )
    torch.save(smoke_dataset, e2e_cache)
    print(f"Prepared tiny local smoke caches: {block_cache} and {e2e_cache}", flush=True)


if __name__ == "__main__":
    main()
