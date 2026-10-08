"""Local-only dataset entry point for training and the lm-eval harness."""
import os
from functools import lru_cache
from pathlib import Path

# Set before importing datasets/transformers: experimental runs never query the Hub.
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "hf_cache/datasets"))
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import datasets

_ORIGINAL_LOAD = datasets.load_dataset
DATA = Path(os.environ.get("LOCAL_DATASETS_DIR", ROOT / "data/local_datasets"))
CACHE = Path(os.environ["HF_DATASETS_CACHE"])


def _arrow(directory, prefix, splits):
    result = datasets.DatasetDict()
    for split in splits:
        files = sorted(directory.rglob(f"{prefix}-{split}.arrow"))
        if len(files) != 1:
            raise FileNotFoundError(f"Expected one local {prefix}/{split} Arrow file under {directory}, found {len(files)}")
        result[split] = datasets.Dataset.from_file(str(files[0]))
    return result


def _files(builder, directory, pattern="*.parquet"):
    files = sorted(directory.glob(pattern))
    if not files or any(f.stat().st_size == 0 for f in files):
        raise FileNotFoundError(f"Missing or empty local data: {directory}/{pattern}")
    mapping = {}
    for f in files:
        split = "validation" if "validation" in f.name else "test" if "test" in f.name else "train"
        mapping.setdefault(split, []).append(str(f))
    return _ORIGINAL_LOAD(builder, data_files=mapping)


@lru_cache(maxsize=None)
def _load(path, name):
    if path in ("wikitext", "Salesforce/wikitext"):
        if name != "wikitext-2-raw-v1":
            raise ValueError(f"Unsupported local WikiText configuration: {name}")
        return _arrow(CACHE / "wikitext/wikitext-2-raw-v1", "wikitext", ("train", "validation", "test"))
    if path in ("hails/mmlu_no_train", "cais/mmlu"):
        if not name or name == "all":
            raise ValueError("Local MMLU requires an individual subject configuration")
        return _arrow(CACHE / "hails___mmlu_no_train" / name, "mmlu_no_train", ("test", "validation", "dev"))
    if path in ("allenai/ai2_arc", "ai2_arc"):
        if name not in ("ARC-Easy", "ARC-Challenge"):
            raise ValueError(f"Invalid ARC configuration: {name}")
        return _files("parquet", DATA / "arc" / name)
    directories = {"piqa": "piqa", "ybisk/piqa": "piqa", "hellaswag": "hellaswag", "Rowan/hellaswag": "hellaswag", "winogrande": "winogrande/winogrande_xl", "allenai/winogrande": "winogrande/winogrande_xl"}
    if path in directories:
        return _files("parquet", DATA / directories[path])
    if path == "togethercomputer/RedPajama-Data-1T-Sample":
        shards = list((DATA / "redpajama/data").glob("train-*-of-00011.parquet"))
        if len(shards) != 11:
            raise FileNotFoundError(f"RedPajama needs 11 local shards; found {len(shards)}")
        return _files("parquet", DATA / "redpajama/data")
    if path == "allenai/c4":
        return _files("json", DATA / "c4/en", "*.json.gz")
    if path == "tatsu-lab/alpaca":
        return _files("json", DATA / "alpaca", "alpaca_data.json")
    raise FileNotFoundError(f"Dataset {path!r} ({name!r}) has no local mapping; download it and register its files before running")


def load_dataset(path, name=None, **kwargs):
    """Ignore Hub revisions and resolve supported datasets from explicit local files."""
    if path in ("json", "parquet", "arrow", "csv", "text"):
        # Built-in readers can only receive existing local files here.
        supplied = kwargs.get("data_files")
        if not supplied:
            raise ValueError("Local readers require data_files")
        values = supplied.values() if isinstance(supplied, dict) else [supplied]
        for value in values:
            for filename in value if isinstance(value, (list, tuple)) else [value]:
                if not Path(filename).is_file():
                    raise FileNotFoundError(f"Local data file missing: {filename}")
        return _ORIGINAL_LOAD(path, name=name, **kwargs)
    result = _load(path, name)
    split = kwargs.get("split")
    return result[split] if split else result


# lm-eval calls datasets.load_dataset directly. Route it through the same resolver.
datasets.load_dataset = load_dataset
