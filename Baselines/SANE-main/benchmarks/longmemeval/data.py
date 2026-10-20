import json
import os
import urllib.request

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
DATASET_REPO = "xiaowu0162/longmemeval-cleaned"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "benchmarks", "longmemeval")

_FILE_MAP = {
    "oracle": "longmemeval_oracle.json",
    "s_cleaned": "longmemeval_s_cleaned.json",
    "m_cleaned": "longmemeval_m_cleaned.json",
}


def _download_file(filename: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, filename)
    if os.path.exists(filepath):
        return filepath

    url = f"{HF_ENDPOINT}/datasets/{DATASET_REPO}/resolve/main/{filename}"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, filepath)
    print(f"Saved to {filepath}")
    return filepath


def load_longmemeval_data(split: str = "oracle", cache_dir: str | None = None) -> list[dict]:
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    filename = _FILE_MAP.get(split)
    if filename is None:
        raise ValueError(f"Unknown split: {split}. Use 'oracle', 's_cleaned', or 'm_cleaned'.")

    filepath = _download_file(filename, cache_dir)
    with open(filepath) as f:
        return json.load(f)



def _json_default(obj):
    """Handle non-JSON-serialisable objects (ellipsis, etc.) gracefully."""
    if isinstance(obj, type(...)):
        return "..."
    try:
        return str(obj)
    except Exception:
        return "<non-serializable>"


def save_predictions(predictions: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False, default=_json_default) + "\n")


def load_predictions(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]
