import json
import hashlib
from pathlib import Path
from backend.config import CLASSES, RANDOM_SEED

def get_fallback_split(video_id: str, train_ratio: float = 0.70, val_ratio: float = 0.15) -> str:
    """
    Deterministically assigns a video_id to train/val/test using MD5 hash.
    """
    hash_val = int(hashlib.md5(str(video_id).encode('utf-8')).hexdigest(), 16) % 100
    if hash_val < int(train_ratio * 100):
        return "train"
    elif hash_val < int((train_ratio + val_ratio) * 100):
        return "val"
    return "test"

def parse_wlasl_splits(metadata_json_path: Path, video_dir: Path):
    """
    Parses WLASL JSON, filters to configured classes, checks video availability.
    Enforces signer-level split integrity: a signer present in train must NOT
    appear in val or test (prevents identity leakage).
    """
    if not metadata_json_path.exists():
        print(f"[WARNING] Metadata file {metadata_json_path} not found. Fallback splitting will be used.")
        return None

    try:
        with open(metadata_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not parse metadata json: {e}")
        return None

    target_classes = set(CLASSES)

    # First pass: collect all instances and build signer -> split mapping
    raw_instances = []
    signer_split: dict[str, str] = {}  # signer_id -> first assigned split

    for entry in data:
        gloss = entry.get("gloss", "").strip().lower()
        if gloss not in target_classes:
            continue
        for inst in entry.get("instances", []):
            video_id  = str(inst.get("video_id"))
            signer_id = str(inst.get("signer_id", ""))
            split     = inst.get("split", "")
            if split not in ("train", "val", "test"):
                split = get_fallback_split(video_id)

            # Signer-level consistency: once a signer is assigned to a split, keep it
            if signer_id and signer_id in signer_split:
                split = signer_split[signer_id]
            elif signer_id:
                signer_split[signer_id] = split

            video_path = video_dir / f"{video_id}.mp4"
            if video_path.exists():
                raw_instances.append({
                    "video_id":   video_id,
                    "video_path": str(video_path),
                    "label":      gloss,
                    "split":      split,
                    "signer_id":  signer_id,
                })

    return raw_instances
