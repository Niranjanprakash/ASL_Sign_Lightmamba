import json
import hashlib
from pathlib import Path
from backend.config import CLASSES, RANDOM_SEED

def get_fallback_split(video_id: str, train_ratio: float = 0.70, val_ratio: float = 0.15) -> str:
    """
    Deterministically assigns a video_id to train, val, or test split using MD5 hash.
    Ensures zero leakage across executions.
    """
    hash_val = int(hashlib.md5(video_id.encode('utf-8')).hexdigest(), 16) % 100
    if hash_val < (train_ratio * 100):
        return "train"
    elif hash_val < ((train_ratio + val_ratio) * 100):
        return "val"
    else:
        return "test"

def parse_wlasl_splits(metadata_json_path: Path, video_dir: Path):
    """
    Parses the WLASL JSON file, filter classes to configured 10,
    check video file availability, and assign to split.
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

    samples = []
    # Build class list matching exactly the config classes
    target_classes = set(CLASSES)

    for entry in data:
        gloss = entry.get("gloss", "").strip().lower()
        if gloss not in target_classes:
            continue

        instances = entry.get("instances", [])
        for inst in instances:
            video_id = str(inst.get("video_id"))
            split = inst.get("split")
            
            # WLASL uses train, val, test
            if split not in ["train", "val", "test"]:
                split = get_fallback_split(video_id)

            # Check if video exists (Kaggle WLASL usually saves them as video_id.mp4)
            # Support both raw name or video_id matching
            video_path = video_dir / f"{video_id}.mp4"
            if video_path.exists():
                samples.append({
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "label": gloss,
                    "split": split
                })
                
    return samples
