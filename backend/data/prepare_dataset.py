import os
import cv2
import pandas as pd
from pathlib import Path
from backend.config import METADATA_DIR, VIDEO_DIR, PROCESSED_DIR, CLASSES
from backend.data.split_manager import parse_wlasl_splits, get_fallback_split

def check_video_validity(video_path: str) -> bool:
    """Returns True if OpenCV can open the video and it has > 0 frames."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames > 0
    except Exception:
        return False

def prepare_dataset():
    print("[PREPARATION] Starting dataset preparation...")
    
    # 1. Look for metadata files
    wlasl_json = METADATA_DIR / "WLASL_v0.3.json"
    
    samples = None
    if wlasl_json.exists():
        print(f"[PREPARATION] Metadata JSON found: {wlasl_json}")
        samples = parse_wlasl_splits(wlasl_json, VIDEO_DIR)
    
    # 2. Fallback if no JSON or zero matching samples found
    if not samples:
        print("[PREPARATION] No metadata matching classes. Scanning raw video directory directly...")
        samples = []
        # In Kaggle WLASL datasets, sometimes videos are named directly, or there's a custom mapping.
        # If we fall back, we will scan the video directory. We might need a way to assign labels.
        # Let's assume files might be named like "label_vid.mp4" or we search for label substrings.
        # For academic completeness, let's scan video filenames containing classes.
        for path in VIDEO_DIR.glob("*.mp4"):
            filename = path.stem.lower()
            for cls in CLASSES:
                # If filename contains class name, e.g. "book_001" or starts with "book"
                if filename.startswith(cls) or f"_{cls}" in filename or f"{cls}_" in filename:
                    video_id = path.stem
                    samples.append({
                        "video_id": video_id,
                        "video_path": str(path),
                        "label": cls,
                        "split": get_fallback_split(video_id)
                    })
                    break

    # 3. Sanity check: Validate videos and split
    valid_samples = []
    corrupted_count = 0
    missing_count = 0
    
    print(f"[PREPARATION] Found {len(samples)} candidates. Validating files...")
    
    for s in samples:
        v_path = Path(s["video_path"])
        if not v_path.exists():
            missing_count += 1
            continue
            
        if check_video_validity(str(v_path)):
            # Assign label_id
            s["label_id"] = CLASSES.index(s["label"])
            valid_samples.append(s)
        else:
            corrupted_count += 1

    # 4. Generate splits
    df = pd.DataFrame(valid_samples)
    if df.empty:
        print("[ERROR] No valid videos found for the selected classes! Please ensure videos exist in dataset/raw/videos/")
        return

    splits_dir = PROCESSED_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    # Prevent Data Leakage check
    train_ids = set(train_df["video_id"])
    val_ids = set(val_df["video_id"])
    test_ids = set(test_df["video_id"])

    assert len(train_ids.intersection(val_ids)) == 0, "Data Leakage detected between Train and Val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Data Leakage detected between Train and Test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Data Leakage detected between Val and Test!"
    
    # Save CSVs
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print("\n" + "="*50)
    print("DATASET PREPARATION REPORT")
    print("="*50)
    print(f"Total videos candidates : {len(samples)}")
    print(f"Total valid videos      : {len(df)}")
    print(f"Missing videos          : {missing_count}")
    print(f"Corrupted videos        : {corrupted_count}")
    print(f"Training videos (train) : {len(train_df)}")
    print(f"Validation videos (val) : {len(val_df)}")
    print(f"Testing videos (test)   : {len(test_df)}")
    print("="*50)
    
    # Class statistics
    stats = []
    for cls in CLASSES:
        cls_df = df[df["label"] == cls]
        tr_c = len(cls_df[cls_df["split"] == "train"])
        va_c = len(cls_df[cls_df["split"] == "val"])
        te_c = len(cls_df[cls_df["split"] == "test"])
        stats.append({
            "Class": cls,
            "Train": tr_c,
            "Val": va_c,
            "Test": te_c,
            "Total": len(cls_df)
        })
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    print("="*50)
    
    # Save stats to processed for evaluation reference
    stats_df.to_csv(PROCESSED_DIR / "dataset_stats.csv", index=False)

if __name__ == "__main__":
    prepare_dataset()
