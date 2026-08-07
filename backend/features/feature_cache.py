import pickle
import numpy as np
from pathlib import Path
from backend.config import PROCESSED_DIR
from backend.features.landmark_extractor import LandmarkExtractor
from backend.features.landmark_normalizer import normalize_landmarks
from backend.features.motion_features import compute_motion_features

_extractor = None

def get_landmark_extractor():
    global _extractor
    if _extractor is None:
        _extractor = LandmarkExtractor()
    return _extractor

def get_or_extract_features(video_id: str, frames: np.ndarray = None) -> dict:
    """
    Looks for cached features under dataset/processed/landmarks/{video_id}.pkl.
    If not found and frames is provided, runs extraction, normalizes,
    computes motion features, saves cache, and returns.
    """
    cache_dir = PROCESSED_DIR / "landmarks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{video_id}.pkl"
    
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"[CACHE] Error reading cache for {video_id}, re-extracting: {e}")

    # Extract if not cached and frames are available
    if frames is None:
        raise ValueError(f"Features not cached for {video_id} and frames not provided!")

    extractor = get_landmark_extractor()
    raw_landmarks, mask = extractor.extract_video_sequence(frames)
    
    # Normalize
    normalized = normalize_landmarks(raw_landmarks, mask)
    
    # Compute Motion
    motion = compute_motion_features(normalized)
    
    # Flatten landmarks to [T, 75 * 3 = 225] for model feeding
    T, N, D = normalized.shape
    flat_landmarks = normalized.reshape(T, N * D)

    data = {
        "landmarks": flat_landmarks, # [T, 225]
        "mask": mask,                # [T, 3]
        "motion": motion             # [T, 225] or [T, 450]
    }
    
    # Save to cache
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[CACHE] Error writing cache for {video_id}: {e}")
        
    return data
