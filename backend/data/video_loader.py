import cv2
import numpy as np
from backend.data.frame_sampler import get_sampled_indices

def load_video(video_path: str, target_frames: int = 32, image_size: int = 224) -> np.ndarray:
    """
    Decodes an MP4 video file, extracts target_frames uniformly, 
    resizes to (image_size, image_size), and returns an array of shape [T, H, W, C] in RGB.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_indices = get_sampled_indices(total_frames, target_frames)
    
    frames = []
    frame_idx = 0
    success = True
    
    # Cache all frames to allow random access or step through them
    all_frames = []
    while success:
        success, frame = cap.read()
        if success:
            all_frames.append(frame)
    cap.release()

    if len(all_frames) == 0:
        # Fallback to zero frames if empty
        return np.zeros((target_frames, image_size, image_size, 3), dtype=np.uint8)

    for idx in sampled_indices:
        # Safeguard index out of bounds
        actual_idx = min(idx, len(all_frames) - 1)
        frame = all_frames[actual_idx]
        
        # Convert BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame_resized = cv2.resize(frame_rgb, (image_size, image_size))
        frames.append(frame_resized)
        
    return np.stack(frames, axis=0) # [T, H, W, C]
