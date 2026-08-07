import numpy as np

def normalize_landmarks(landmarks: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Normalizes coordinates.
    landmarks shape: [T, 75, 3] where:
      - 0 to 20: Left Hand
      - 21 to 41: Right Hand
      - 42 to 74: Pose
    mask shape: [T, 3] where mask[t] = [lh_valid, rh_valid, pose_valid]
    
    Normalization steps:
      - Center and scale hands relative to wrist (index 0 of each hand)
      - Center and scale pose relative to shoulder mid-point and shoulder distance.
    """
    T, N, D = landmarks.shape
    normalized = np.zeros_like(landmarks)
    
    for t in range(T):
        lh_valid = mask[t, 0] > 0.5
        rh_valid = mask[t, 1] > 0.5
        pose_valid = mask[t, 2] > 0.5
        
        # 1. Left Hand (0 to 21)
        if lh_valid:
            lh = landmarks[t, 0:21].copy()
            wrist = lh[0] # Left hand wrist
            lh_centered = lh - wrist
            # Scale by max distance to any joint to normalize size to [0, 1] bounds roughly
            scale = np.max(np.linalg.norm(lh_centered, axis=1))
            if scale > 1e-5:
                lh_centered /= scale
            normalized[t, 0:21] = lh_centered
            
        # 2. Right Hand (21 to 42)
        if rh_valid:
            rh = landmarks[t, 21:42].copy()
            wrist = rh[0] # Right hand wrist
            rh_centered = rh - wrist
            scale = np.max(np.linalg.norm(rh_centered, axis=1))
            if scale > 1e-5:
                rh_centered /= scale
            normalized[t, 21:42] = rh_centered
            
        # 3. Pose (42 to 75)
        if pose_valid:
            pose = landmarks[t, 42:75].copy()
            # Left shoulder is Pose index 11 (42 + 11 = 53), Right shoulder is index 12 (42 + 12 = 54)
            left_shoulder = pose[11]
            right_shoulder = pose[12]
            shoulder_mid = (left_shoulder + right_shoulder) / 2.0
            pose_centered = pose - shoulder_mid
            
            # Scale by distance between shoulders
            scale = np.linalg.norm(left_shoulder - right_shoulder)
            if scale > 1e-5:
                pose_centered /= scale
            normalized[t, 42:75] = pose_centered

    # Ensure no NaN or Inf values
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    return normalized
