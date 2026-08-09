import numpy as np

def normalize_landmarks(landmarks: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Normalizes landmarks using VIDEO-LEVEL fixed scale to avoid per-frame noise.
    landmarks: [T, 75, 3]  mask: [T, 3]
    Returns:   [T, 75, 3]

    Strategy:
      - Hands: centered on wrist; scale = median of per-frame max-joint-distances
        across all valid frames (video-level stable reference).
      - Pose: centered on shoulder midpoint; scale = median shoulder distance
        across all valid pose frames.
    """
    T, N, D = landmarks.shape
    normalized = np.zeros_like(landmarks)

    # --- Pre-compute video-level scales ---
    lh_scales, rh_scales, pose_scales = [], [], []
    for t in range(T):
        if mask[t, 0] > 0.5:
            lh = landmarks[t, 0:21]
            d = np.max(np.linalg.norm(lh - lh[0], axis=1))
            if d > 1e-5:
                lh_scales.append(d)
        if mask[t, 1] > 0.5:
            rh = landmarks[t, 21:42]
            d = np.max(np.linalg.norm(rh - rh[0], axis=1))
            if d > 1e-5:
                rh_scales.append(d)
        if mask[t, 2] > 0.5:
            pose = landmarks[t, 42:75]
            d = np.linalg.norm(pose[11] - pose[12])  # shoulder distance
            if d > 1e-5:
                pose_scales.append(d)

    lh_scale   = float(np.median(lh_scales))   if lh_scales   else 1.0
    rh_scale   = float(np.median(rh_scales))   if rh_scales   else 1.0
    pose_scale = float(np.median(pose_scales)) if pose_scales else 1.0

    # --- Apply per-frame centering with video-level scale ---
    for t in range(T):
        if mask[t, 0] > 0.5:
            lh = landmarks[t, 0:21].copy()
            normalized[t, 0:21] = (lh - lh[0]) / lh_scale

        if mask[t, 1] > 0.5:
            rh = landmarks[t, 21:42].copy()
            normalized[t, 21:42] = (rh - rh[0]) / rh_scale

        if mask[t, 2] > 0.5:
            pose = landmarks[t, 42:75].copy()
            shoulder_mid = (pose[11] + pose[12]) / 2.0
            normalized[t, 42:75] = (pose - shoulder_mid) / pose_scale

    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    return normalized
