import numpy as np
from backend.config import USE_SECOND_ORDER_MOTION

def compute_motion_features(landmarks: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Computes mask-aware first (and optional second) order motion features.
    landmarks: [T, N, D]  (3-D landmark array)
    mask:      [T, 3]     (optional validity mask — left_hand, right_hand, pose)
    Returns:   [T, N*D] or [T, N*D*2]

    Rule: motion at frame t is non-zero only when BOTH frame t-1 AND frame t
    have at least one valid landmark group detected. This prevents fake spikes
    caused by MediaPipe detection flicker (landmark appearing/disappearing).
    """
    T, N, D = landmarks.shape
    flat = landmarks.reshape(T, N * D)

    # Build per-frame validity flag: True if any landmark group is detected
    if mask is not None:
        frame_valid = mask.sum(axis=1) > 0  # [T] bool
    else:
        frame_valid = np.ones(T, dtype=bool)

    # First-order motion
    first_order = np.zeros_like(flat)
    for t in range(1, T):
        if frame_valid[t] and frame_valid[t - 1]:
            first_order[t] = flat[t] - flat[t - 1]
        # else: stays zero — no fake spike

    if USE_SECOND_ORDER_MOTION:
        second_order = np.zeros_like(first_order)
        for t in range(1, T):
            if frame_valid[t] and frame_valid[t - 1]:
                second_order[t] = first_order[t] - first_order[t - 1]
        return np.concatenate([first_order, second_order], axis=-1)

    return first_order
