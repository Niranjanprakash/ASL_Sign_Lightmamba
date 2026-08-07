import numpy as np
from backend.config import USE_SECOND_ORDER_MOTION

def compute_motion_features(normalized_landmarks: np.ndarray) -> np.ndarray:
    """
    Computes first-order differences (displacement) of landmarks across time.
    normalized_landmarks shape: [T, N, D]
    Returns motion features of shape [T, N * D] or [T, N * D * 2] if second order is enabled.
    """
    T, N, D = normalized_landmarks.shape
    flat_landmarks = normalized_landmarks.reshape(T, N * D)
    
    # First-order motion: delta_t = L_t - L_{t-1}
    first_order = np.zeros_like(flat_landmarks)
    first_order[1:] = flat_landmarks[1:] - flat_landmarks[:-1]
    
    if USE_SECOND_ORDER_MOTION:
        # Second-order motion: delta2_t = delta_t - delta_{t-1}
        second_order = np.zeros_like(first_order)
        second_order[1:] = first_order[1:] - first_order[:-1]
        
        # Concatenate first and second order
        motion_features = np.concatenate([first_order, second_order], axis=-1)
    else:
        motion_features = first_order
        
    return motion_features
