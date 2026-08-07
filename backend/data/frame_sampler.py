import numpy as np

def get_sampled_indices(total_frames: int, target_frames: int = 32) -> list:
    """
    Uniformly samples target_frames indices from total_frames.
    Handles short videos (fewer frames than target_frames) using final-frame repetition.
    """
    if total_frames <= 0:
        return [0] * target_frames

    if total_frames >= target_frames:
        # Uniform sampling across the complete video duration
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        return list(indices)
    else:
        # Short video handling: uniform duplication / repeat final frame
        # Repeat final frame strategy
        indices = list(range(total_frames))
        while len(indices) < target_frames:
            indices.append(total_frames - 1)
        return indices
