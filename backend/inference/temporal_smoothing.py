import numpy as np
from collections import deque
from backend.config import TEMPORAL_SMOOTHING_WINDOW

class TemporalSmoothing:
    """
    Maintains a rolling buffer of probability distributions from recent inferences
    and averages them to reduce label flicker/fluctuation.
    """
    def __init__(self, window_size: int = TEMPORAL_SMOOTHING_WINDOW):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, prob_vector: np.ndarray) -> np.ndarray:
        """
        prob_vector: array of shape [num_classes]
        Returns: averaged probability vector of shape [num_classes]
        """
        self.buffer.append(prob_vector)
        
        # Calculate mean along the time axis
        mean_probs = np.mean(self.buffer, axis=0)
        return mean_probs

    def clear(self):
        self.buffer.clear()
