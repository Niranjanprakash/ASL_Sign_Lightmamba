import numpy as np
from collections import deque
from backend.config import TEMPORAL_SMOOTHING_WINDOW

class TemporalSmoothing:
    """
    Exponential weighted averaging over a rolling buffer of probability distributions.
    Recent frames get higher weight — reduces flicker while reacting to sign changes faster.
    """
    def __init__(self, window_size: int = TEMPORAL_SMOOTHING_WINDOW):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        # Exponential weights: most recent frame gets highest weight
        self._weights = np.array([np.exp(0.4 * i) for i in range(window_size)])

    def update(self, prob_vector: np.ndarray) -> np.ndarray:
        self.buffer.append(prob_vector)
        n = len(self.buffer)
        weights = self._weights[-n:]
        weights = weights / weights.sum()
        stacked = np.stack(list(self.buffer), axis=0)  # [n, num_classes]
        return np.average(stacked, axis=0, weights=weights)

    def clear(self):
        self.buffer.clear()
