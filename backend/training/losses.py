import torch
import torch.nn as nn
import numpy as np

def get_loss_fn(class_counts: list = None, use_weights: bool = False) -> nn.Module:
    """
    Returns CrossEntropyLoss. Optionally applies class weights to handle imbalance.
    """
    if use_weights and class_counts is not None:
        total = sum(class_counts)
        # Inverse frequency weighting
        weights = [total / (len(class_counts) * count) if count > 0 else 0 for count in class_counts]
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        print(f"[LOSS] Using class-weighted CrossEntropyLoss with weights: {weights}")
        return nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        return nn.CrossEntropyLoss()
