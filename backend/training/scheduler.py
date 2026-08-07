import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str = "ReduceLROnPlateau") -> torch.optim.lr_scheduler._LRScheduler:
    """
    Returns a learning rate scheduler.
    """
    if scheduler_type == "ReduceLROnPlateau":
        print("[SCHEDULER] Using ReduceLROnPlateau scheduler (monitoring validation loss).")
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif scheduler_type == "CosineAnnealingLR":
        print("[SCHEDULER] Using CosineAnnealingLR scheduler.")
        return CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
