import torch
from pathlib import Path
from backend.config import CHECKPOINT_DIR, CLASSES, NUM_CLASSES, NUM_FRAMES, USE_MOTION
from backend.models.lightmamba_asl import LightMambaASL
from backend.utils import get_device

_model_instance = None
_device = None

def get_loaded_model():
    """Lazily loads and returns the model instance in eval mode."""
    global _model_instance, _device
    if _model_instance is not None:
        return _model_instance, _device
        
    _device = get_device()
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    
    # Initialize architecture
    model = LightMambaASL(pretrained=False, freeze_backbone=False)
    
    if checkpoint_path.exists():
        print(f"[SERVICE] Loading best model weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=_device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"[SERVICE] WARNING: Checkpoint not found at {checkpoint_path}. Model running with randomized weights.")
        
    model.to(_device)
    model.eval()
    _model_instance = model
    return _model_instance, _device

def get_checkpoint_status() -> dict:
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    return {
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None
    }
