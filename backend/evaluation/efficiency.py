import time
import torch
from pathlib import Path
from backend.config import CHECKPOINT_DIR, NUM_FRAMES, USE_MOTION

def measure_efficiency(model: torch.nn.Module, device: torch.device) -> dict:
    """
    Measures model size, parameter counts, and average inference latency.
    """
    model.eval()
    
    # 1. Parameter Counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. Model file size
    model_size_mb = 0.0
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if checkpoint_path.exists():
        model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        
    # 3. Latency & FPS measurements
    from backend.config import USE_SECOND_ORDER_MOTION
    dummy_rgb = torch.randn(1, NUM_FRAMES, 3, 224, 224).to(device)
    if not USE_MOTION:
        landmark_dim = 225
    elif USE_SECOND_ORDER_MOTION:
        landmark_dim = 675  # 225 base + 225 first-order + 225 second-order
    else:
        landmark_dim = 450  # 225 base + 225 first-order
    dummy_landmarks = torch.randn(1, NUM_FRAMES, landmark_dim).to(device)
    dummy_mask = torch.ones(1, NUM_FRAMES, 3).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_rgb, dummy_landmarks, dummy_mask)
            
    # Measure
    start_time = time.time()
    runs = 100
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_rgb, dummy_landmarks, dummy_mask)
    elapsed = time.time() - start_time
    
    avg_latency_ms = (elapsed / runs) * 1000
    fps = 1000.0 / avg_latency_ms
    
    metrics = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_file_size_mb": float(round(model_size_mb, 2)),
        "inference_latency_ms": float(round(avg_latency_ms, 2)),
        "approximate_fps": float(round(fps, 2)),
        "execution_device": str(device)
    }
    
    print("\n" + "="*50)
    print("MODEL EFFICIENCY PROFILE")
    print("="*50)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Model Checkpoint Size: {model_size_mb:.2f} MB")
    print(f"Average Latency      : {avg_latency_ms:.2f} ms")
    print(f"Approximate FPS      : {fps:.2f} (Target: >25 FPS)")
    print(f"Device               : {device}")
    print("="*50)
    
    return metrics
