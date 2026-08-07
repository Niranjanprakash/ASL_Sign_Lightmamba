import sys
import torch
from pathlib import Path

# Add workspace root to system path
sys.path.append(str(Path(__file__).resolve().parent))

from backend.models.lightmamba_asl import LightMambaASL
from backend.config import NUM_FRAMES, IMAGE_SIZE, NUM_CLASSES, USE_MOTION, USE_SECOND_ORDER_MOTION

def main():
    print("="*60)
    print("LIGHTMAMBA-ASL PIPELINE SANITY CHECK")
    print("="*60)
    
    # 1. Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 2. Setup inputs
    batch_size = 4
    T = NUM_FRAMES
    C = 3
    H = IMAGE_SIZE
    W = IMAGE_SIZE
    
    # Landmark features dimension calculation
    base_dim = 225
    if USE_MOTION:
        if USE_SECOND_ORDER_MOTION:
            landmark_dim = base_dim * 3
        else:
            landmark_dim = base_dim * 2
    else:
        landmark_dim = base_dim
        
    print(f"Input dimensions:")
    print(f"  RGB batch: [{batch_size}, {T}, {C}, {H}, {W}]")
    print(f"  Landmarks batch: [{batch_size}, {T}, {landmark_dim}]")
    print(f"  Mask batch: [{batch_size}, {T}, 3]")
    
    rgb_dummy = torch.randn(batch_size, T, C, H, W).to(device)
    landmarks_dummy = torch.randn(batch_size, T, landmark_dim).to(device)
    mask_dummy = torch.ones(batch_size, T, 3).to(device)
    
    # 3. Model loading
    print("\nLoading LightMambaASL model...")
    try:
        model = LightMambaASL().to(device)
        model.eval()
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        sys.exit(1)
        
    # 4. Forward pass
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            output = model(rgb_dummy, landmarks_dummy, mask_dummy)
        print("Forward pass successful!")
        print(f"Output shape: {output.shape} (Expected: [{batch_size}, {NUM_CLASSES}])")
        print(f"Sample prediction probabilities (softmax):\n{torch.softmax(output, dim=-1)}")
        print("\n" + "="*60)
        print("SANITY CHECK PASSED SUCCESSFULLY!")
        print("="*60)
    except Exception as e:
        print(f"Forward pass failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
