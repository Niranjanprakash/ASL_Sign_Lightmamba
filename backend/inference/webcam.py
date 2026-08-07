import cv2
import torch
import numpy as np
import time
from collections import deque
from torchvision import transforms

from backend.config import CHECKPOINT_DIR, CLASSES, NUM_FRAMES, IMAGE_SIZE, USE_MOTION
from backend.utils import get_device, logger
from backend.features.landmark_extractor import LandmarkExtractor
from backend.features.landmark_normalizer import normalize_landmarks
from backend.features.motion_features import compute_motion_features
from backend.models.lightmamba_asl import LightMambaASL
from backend.inference.temporal_smoothing import TemporalSmoothing
from backend.inference.confidence import calibrate_prediction

def main():
    device = get_device()
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return

    # Load Model
    print("[WEBCAM] Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LightMambaASL(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load media extractors
    landmark_extractor = LandmarkExtractor()
    smoothing = TemporalSmoothing(window_size=5)
    
    # Image normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Rolling frame buffer
    frame_buffer = deque(maxlen=NUM_FRAMES)
    
    # Open camera stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[WEBCAM] Starting webcam feed. Press 'q' to quit.")
    
    last_inference_time = 0
    fps = 0.0
    latency_ms = 0.0
    pred_label = "Waiting..."
    confidence = 0.0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for natural webcam mirroring
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Prepare frame for model input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Add to rolling buffer
        frame_buffer.append(frame_resized)
        
        # Only run inference when buffer is full
        if len(frame_buffer) == NUM_FRAMES:
            inf_start = time.time()
            
            # 1. Prepare RGB sequence tensor
            rgb_tensors = []
            for f in frame_buffer:
                f_t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                f_t = normalize(f_t)
                rgb_tensors.append(f_t)
            rgb_tensor = torch.stack(rgb_tensors, dim=0).unsqueeze(0).to(device) # [1, 32, 3, 224, 224]

            # 2. Extract Landmarks
            frames_np = np.stack(list(frame_buffer), axis=0)
            raw_landmarks, mask = landmark_extractor.extract_video_sequence(frames_np)
            
            normalized = normalize_landmarks(raw_landmarks, mask)
            motion = compute_motion_features(normalized)
            
            # Reshape & tensorize landmarks
            T, N, D = normalized.shape
            flat_landmarks = normalized.reshape(T, N * D)
            
            landmarks_t = torch.tensor(flat_landmarks, dtype=torch.float32)
            motion_t = torch.tensor(motion, dtype=torch.float32)
            mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
            
            if USE_MOTION:
                landmark_features = torch.cat([landmarks_t, motion_t], dim=-1)
            else:
                landmark_features = landmarks_t
            landmark_features = landmark_features.unsqueeze(0).to(device) # [1, 32, L_DIM]
            
            # 3. Forward Pass
            with torch.no_grad():
                logits = model(rgb_tensor, landmark_features, mask_t)
                
            # Softmax
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            # 4. Temporal Smoothing
            smoothed_probs = smoothing.update(probs)
            
            # 5. Calibrate Prediction
            pred_idx, confidence, top_k = calibrate_prediction(torch.tensor(smoothed_probs).unsqueeze(0))
            
            if pred_idx == -1:
                pred_label = "UNCERTAIN"
            else:
                pred_label = CLASSES[pred_idx]
                
            latency_ms = (time.time() - inf_start) * 1000

        # Calculate Display FPS
        fps = 1.0 / (time.time() - start_time)

        # Draw Overlay UI
        cv2.rectangle(frame, (10, 10), (320, 140), (0, 0, 0), -1)
        cv2.putText(frame, f"Prediction: {pred_label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Latency: {latency_ms:.1f} ms", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("LightMamba-ASL Webcam Stream", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmark_extractor.close()
    print("[WEBCAM] Stream terminated.")

if __name__ == "__main__":
    main()
