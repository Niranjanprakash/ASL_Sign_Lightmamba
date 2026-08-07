import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

from backend.config import NUM_FRAMES, IMAGE_SIZE, USE_RGB, USE_LANDMARKS, USE_MOTION
from backend.data.video_loader import load_video
from backend.data.augmentations import VideoAugmentation

class ASLVideoDataset(Dataset):
    def __init__(self, csv_path: str, transform=None, is_training: bool = False, use_horizontal_flip: bool = False):
        self.df = pd.read_csv(csv_path)
        self.is_training = is_training
        
        # ImageNet normalization for MobileNetV3
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Augmentation pipeline
        self.augment = VideoAugmentation(use_horizontal_flip=use_horizontal_flip) if is_training else None
        
        # We will dynamically fetch features/landmarks from caching services during __getitem__
        # to ensure separation of concerns.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        video_id = row["video_id"]
        label_id = int(row["label_id"])
        
        # 1. Load video frames
        frames = load_video(video_path, target_frames=NUM_FRAMES, image_size=IMAGE_SIZE) # [T, H, W, C]
        
        # 2. Apply augmentation
        if self.augment:
            frames = self.augment(frames)
            
        # 3. Preprocess RGB frames to PyTorch Tensor shape [T, C, H, W]
        rgb_tensor = []
        for frame in frames:
            # normalize frame to [0, 1]
            f_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            f_tensor = self.normalize(f_tensor)
            rgb_tensor.append(f_tensor)
        rgb_tensor = torch.stack(rgb_tensor, dim=0) # [T, C, H, W]
        
        # 4. Extract landmarks & motion features (Phase 3/4 integration)
        # Import lazily to avoid circular dependency
        from backend.features.feature_cache import get_or_extract_features
        landmark_data = get_or_extract_features(video_id, frames)
        
        # landmark_data contains "landmarks", "mask", "motion"
        landmarks = torch.tensor(landmark_data["landmarks"], dtype=torch.float32)
        mask = torch.tensor(landmark_data["mask"], dtype=torch.float32)
        motion = torch.tensor(landmark_data["motion"], dtype=torch.float32)
        
        # Concatenate normalized coordinates and motion features if USE_MOTION is active
        # landmarks: [T, L_DIM]
        # motion: [T, M_DIM]
        if USE_MOTION:
            landmark_features = torch.cat([landmarks, motion], dim=-1)
        else:
            landmark_features = landmarks
            
        return {
            "video_id": video_id,
            "rgb": rgb_tensor,
            "landmarks": landmark_features,
            "mask": mask,
            "label": label_id
        }
