import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from backend.config import NUM_FRAMES, IMAGE_SIZE, USE_MOTION
from backend.data.video_loader import load_video
from backend.data.augmentations import VideoAugmentation, apply_landmark_augmentation
from backend.features.feature_cache import get_or_extract_features
from backend.features.motion_features import compute_motion_features

class ASLVideoDataset(Dataset):
    def __init__(self, csv_path: str, transform=None, is_training: bool = False, use_horizontal_flip: bool = False):
        self.df = pd.read_csv(csv_path)
        self.is_training = is_training
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.augment = VideoAugmentation(use_horizontal_flip=use_horizontal_flip) if is_training else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        video_id = str(row["video_id"])
        label_id = int(row["label_id"])

        # 1. Load frames once
        frames = load_video(video_path, target_frames=NUM_FRAMES, image_size=IMAGE_SIZE)

        # 2. Get cached landmarks (from clean original frames)
        landmark_data = get_or_extract_features(video_id, frames)
        # Work on copies so cache is never mutated
        raw_landmarks = np.array(landmark_data["landmarks"], dtype=np.float32)  # [T, 225]
        raw_mask      = np.array(landmark_data["mask"],      dtype=np.float32)  # [T, 3]

        # 3. Apply augmentation — RGB + synced landmarks
        aug_params = None
        if self.augment:
            frames, aug_params = self.augment(frames)
            # Reshape to [T, 75, 3] for landmark augmentation
            lm_3d = raw_landmarks.reshape(-1, 75, 3)
            lm_3d, raw_mask = apply_landmark_augmentation(lm_3d, raw_mask, aug_params)
            raw_landmarks = lm_3d.reshape(-1, 225)

        # 4. Recompute motion on (possibly augmented) landmarks
        if USE_MOTION:
            lm_3d_for_motion = raw_landmarks.reshape(-1, 75, 3)
            motion = compute_motion_features(lm_3d_for_motion, raw_mask)  # mask-aware
            landmark_features = torch.tensor(
                np.concatenate([raw_landmarks, motion], axis=-1), dtype=torch.float32
            )
        else:
            landmark_features = torch.tensor(raw_landmarks, dtype=torch.float32)

        # 5. RGB tensor [T, C, H, W]
        rgb_tensor = []
        for frame in frames:
            f_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            f_t = self.normalize(f_t)
            rgb_tensor.append(f_t)
        rgb_tensor = torch.stack(rgb_tensor, dim=0)

        return {
            "video_id": video_id,
            "rgb": rgb_tensor,
            "landmarks": landmark_features,
            "mask": torch.tensor(raw_mask, dtype=torch.float32),
            "label": label_id
        }
