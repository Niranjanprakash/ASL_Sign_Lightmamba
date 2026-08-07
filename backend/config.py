import os
from pathlib import Path

# Paths Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
VIDEO_DIR = DATASET_ROOT / "raw" / "videos"
METADATA_DIR = DATASET_ROOT / "metadata"
PROCESSED_DIR = DATASET_ROOT / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOAD_DIR = PROJECT_ROOT / "uploads"

# Ensure all critical folders exist
for folder in [DATASET_ROOT, VIDEO_DIR, METADATA_DIR, PROCESSED_DIR, CHECKPOINT_DIR, OUTPUT_DIR, UPLOAD_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Centralized Classes Configuration (Phase 1: 10 ASL Words)
CLASSES = [
    "book",
    "drink",
    "computer",
    "before",
    "chair",
    "go",
    "clothes",
    "who",
    "candy",
    "cousin"
]
NUM_CLASSES = len(CLASSES)

# Preprocessing Hyperparameters
NUM_FRAMES = 32
IMAGE_SIZE = 224

# Training Hyperparameters
BATCH_SIZE = 8
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
RANDOM_SEED = 42

# Model Dimensions
RGB_FEATURE_DIM = 576  # MobileNetV3-Small feature dimension (last channel before classifier)
LANDMARK_EMBED_DIM = 256
FUSION_DIM = 256
MAMBA_HIDDEN_DIM = 256
DROPOUT = 0.3

# Ablation & Novelty Configuration Switches
USE_RGB = True
USE_LANDMARKS = True
USE_MOTION = True
USE_SECOND_ORDER_MOTION = False
USE_RELIABILITY_FUSION = False
USE_HORIZONTAL_FLIP = False

# Inference Config
CONFIDENCE_THRESHOLD = 0.70
TEMPORAL_SMOOTHING_WINDOW = 5
