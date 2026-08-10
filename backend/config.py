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

# Active classes (10 — matches current checkpoint)
# Switch to 104 classes after Colab retraining
CLASSES = [
    "before", "thin", "cool", "drink", "go",
    "computer", "who", "cousin", "help", "candy"
]

# Top 104 classes (use after Colab training completes)
# CLASSES = [
#     "before", "thin", "cool", "drink", "go", "computer", "who", "cousin", "help", "candy",
#     "thanksgiving", "bed", "bowling", "tall", "accident", "short", "trade", "yes", "what", "later",
#     "man", "shirt", "change", "corn", "dark", "last", "pizza", "basketball", "call", "cold",
#     "deaf", "no", "walk", "mother", "woman", "dog", "family", "apple", "play", "letter",
#     "thursday", "bar", "brother", "check", "laugh", "room", "take", "why", "example", "far",
#     "leave", "soon", "champion", "delay", "environment", "score", "year", "black", "hot", "like",
#     "many", "orange", "fish", "graduate", "language", "study", "white", "bird", "doctor", "give",
#     "secretary", "work", "cheat", "full", "son", "tell", "wait", "cry", "snow", "write",
#     "yesterday", "bad", "blanket", "daughter", "good", "balance", "because", "convince", "fat", "government",
#     "interest", "order", "sandwich", "theory", "argue", "delicious", "move", "perspective", "silly", "sweet",
#     "appointment", "ready", "speech", "toast"
# ]
NUM_CLASSES = len(CLASSES)

# Preprocessing Hyperparameters
NUM_FRAMES = 32
IMAGE_SIZE = 224

# Training Hyperparameters
BATCH_SIZE = 8
MAX_EPOCHS = 120
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
RANDOM_SEED = 42

# Model Dimensions
RGB_FEATURE_DIM = 576  # MobileNetV3-Small feature dimension (last channel before classifier)
LANDMARK_EMBED_DIM = 256
FUSION_DIM = 256
MAMBA_HIDDEN_DIM = 256
DROPOUT = 0.5

# Ablation & Novelty Configuration Switches
USE_RGB = True
USE_LANDMARKS = True
USE_MOTION = True
USE_SECOND_ORDER_MOTION = True
USE_RELIABILITY_FUSION = True
USE_HORIZONTAL_FLIP = True

# Inference Config
CONFIDENCE_THRESHOLD = 0.25
TEMPORAL_SMOOTHING_WINDOW = 5
