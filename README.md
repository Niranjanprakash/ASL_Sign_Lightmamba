# LightMamba-ASL: A Lightweight Hierarchical Multi-Scale Mamba Framework for Dynamic Video-Based ASL Recognition

LightMamba-ASL is a dynamic, video-based American Sign Language (ASL) word recognition framework designed for efficient deployment on resource-constrained devices. By combining spatial visual features from **MobileNetV3-Small** and geometric hand/body landmark coordinates from **MediaPipe Holistic**, the model captures complementary appearance and joint trajectory details. A **Hierarchical Multi-Scale Mamba (HMS-Mamba)** sequence modeling block is deployed to process dynamic temporal motions across three distinct resolutions (Fine, Intermediate, and Global scales).

---

## Key Academic Contributions
1. **Hierarchical Multi-Scale Mamba Sequence Modeling**: Downsamples feature representations across three levels (T=32, T=16, T=8) to model fine-grained hand/finger movements, coordinate hand trajectories, and recognize overall gesture evolution.
2. **Dynamic Reliability-Aware Multimodal Fusion**: Learns adaptive gating weights to prioritize either spatial RGB features or geometric landmarks based on motion blur or joint occlusions.
3. **Explicit Temporal Motion Features**: Captures coordinate displacement over time via first-order and optional second-order differences.
4. **Missing Landmark Validity Masks**: Distinguishes actual joint coordinate measurements from zeroed coordinates during MediaPipe tracking failures.

---

## Project Folder Structure

```text
LightMamba-ASL/
│
├── app.py                      # Flask API Server runner
├── requirements.txt            # Python dependencies
├── README.md                   # Project Documentation
├── .gitignore                  # Git untracked pattern configuration
│
├── dataset/                    # Local WLASL Dataset
│   ├── raw/
│   │   └── videos/             # Place raw .mp4 videos here
│   │
│   ├── metadata/
│   │   └── WLASL_v0.3.json     # Place official WLASL JSON here
│   │
│   └── processed/
│       ├── splits/             # Generated splits CSVs (train/val/test)
│       └── landmarks/          # Cached MediaPipe features (.pkl)
│
├── backend/
│   ├── config.py               # Central configuration (Hyperparameters, switches)
│   ├── routes.py               # Flask Route blueprints
│   │
│   ├── data/                   # Video sampling, split managers, Dataset loaders
│   ├── features/               # MediaPipe / MobileNetV3 extractors & normalizers
│   ├── models/                 # Neural Network blocks (HMS-Mamba, Fusion, branches)
│   ├── training/               # Losses, Metrics, Optimizers, Training loops
│   ├── evaluation/             # Test evaluation, Confusion matrix, profiles
│   ├── inference/              # Video prediction, confidence checks, webcam stream
│   └── services/               # Flask lazy model prediction services
│
├── checkpoints/
│   └── best_model.pth          # Saved best weights (automatically generated)
│
└── outputs/
    ├── plots/                  # Loss/Accuracy curves
    ├── confusion_matrix/       # Confusion matrix plots and reports
    └── metrics/                # Output evaluation logs
```

---

## Installation & Environment Setup

We recommend **Python 3.11** or **Python 3.10** for optimum compatibility with PyTorch and MediaPipe on Windows.

### 1. Create Virtual Environment
Open PowerShell inside the project directory:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

> [!NOTE]
> **Windows Mamba Compatibility**: Native `mamba-ssm` requires a complex Linux or WSL compilation setup. To ensure immediate functionality out-of-the-box on Windows, this project incorporates an optimized PyTorch-only **SSM Fallback Block** in [mamba_block.py](file:///c:/Users/NIRANJAN/second_review/backend/models/mamba_block.py) that mimics Mamba’s linear-time sequence scan. The real native Mamba integration path remains completely preserved if you deploy on Linux or a correctly configured WSL environment.

---

## Step-by-Step Execution Workflow

Follow this sequence to prepare, train, evaluate, and test the project.

### Step 1: Place Your Local Dataset
1. Place WLASL metadata file `WLASL_v0.3.json` inside [dataset/metadata/](file:///c:/Users/NIRANJAN/second_review/dataset/metadata/).
2. Place all the WLASL `.mp4` video files inside [dataset/raw/videos/](file:///c:/Users/NIRANJAN/second_review/dataset/raw/videos/).

### Step 2: Prepare the Dataset & Run Validation
Validates video integrity, splits data without leakage, and maps the initial 10 ASL classes:
```powershell
python -m backend.data.prepare_dataset
```

### Step 3: Run Model Training
Trains the LightMamba-ASL network using Two-Stage Transfer Learning and saves the best model:
```powershell
python -m backend.training.train
```

### Step 4: Run Evaluation & Model Profiling
Evaluates model on the unseen test set, generates confusion matrices, and profiles model parameters/FPS/latency:
```powershell
python -m backend.evaluation.evaluate
```

### Step 5: Test a Single Video File
Predicts the sign for a standalone MP4 video and displays top predictions:
```powershell
python -m backend.inference.predict_video --video "dataset/raw/videos/07069.mp4"
```

### Step 6: Run Real-time Webcam Recognition
Stream video from your HD webcam and recognize dynamic gestures with temporal prediction smoothing:
```powershell
python -m backend.inference.webcam
```

### Step 7: Launch Flask Backend Server
Launches backend REST APIs on port 5000:
```powershell
python app.py
```
* **Endpoints**:
  * `GET /` : Core API Status
  * `GET /api/classes` : List of the 10 configured classes
  * `GET /api/model/info` : Checkpoint and model dimensions details
  * `POST /api/predict/video` : Upload an MP4 video clip for inference
