# LightMamba-ASL: Complete Technical Report

---

## 1. DATASET SELECTION

### 1.1 Why WLASL Was Selected

The **Word-Level American Sign Language (WLASL)** dataset was selected for this project because:

- It is the **largest publicly available word-level ASL video dataset** at the time of development
- It provides **real-world signer diversity** — multiple signers per word from different sources
- It includes **official train/val/test splits** with signer metadata enabling leakage-free evaluation
- It covers **2,000 ASL word classes** — sufficient for a meaningful recognition task
- It is **freely available** for academic research use
- The **JSON metadata format** provides rich per-instance information (signer_id, bbox, fps, frame range)

---

### 1.2 Available ASL Datasets Comparison

| Dataset | Classes | Videos | Type | Public | Format |
|---------|---------|--------|------|--------|--------|
| **WLASL v0.3** ✅ | 2,000 | 21,083 | Word-level | Yes | MP4 + JSON |
| MS-ASL | 1,000 | 25,513 | Word-level | Yes | MP4 |
| ASL Citizen | 2,731 | 83,399 | Word-level | Yes (Research) | MP4 |
| How2Sign | N/A | ~35hrs | Sentence-level | Yes | MP4 |
| AUTSL | 226 | 38,336 | Word-level | Yes | MP4 (Turkish SL) |
| ASL-LEX | 2,723 | Limited | Lexical | Yes | MP4 |

---

### 1.3 WLASL vs Other Datasets

| Criteria | WLASL | MS-ASL | ASL Citizen |
|----------|-------|--------|-------------|
| Word classes | 2,000 | 1,000 | 2,731 |
| Signer diversity | High | Medium | High |
| Metadata richness | High (JSON) | Medium | Medium |
| Code compatibility | Native ✅ | Needs rewrite | Needs rewrite |
| Dead links issue | ~43% missing | ~20% missing | Minimal |
| Research citations | 500+ | 200+ | 50+ (newer) |

---

### 1.4 Advantages and Disadvantages of WLASL

**Advantages:**
- Largest word-level ASL dataset with official splits
- Rich metadata: signer_id, bbox, fps, frame_start/end per instance
- Multiple sources (YouTube, ASLPro, SigningSavvy, Handspeak, etc.) → signer diversity
- Widely used in research → easy comparison with published results
- JSON format → easy programmatic access

**Disadvantages:**
- ~9,000 videos unavailable due to dead YouTube/source links (~43% missing)
- Highly imbalanced — top classes have 30+ videos, rare classes have 1-2
- No standardized body crop → signers appear at different scales
- Some classes have only 1 variation → poor generalization
- `thin` and `cool` classes not present in WLASL v0.3 → manually sourced

---

## 2. DATA PREPROCESSING PIPELINE

### 2.1 Complete Pipeline Overview

```
Raw MP4 Videos
      ↓
Step 1: Video Integrity Validation
      ↓
Step 2: Signer-Level Split Assignment
      ↓
Step 3: Frame Sampling (Uniform T=32)
      ↓
Step 4: RGB Preprocessing (Resize 224×224, Normalize)
      ↓
Step 5: MediaPipe Holistic Landmark Extraction
      ↓
Step 6: Landmark Normalization (Video-Level Scale)
      ↓
Step 7: Validity Mask Generation
      ↓
Step 8: Motion Feature Computation (1st + 2nd Order)
      ↓
Step 9: Feature Caching (.pkl)
      ↓
Clean Dataset Ready for Training
```

---

### 2.2 Step-by-Step Description

**Step 1 — Video Integrity Validation**
- Each MP4 file is opened with OpenCV
- Videos with 0 frames or unreadable files are rejected
- Only videos with matching video_id in WLASL_v0.3.json are accepted

**Step 2 — Signer-Level Split Assignment**
- WLASL JSON provides official train/val/test splits per instance
- **Signer-level consistency enforced**: if signer_id=5 appears in train, ALL their videos go to train
- Prevents identity leakage — model cannot memorize a specific signer's appearance
- MD5 hash-based fallback for videos without official split assignment

**Step 3 — Frame Sampling**
- Each video uniformly sampled to exactly **T=32 frames**
- Uniform sampling: `indices = linspace(0, total_frames-1, 32)`
- Handles both short videos (< 32 frames) and long videos (> 32 frames)
- Ensures fixed-length input regardless of original video duration

**Step 4 — RGB Preprocessing**
- Each frame resized to **224×224 pixels** (IMAGE_SIZE)
- Normalized with ImageNet statistics:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- Converted to float32 tensor [C, H, W]

**Step 5 — MediaPipe Holistic Landmark Extraction**
- MediaPipe Holistic model processes each frame
- Extracts 3 landmark groups:
  - Left hand: 21 keypoints × 3 (x, y, z) = 63 values
  - Right hand: 21 keypoints × 3 = 63 values
  - Pose (upper body): 33 keypoints × 3 = 99 values → filtered to 11 upper-body joints = 33 values
- Total: 75 keypoints × 3 = **225 raw landmark values per frame**

**Step 6 — Landmark Normalization (Video-Level)**
- **Hands**: centered on wrist (joint 0); scale = median of max joint distances across all valid frames
- **Pose**: centered on shoulder midpoint; scale = median shoulder distance across valid frames
- Video-level median scale prevents per-frame scale noise
- NaN/Inf values replaced with 0.0

**Step 7 — Validity Mask Generation**
- Binary mask [T, 3] for [left_hand, right_hand, pose]
- mask[t, 0] = 1.0 if left hand detected at frame t, else 0.0
- Distinguishes actual zero coordinates from MediaPipe tracking failures
- Used in fusion layer to weight modality reliability

**Step 8 — Motion Feature Computation**
- **First-order motion**: `Δlandmark[t] = landmark[t] - landmark[t-1]`
- **Second-order motion**: `Δ²landmark[t] = Δlandmark[t] - Δlandmark[t-1]`
- **Mask-aware**: motion set to zero if either frame t or t-1 has no valid detection
- Prevents fake motion spikes from MediaPipe detection flicker
- Final landmark feature dim: 225 (raw) + 225 (1st order) + 225 (2nd order) = **675 per frame**

**Step 9 — Feature Caching**
- Extracted features saved as `.pkl` files in `dataset/processed/landmarks/`
- Cache key: `{sanitized_video_id}_f32_s224.pkl`
- Avoids re-running MediaPipe on every training epoch
- Cache invalidated if config (NUM_FRAMES or IMAGE_SIZE) changes

---

### 2.3 Data Augmentation (Training Only)

| Augmentation | Description |
|-------------|-------------|
| Horizontal Flip | Mirror frames + swap left↔right hand landmarks semantically |
| Speed Jitter | Random frame subsampling to simulate faster/slower signing |
| Applied consistently | Same augmentation params applied to both RGB and landmarks |

---

### 2.4 Final Clean Dataset Statistics

| Split | Videos | Classes |
|-------|--------|---------|
| Train | 103 | 10 |
| Validation | 26 | 10 |
| Test | 18 | 10 |
| **Total** | **147** | **10** |

---

## 3. MODEL DESCRIPTION

### 3.1 Why LightMamba-ASL Was Selected

The model was designed specifically for **resource-constrained deployment** with these goals:
- Lightweight enough to run on CPU/mobile devices
- Capture both spatial appearance (RGB) and geometric structure (landmarks)
- Model temporal dynamics at multiple time scales
- Handle missing/occluded landmarks gracefully

No existing model satisfied all four requirements simultaneously — hence a custom architecture was designed.

---

### 3.2 Model Algorithm

```
INPUT: Video V = {f₁, f₂, ..., f₃₂} (32 RGB frames)
       Landmarks L = {l₁, l₂, ..., l₃₂} (75 keypoints × 3 per frame)
       Mask M = {m₁, m₂, ..., m₃₂} (validity per frame)

STEP 1 — RGB Branch (MobileNetV3-Small):
  For each frame fₜ:
    rgb_feat[t] = MobileNetV3(fₜ) → [576-dim vector]
  Output: RGB_sequence [B, 32, 576]

STEP 2 — Landmark Branch (Linear Projection):
  landmark_input = concat(raw_landmarks, motion_1st, motion_2nd) → [B, 32, 675]
  land_emb[t] = Linear(675 → 256) + LayerNorm + GELU
  Output: Landmark_sequence [B, 32, 256]

STEP 3 — Dynamic Reliability-Aware Fusion:
  gate = Sigmoid(Linear(concat(rgb_feat, land_emb, mask)))
  fused[t] = gate × rgb_projected[t] + (1-gate) × land_emb[t]
  Output: Fused_sequence [B, 32, 256]

STEP 4 — HMS-Mamba (3-Scale Temporal Modeling):
  Scale 1 (Fine, T=32):
    fine_feats = MambaBlock(fused) → [B, 32, 256]
    fine_rep = AvgPool(fine_feats) → [B, 256]

  Downsample T=32 → T=16:
    x_down1 = Conv1d(stride=2)(fine_feats) → [B, 16, 256]

  Scale 2 (Intermediate, T=16):
    inter_feats = MambaBlock(x_down1) → [B, 16, 256]
    inter_rep = AvgPool(inter_feats) → [B, 256]

  Downsample T=16 → T=8:
    x_down2 = Conv1d(stride=2)(inter_feats) → [B, 8, 256]

  Scale 3 (Global, T=8):
    global_feats = MambaBlock(x_down2) → [B, 8, 256]
    global_rep = AvgPool(global_feats) → [B, 256]

  Multi-Scale Fusion:
    combined = concat(fine_rep, inter_rep, global_rep) → [B, 768]
    temporal_rep = Linear(768→256) + LayerNorm + GELU → [B, 256]

STEP 5 — Classifier:
  logits = LayerNorm → Dropout(0.5) → Linear(256→10) → [B, 10]

OUTPUT: Class probabilities via Softmax
```

---

### 3.3 Working Principle

**MambaBlock (SSM — State Space Model):**
- Processes sequences in linear time O(T) vs Transformer's O(T²)
- Maintains a hidden state that selectively remembers relevant temporal information
- On Windows: PyTorch fallback implementation used (native mamba_ssm requires Linux CUDA)

**Hierarchical Multi-Scale Design:**
- Fine scale (T=32): captures finger micro-movements and hand shape details
- Intermediate scale (T=16): captures hand trajectory and transition patterns
- Global scale (T=8): captures overall gesture evolution and body movement

**Dynamic Reliability Fusion:**
- When MediaPipe fails to detect hands (occlusion, blur), mask=0
- Gate automatically shifts weight toward RGB features
- When landmarks are clean, gate shifts toward geometric features

---

### 3.4 Model Comparison

| Model | Params | Approach | Temporal | Landmark | Lightweight |
|-------|--------|----------|----------|----------|-------------|
| **LightMamba-ASL** | **6.49M** | Dual-stream | HMS-Mamba | Yes + Motion | Yes ✅ |
| I3D | 28M | RGB only | 3D Conv | No | No |
| SlowFast | 34M | RGB only | Dual-path | No | No |
| MediaPipe + LSTM | ~2M | Landmark only | LSTM | Yes | Yes |
| ST-GCN | 3.1M | Skeleton only | GCN | Yes | Yes |
| SMKD (SOTA) | 86M | RGB + Distill | Transformer | No | No |

---

### 3.5 Advantages and Disadvantages

**Advantages:**
- Lightweight (6.49M params) — deployable on CPU/mobile
- Dual-stream captures both appearance and geometry
- HMS-Mamba models temporal dynamics at 3 scales simultaneously
- Reliability-aware fusion handles occlusion/blur gracefully
- Missing landmark mask prevents false signal propagation
- Two-stage transfer learning stabilizes training

**Disadvantages:**
- Native Mamba SSM requires Linux + CUDA (Windows uses PyTorch fallback)
- Small dataset limits generalization (10 classes, ~10 videos/class)
- CPU inference: 474ms latency (not real-time without GPU)
- No attention visualization for interpretability
- Second-order motion adds complexity without guaranteed benefit on small data

---

## 4. TRAINING PROCESS

### 4.1 Two-Stage Transfer Learning Strategy

**Stage 1 (Epochs 1–19): Frozen Backbone**
- MobileNetV3-Small backbone weights frozen (pretrained on ImageNet)
- Only landmark branch, fusion, HMS-Mamba, and classifier trained
- Learning rate: 1e-4
- Goal: Learn temporal and landmark representations without disturbing pretrained features

**Stage 2 (Epoch 20+): Fine-tuning**
- MobileNetV3 final blocks unfrozen
- Learning rate reduced to 1e-5 (10× lower)
- Full end-to-end fine-tuning
- Goal: Adapt visual features to ASL-specific appearance

---

### 4.2 Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Batch Size | 8 |
| Max Epochs | 120 |
| Learning Rate (Stage 1) | 1e-4 |
| Learning Rate (Stage 2) | 1e-5 |
| Weight Decay | 0.05 |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau |
| Early Stopping Patience | 10 epochs |
| Gradient Clip | max_norm=1.0 |
| Random Seed | 42 |
| Loss Function | Weighted Cross-Entropy |

---

### 4.3 Training Results (Current Run)

| Metric | Value |
|--------|-------|
| Total Epochs Run | 27 (early stopped) |
| Best Train Accuracy | 38.83% |
| Best Val Accuracy | 23.08% |
| Early Stop Reason | Val loss no improvement for 10 epochs |

**Before Bug Fixes (Initial Run):**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Top-1 Test Accuracy | 7.69% | 16.67% | +8.98% |
| Top-5 Test Accuracy | 53.85% | 88.89% | +35.04% |

**Bug fixes that caused improvement:**
1. RGB/Landmark sync fix in dataset.py
2. Augmentation parameter return fix
3. Mask-aware motion computation
4. Video-level landmark normalization
5. Feature cache path sanitization
6. Signer-level split leakage fix
7. Shared webcam pipeline

---

## 5. EVALUATION METRICS

### 5.1 Available Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Top-1 Accuracy** | Correct / Total | Exact match rate |
| **Top-5 Accuracy** | Top-5 contains true label / Total | Whether true class is in top 5 predictions |
| **Macro F1-Score** | Mean F1 across all classes (equal weight) | Per-class balance |
| **Weighted F1-Score** | F1 weighted by class support | Overall F1 accounting for class size |
| **Per-Class Accuracy** | Correct[c] / Total[c] per class | Individual class performance |
| **Confusion Matrix** | [N×N] predicted vs true | Misclassification patterns |
| **Inference Latency** | ms per video | Real-time feasibility |
| **FPS** | 1000 / latency_ms | Frames processable per second |
| **Parameter Count** | Total trainable weights | Model complexity |

---

### 5.2 How Each Metric is Evaluated

**Top-1 Accuracy:**
```python
pred = argmax(softmax(logits))
correct = (pred == true_label)
top1_acc = sum(correct) / total_samples
```

**Top-5 Accuracy:**
```python
top5_preds = argsort(logits, descending=True)[:5]
correct = true_label in top5_preds
top5_acc = sum(correct) / total_samples
```

**Macro F1:**
```python
F1[c] = 2 × Precision[c] × Recall[c] / (Precision[c] + Recall[c])
Macro_F1 = mean(F1[c] for all classes c)
```

**Confidence Calibration:**
```python
confidence = max(softmax(logits))
if confidence < THRESHOLD (0.25):
    prediction = "UNCERTAIN"
```

---

### 5.3 Current Evaluation Results

| Metric | Value |
|--------|-------|
| Top-1 Test Accuracy | 16.67% |
| Top-5 Test Accuracy | **88.89%** |
| Macro F1-Score | 8.57% |
| Weighted F1-Score | 7.94% |
| Total Parameters | 6,494,551 |
| Checkpoint Size | 55.17 MB |
| CPU Latency | 474.52 ms |
| Approximate FPS | 2.11 |

**Per-Class Results:**

| Class | Test Accuracy | Test Videos |
|-------|--------------|-------------|
| before | 0% | 3 |
| thin | 0% | 2 |
| cool | 0% | 2 |
| drink | 0% | 1 |
| go | 0% | 1 |
| computer | 0% | 2 |
| **who** | **100%** | 1 |
| cousin | 0% | 2 |
| **help** | **100%** | 2 |
| candy | 0% | 3 |

**Top Confused Pairs:**

| Pair | Confusions |
|------|-----------|
| cool → thin | 2 |
| cousin → drink | 2 |
| candy → who | 2 |
| before → who | 1 |
| before → cousin | 1 |

---

## 6. COMPLETE ARCHITECTURE

### 6.1 Architecture Diagram (Text)

```
INPUT VIDEO (MP4)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER                     │
│  Frame Sampling (T=32) │ Resize 224×224 │ Normalize      │
└─────────────────────────────────────────────────────────┘
        │                           │
        ▼                           ▼
┌──────────────────┐    ┌──────────────────────────────────┐
│   RGB BRANCH     │    │        LANDMARK BRANCH            │
│                  │    │                                   │
│ MobileNetV3-Small│    │  MediaPipe Holistic               │
│ (pretrained)     │    │  75 keypoints × 3 = 225 raw       │
│                  │    │  + 225 first-order motion         │
│ Per-frame:       │    │  + 225 second-order motion        │
│ [224,224,3]      │    │  = 675 dim per frame              │
│     ↓            │    │         ↓                         │
│ [576-dim feat]   │    │  Linear(675→256) + LN + GELU      │
│                  │    │  [256-dim embedding]              │
│ Output:          │    │  Output:                          │
│ [B, 32, 576]     │    │  [B, 32, 256]                     │
└──────────────────┘    └──────────────────────────────────┘
        │                           │
        └───────────┬───────────────┘
                    ▼
┌─────────────────────────────────────────────────────────┐
│         DYNAMIC RELIABILITY-AWARE FUSION                 │
│                                                         │
│  Validity Mask [B, 32, 3] → gate weights                │
│  gate = Sigmoid(Linear(concat(rgb, land, mask)))        │
│  fused = gate × rgb_proj + (1-gate) × land_emb         │
│                                                         │
│  Output: [B, 32, 256]                                   │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              HMS-MAMBA (3-SCALE TEMPORAL)                │
│                                                         │
│  Scale 1 — Fine (T=32):                                 │
│    MambaBlock → [B, 32, 256] → AvgPool → [B, 256]       │
│         │                                               │
│    Conv1d(stride=2) ↓                                   │
│                                                         │
│  Scale 2 — Intermediate (T=16):                         │
│    MambaBlock → [B, 16, 256] → AvgPool → [B, 256]       │
│         │                                               │
│    Conv1d(stride=2) ↓                                   │
│                                                         │
│  Scale 3 — Global (T=8):                                │
│    MambaBlock → [B, 8, 256] → AvgPool → [B, 256]        │
│                                                         │
│  Concat: [B, 768] → Linear(768→256) + LN + GELU        │
│  Output: [B, 256]                                       │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│                  CLASSIFIER HEAD                         │
│  LayerNorm → Dropout(0.5) → Linear(256→10)              │
│  Output: [B, 10] logits                                 │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              CONFIDENCE CALIBRATION                      │
│  Softmax → max confidence                               │
│  if confidence < 0.25 → "UNCERTAIN"                     │
│  else → CLASSES[argmax]                                 │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
            PREDICTED ASL SIGN
```

---

### 6.2 Output at Each Stage (Based on Current Workflow)

| Stage | Input Shape | Output Shape | Output Value Example |
|-------|------------|--------------|---------------------|
| Frame Sampling | Variable MP4 | [32, 224, 224, 3] | 32 uniform frames |
| RGB Branch | [B, 32, 3, 224, 224] | [B, 32, 576] | MobileNetV3 features |
| Landmark Extraction | [32, 224, 224, 3] | [32, 75, 3] | Normalized keypoints |
| Motion Features | [32, 75, 3] | [32, 225×2] | Velocity + acceleration |
| Landmark Branch | [B, 32, 675] | [B, 32, 256] | Projected embeddings |
| Fusion | [B, 32, 576+256+3] | [B, 32, 256] | Gated fused features |
| HMS Fine Scale | [B, 32, 256] | [B, 256] | Fine temporal rep |
| HMS Intermediate | [B, 16, 256] | [B, 256] | Mid temporal rep |
| HMS Global | [B, 8, 256] | [B, 256] | Global temporal rep |
| Multi-Scale Concat | [B, 768] | [B, 256] | Video representation |
| Classifier | [B, 256] | [B, 10] | Class logits |
| Softmax | [B, 10] | [B, 10] | Probabilities (sum=1) |
| Final Output | [B, 10] | String | e.g., "help" @ 37.31% |

---

*Report generated for LightMamba-ASL project — 10 classes, WLASL v0.3 dataset, CPU inference on Windows.*
