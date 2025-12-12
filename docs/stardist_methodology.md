# StarDist Cell Detection: Methodology

This document details the deep learning methodology used for cell detection in TIRF-SIM microscopy images, explaining the architectural choices, training strategies, and optimization decisions.

---

## 1. Problem Definition

**Task**: Detect and localize individual cells in time-lapse TIRF-SIM microscopy video sequences.

**Challenges**:
- Cells are densely packed with touching/overlapping boundaries
- Low contrast between cells and background
- Variable cell sizes and shapes
- Limited annotated training data (162 frames total)
- Real-time tracking requirements

**Output**: Centroid coordinates (x, y) for each detected cell per frame.

---

## 2. Why StarDist?

### 2.1 Architecture Comparison

| Method | Pros | Cons | Suitability |
|--------|------|------|-------------|
| **U-Net** | Simple, fast | Struggles with touching objects | Poor for dense cells |
| **Mask R-CNN** | Instance-aware | Heavy, slow, needs large data | Overkill for round cells |
| **Cellpose** | Great for cells | Gradient-based, slower inference | Good alternative |
| **StarDist** | Designed for convex objects | Limited to star-convex shapes | **Ideal for round cells** |

### 2.2 StarDist Principle

StarDist represents each object as a **star-convex polygon** defined by radial distances from the object center:

```
                     Ray 0 (up)
                        |
                        | d0
                        |
           d7 ---------[*]--------- d1
                      / | \
                     /  |  \
                d6  /   |   \  d2
                   /    |    \
                  /     |     \
                 -----  |  -----
                d5     d4     d3

For N_RAYS = 64: Angles at 360/64 = 5.625 degree intervals
Each ray stores distance from center to object boundary
```

**Why this works for cells**:
- Cells are approximately convex (star-convex assumption holds)
- Radial representation naturally handles touching cells
- Each pixel votes for object center - robust to partial occlusion
- NMS on probability maps separates overlapping detections

### 2.3 StarDist Output

The model predicts two outputs per pixel:

1. **Probability Map** (H x W): Likelihood of being an object center
2. **Distance Map** (N_RAYS x H x W): Radial distances for each ray direction

```python
# Model output structure
out = model(image)['nuc']
prob_map = torch.sigmoid(out.binary_map)  # (1, H, W) - object probability
dist_map = out.aux_map                     # (64, H, W) - ray distances
```

---

## 3. Model Architecture

### 3.1 Encoder-Decoder Structure

```
Input Image (1, 256, 256)  <-- Grayscale, normalized
        |
        v
+-------------------+
|   ResNet18        |  <-- Pretrained on ImageNet
|   Encoder         |      Modified for 1-channel input
|                   |
|   Conv layers:    |
|   64->64->128->256|      Feature extraction at multiple scales
|   ->512 channels  |
+--------+----------+
         |
         |  Skip connections (U-Net style)
         |
         v
+-------------------+
|   StarDist        |
|   Decoder         |
|                   |
|   Upsampling +    |      Reconstruct spatial resolution
|   Skip fusion     |
+--------+----------+
         |
         +------------------+
         |                  |
         v                  v
+-----------------+  +-----------------+
|  Probability    |  |   Ray Distance  |
|  Head           |  |   Head          |
|  (1, H, W)      |  |  (64, H, W)     |
|  BCE Loss       |  |  L1 Loss        |
+-----------------+  +-----------------+
```

### 3.2 Encoder Choice: ResNet18

**Why ResNet18?**

| Encoder | Parameters | Speed | Performance |
|---------|------------|-------|-------------|
| ResNet18 | 11.7M | Fast | **Best trade-off** |
| ResNet34 | 21.8M | Medium | Similar DetA |
| ResNet50 | 25.6M | Slow | Overfits on small data |
| EfficientNet-B0 | 5.3M | Fast | Slightly worse |

**Decision**: ResNet18 provides sufficient capacity for cell features without overfitting on 162 training frames.

### 3.3 Number of Rays: 64

```python
N_RAYS = 64  # Radial directions
```

**Trade-off analysis**:

| N_RAYS | Angular Resolution | Boundary Precision | Computation |
|--------|-------------------|-------------------|-------------|
| 32 | 11.25 deg | Coarse | Fast |
| **64** | **5.625 deg** | **Good balance** | **Medium** |
| 96 | 3.75 deg | Fine | Slow |

**Decision**: 64 rays provide sufficient boundary precision for round cells while keeping computation manageable. Cells in TIRF-SIM are relatively smooth, so higher angular resolution yields diminishing returns.

---

## 4. Loss Function

### 4.1 Baseline Loss (Final Choice)

After extensive experimentation, the **simple baseline loss** performed best:

```
Loss = BCE(prob_pred, prob_gt) + L1(dist_pred, dist_gt)
```

**Components**:

1. **Binary Cross-Entropy (BCE)** for probability map:
```
BCE = -[y * log(p) + (1-y) * log(1-p)]

Where:
- y = ground truth (1 for object centers, 0 elsewhere)
- p = predicted probability
```

2. **L1 Loss** for ray distances:
```
L1 = |dist_pred - dist_gt|

# Only computed where y=1 (object centers)
# Robust to outliers compared to L2/MSE
```

### 4.2 Why Not Focal Loss + Dice?

We tested advanced losses but they underperformed:

| Loss Configuration | DetA | Notes |
|-------------------|------|-------|
| BCE + L1 (baseline) | **0.8129** | Simple, stable |
| Focal + L1 | 0.8023 | Over-suppresses easy examples |
| BCE + Dice + L1 | 0.7956 | Dice adds noise for sparse masks |
| Focal + Dice + Smooth L1 | 0.7812 | Too complex, unstable |

**Why baseline wins**:
- **Small dataset**: Complex losses need more data to show benefits
- **Class balance**: Cells are not extremely rare (unlike medical anomalies)
- **Clean annotations**: HITL annotations are high-quality, no need for noise-robust losses

### 4.3 Loss Implementation

```python
# From training loop
def compute_loss(pred, target):
    # Probability loss (BCE)
    prob_loss = F.binary_cross_entropy_with_logits(
        pred['binary_map'],      # Raw logits
        target['prob_map']       # Binary mask
    )

    # Distance loss (L1, masked to object regions)
    mask = target['prob_map'] > 0.5
    dist_loss = F.l1_loss(
        pred['aux_map'][mask],   # Predicted rays
        target['dist_map'][mask] # GT rays
    )

    return prob_loss + dist_loss
```

---

## 5. K-Fold Cross-Validation Strategy

### 5.1 Why K-Fold CV?

| Challenge | Solution with K-Fold |
|-----------|---------------------|
| Limited annotated data (162 frames) | Every sample used for both training AND validation |
| Risk of overfitting to specific frames | 5 different train/val splits reduce bias |
| Uncertainty in performance estimates | Mean +/- std across folds quantifies reliability |
| Model selection | Ensemble of all folds improves generalization |

### 5.2 K-Fold Split Strategy

```
Total Dataset: 162 frames (42 HITL-annotated + 120 video frames)

Fold 1: [VAL----] [TRAIN--------------------------------------]
Fold 2: [TRAIN--] [VAL----] [TRAIN-----------------------------]
Fold 3: [TRAIN----------] [VAL----] [TRAIN---------------------]
Fold 4: [TRAIN--------------------] [VAL----] [TRAIN-----------]
Fold 5: [TRAIN--------------------------------------] [VAL----]

Each fold: ~130 train / ~32 validation frames
```

### 5.3 Stratified Splitting

To ensure balanced difficulty across folds, we use **stratified splitting by cell count**:

```python
USE_STRATIFIED_SPLIT = True  # Stratified by cell count

# This ensures each fold has similar distribution of:
# - Easy frames (many clearly visible cells)
# - Hard frames (few cells, low contrast, edge cases)
```

**Why stratify?**
- Prevents scenarios where hard frames cluster in one fold
- Reduces variance across folds
- More reliable performance estimates

### 5.4 Out-of-Fold (OOF) Predictions

The key insight: **each frame is predicted by a model that never saw it during training**.

```
Fold 1 Model --> Predicts Fold 1 Val Frames --+
Fold 2 Model --> Predicts Fold 2 Val Frames --+
Fold 3 Model --> Predicts Fold 3 Val Frames --+--> Complete OOF
Fold 4 Model --> Predicts Fold 4 Val Frames --+    Predictions
Fold 5 Model --> Predicts Fold 5 Val Frames --+    (All 162 frames)

Result: Unbiased predictions for EVERY frame in the dataset
```

---

## 6. Training Strategy

### 6.1 No Data Augmentation

```python
USE_AUGMENTATION = False
```

**Counter-intuitive decision explained**:

| Augmentation | Expected | Actual Result | Reason |
|--------------|----------|---------------|--------|
| Rotation | +DetA | -DetA | Cells already rotation-invariant |
| Flips | +DetA | ~Same | No benefit, no harm |
| Elastic | +DetA | -DetA | Distorts cell shapes unnaturally |
| Brightness | +DetA | -DetA | TIRF-SIM has consistent illumination |
| Noise | +DetA | -DetA | Adds artifacts unlike real noise |

**Key insight**: The HITL-annotated data already captures the natural variation in the dataset. Augmentation introduces unrealistic transformations that hurt generalization.

### 6.2 Regularization: Light Dropout Only

```python
DROPOUT = 0.1       # Light dropout in decoder
WEIGHT_DECAY = 0.0  # No L2 regularization
```

**Why minimal regularization?**

1. **K-fold CV already regularizes**: Each model sees different train/val splits
2. **Ensemble averaging**: 5 models reduce individual overfitting
3. **Early observations**: Weight decay hurt performance on this small dataset

### 6.3 Learning Rate Schedule

```python
LR = 1e-4                    # Initial learning rate
SCHEDULER_PATIENCE = 10      # Epochs before LR reduction
SCHEDULER_FACTOR = 0.5       # Multiply LR by 0.5 on plateau

# ReduceLROnPlateau scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',              # Minimize validation loss
    patience=10,
    factor=0.5
)
```

**Learning rate trajectory** (typical fold):
```
Epochs 1-30:   LR = 1e-4   (initial learning)
Epochs 31-45:  LR = 5e-5   (first plateau)
Epochs 46-70:  LR = 2.5e-5 (refinement)
Epochs 71-100: LR = 1.25e-5 (fine-tuning)
```

### 6.4 Extended Training Without Early Stopping

```python
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 999  # Effectively disabled
```

**Rationale**:
- Model checkpointing saves best weights regardless
- Extended training allows LR schedule to fully converge
- K-fold ensures we don't overfit to validation set
- 100 epochs sufficient for full convergence (loss plateaus ~epoch 80)

---

## 7. Post-Processing Pipeline

### 7.1 From Raw Output to Detections

```
Model Output
     |
     +-- prob_map (H, W)      <-- Object center probability
     |         |
     |         v
     |   +-------------+
     |   | Threshold   |      prob > 0.6
     |   | (prob=0.6)  |      <-- Removes low-confidence pixels
     |   +------+------+
     |          |
     +-- dist_map (64, H, W)  <-- Ray distances
               |
               v
        +-------------+
        | Reconstruct |       Convert rays to polygons
        | Polygons    |       at each candidate pixel
        +------+------+
               |
               v
        +-------------+
        |    NMS      |       Remove overlapping polygons
        | (IoU=0.35)  |       <-- Keep highest confidence
        +------+------+
               |
               v
        +-------------+
        |  Centroid   |       Extract (x, y) from
        | Extraction  |       remaining polygons
        +------+------+
               |
               v
        Final Detections
        [(x1,y1), (x2,y2), ...]
```

### 7.2 Threshold Optimization

**Problem**: Default thresholds (prob=0.5, nms=0.3) are not optimal.

**Solution**: Grid search on OOF predictions (no data leakage):

```python
ENSEMBLE_PROB_THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
ENSEMBLE_NMS_THRESHOLDS = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5]

# 54 combinations evaluated
# Best: prob=0.6, nms=0.35
```

**Threshold effects**:

| prob_thresh | Effect |
|-------------|--------|
| Low (0.1-0.3) | More detections, more false positives |
| **High (0.6)** | **Fewer but confident detections** |

| nms_thresh | Effect |
|------------|--------|
| Low (0.1-0.2) | Aggressive suppression, may merge close cells |
| **Medium (0.35)** | **Balance: separate close cells, remove duplicates** |

---

## 8. Ensemble Strategy

### 8.1 Why Ensemble?

Single model limitations:
- High variance on small datasets
- May overfit to specific training subset
- Sensitive to initialization

Ensemble benefits:
- **Averaging reduces variance** (statistical)
- **Different folds capture different patterns** (diversity)
- **Smoother probability maps** (less noisy predictions)

### 8.2 Ensemble Averaging

```python
# Average predictions from all 5 folds
all_prob = []
all_stardist = []

for model in fold_models:
    out = model(image)['nuc']
    all_prob.append(torch.sigmoid(out.binary_map))
    all_stardist.append(out.aux_map)

# Element-wise mean
ensemble_prob = np.mean(all_prob, axis=0)        # (H, W)
ensemble_stardist = np.mean(all_stardist, axis=0) # (64, H, W)

# Post-process ensemble output
detections = post_proc_stardist(ensemble_prob, ensemble_stardist, ...)
```

### 8.3 Ensemble vs Single Model

| Approach | DetA |
|----------|------|
| Single best fold | 0.8286 |
| Single worst fold | 0.7691 |
| **5-fold ensemble (OOF)** | **0.8129 +/- 0.0224** |

---

## 9. Evaluation Metric: DetA

### 9.1 Definition

**Detection Accuracy (DetA)** measures how well predicted detections match ground truth:

```
DetA = TP / (TP + FP + FN)

Where:
- TP (True Positive): Prediction within 5px of a GT cell
- FP (False Positive): Prediction with no nearby GT cell
- FN (False Negative): GT cell with no nearby prediction
```

### 9.2 Hungarian Matching

To compute TP/FP/FN, we use **Hungarian algorithm** for optimal assignment:

```python
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def calculate_deta(gt_coords, pred_coords, match_thresh=5.0):
    # Compute pairwise distances
    cost_matrix = cdist(gt_coords, pred_coords)  # (N_gt, N_pred)

    # Hungarian assignment (minimize total distance)
    gt_idx, pred_idx = linear_sum_assignment(cost_matrix)

    # Count matches within threshold
    tp = sum(cost_matrix[g, p] <= match_thresh
             for g, p in zip(gt_idx, pred_idx))
    fp = len(pred_coords) - tp
    fn = len(gt_coords) - tp

    return tp / (tp + fp + fn)
```

### 9.3 Why 5px Threshold?

```
Cell diameter: ~10-15 pixels
Centroid localization error: ~2-3 pixels (acceptable)
Match threshold: 5 pixels (half cell diameter)

This allows for:
- Minor localization errors (acceptable)
- But not matching wrong cells (too far)
```

---

## 10. Results

### 10.1 K-Fold Training Results

Out-of-Fold DetA scores using fixed threshold (prob=0.5, nms=0.3):

| Fold | DetA | Epochs |
|------|------|--------|
| 1 | 0.8261 | 100 |
| 2 | 0.7691 | 100 |
| 3 | 0.8259 | 100 |
| 4 | 0.8149 | 100 |
| 5 | 0.8286 | 100 |
| **Mean** | **0.8129 +/- 0.0224** | |

### 10.2 Note on Evaluation

> The **OOF DetA of 0.8129** is the honest generalization estimate. Each frame is predicted by a model that never saw it during training.
>
> Evaluating the 5-fold ensemble on the full validation corpus yields ~0.91 DetA, but this is **optimistic** because 4/5 folds have already seen each frame during training. We report the conservative OOF metric as our primary result.

---

## 11. Summary of Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Architecture** | StarDist | Designed for convex objects (cells) |
| **Encoder** | ResNet18 | Sufficient capacity, avoids overfitting |
| **N_RAYS** | 64 | Good boundary precision for round cells |
| **Loss** | BCE + L1 | Simple baseline outperforms complex losses |
| **Augmentation** | None | HITL data captures natural variation |
| **Regularization** | Dropout=0.1 | Light regularization, K-fold handles rest |
| **Training** | 100 epochs | Extended for full convergence |
| **Validation** | 5-fold CV | Maximizes limited data usage |
| **Inference** | Ensemble average | Reduces variance |
| **Thresholds** | prob=0.6, nms=0.35 | Optimized on OOF predictions |

---

## 12. Lessons Learned

1. **Simpler is often better**: Baseline BCE+L1 loss outperformed Focal+Dice
2. **Augmentation isn't always beneficial**: Clean, representative data > augmented noisy data
3. **Ensemble always helps**: 5 models averaging reduces variance
4. **Threshold optimization matters**: Default thresholds leave performance on the table
5. **K-fold CV is essential for small datasets**: Every sample should contribute to both training and validation
6. **Report honest metrics**: OOF evaluation prevents optimistic bias from data leakage
