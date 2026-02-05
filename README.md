# UDHGAM ML Challenge — Prompt Difficulty Regression

## Overview

This project tackles a **regression problem on tokenized prompts**, where each input prompt is represented as:
- `input_ids` (integer tokens)
- `attention_mask`
- a continuous target label in `[0, 1]`

The goal is to predict the label with **minimum Mean Absolute Error (MAE)**.

This repository documents not only the final models, but the **entire modeling journey**:
- architecture exploration
- ensemble design
- Mixture-of-Experts (MoE)
- feature engineering
- leakage detection
- leaderboard overfitting analysis

Final Result:
- **Top-10 finish on Private Leaderboard**
- Team: **ML Monarchs**

---

## Dataset

### Input
Each sample contains:
- `input_ids`: List[int] — tokenized prompt (max length 256)
- `attention_mask`: List[int] — valid token positions
- `label`: float ∈ [0, 1] (train only)

### Key Properties
- Vocabulary size: ~50k tokens
- Variable sequence lengths
- Tokens are **obfuscated** (no semantic meaning)
- No external metadata

### Implication
Since token semantics are unavailable, the task relies heavily on:
- **structural patterns**
- **token repetition**
- **entropy**
- **sequence dynamics**

---

## Validation Strategy

- 5-fold cross-validation using **predefined folds**
- Strict separation of train / validation
- Extensive checks for:
  - duplicate prompts
  - memorization leakage
  - fold contamination

Only **CV-safe** scores were trusted during development.

---

## Base Architectures Explored

### 1. CNN-based Models
- Multi-kernel CNNs (kernel sizes 3, 5, 7)
- Mean pooling + max pooling
- Lightweight and strong baselines

### 2. GRU-based Models
- Bidirectional GRUs
- Mean + max pooling
- Segment-wise pooling (beginning / middle / end)

### 3. Attention Variants
- Attention-pooled CNNs
- Attention-pooled GRUs
- Stability fallback using mean pooling

### 4. Dilated CNNs
- Dilations: 1, 2, 4, 8
- Larger receptive fields without deeper stacks

Each model outputs a **sigmoid-scaled regression value**.

---

## Hand-Engineered Features

In addition to neural embeddings, multiple structural features were extracted:

### Core Features
- Normalized sequence length
- Unique token ratio
- Token entropy
- Adjacent repetition ratio

### Advanced Structural Features
- Constraint score (tokens appearing ≥3 times)
- Relative position variance (RPC)
- Boundary density (tokens concentrated at 25/50/75%)
- Windowed entropy shifts (phase changes)
- Global token rarity profiles
- Compression ratio (zlib)

These features captured **prompt regularity and rigidity**, which showed moderate correlation with labels.

---

## Ensemble Strategy

### Level-1: Model Zoo
Eight diverse neural models were trained:
- CNN
- CNN (different seed)
- BiGRU
- BiGRU (different seed)
- Segmented GRU
- Attention GRU
- Attention CNN
- Dilated CNN

Out-of-fold (OOF) predictions were collected for all models.

---

### Level-2: Meta Models

Multiple stacking approaches were tested:

- LightGBM regression
- Logit-space regression
- Disagreement features (mean, std, range)
- Per-model deviation features
- Defensive regularization variants
- Ridge and ElasticNet baselines

**Best CV MAE achieved:** ~0.146  
**Best Private LB MAE:** ~0.150

---

## Mixture of Experts (MoE)

To address heterogeneous prompt behavior, several MoE strategies were explored.

### Hard-Binned MoE
Experts trained on bins defined by:
- sequence length
- entropy
- constraint score

Predictions were averaged across experts.

### Gated MoE
- Experts trained per bin
- Separate gate model predicts expert weights
- Softmax-normalized expert routing

MoE yielded **small but consistent gains**, confirming that prompt sub-types exist, though gains were bounded by representation limits.

---

## What Didn’t Work (Important)

- KNN correction on embeddings → overfitting
- Excessive feature stacking → diminishing returns
- Over-calibration (isotonic, quantile) → hurt generalization
- Deep meta-stacks → unstable CV

These experiments are documented to show **what *not* to do**.

---

## Key Takeaways

1. **Structural signals matter**, but are weak individually
2. Ensembling provides large gains early, small gains later
3. MoE helps only when experts see meaningfully different data
4. Validation discipline is critical — many signals looked strong but failed on private LB
5. Transformer-style token interactions were likely the missing inductive bias

---

## Final Results

- 🏆 **Top-10 Private Leaderboard**
- 📉 Private MAE: **0.15016**
- 👥 Authors: **MonarchOfCoding**

---

## Repository Structure

