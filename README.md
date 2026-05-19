# 🫁 Pneumonia Detection with Pure ViT-Fuzzy Classification

A **from-scratch**, GPU-accelerated Vision Transformer with a Fuzzy Logic classification head for binary pneumonia detection from chest X-ray images.

> **No PyTorch. No TensorFlow.** The entire deep-learning stack — forward pass, backpropagation, optimizer, and all layers — is implemented manually using **CuPy** for raw CUDA GPU computation.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Dataset](#-dataset)
- [Training](#-training)
- [Inference & Web App](#-inference--web-app)
- [Model Artifact](#-model-artifact)
- [Results](#-results)
- [Configuration](#-configuration)
- [License](#-license)

---

## 🔍 Overview

This project implements a **Pure Vision Transformer (ViT)** enhanced with a **Fuzzy Logic** inference backend for classifying chest X-ray images as either **Normal** or **Pneumonia**.

**Key highlights:**

| Feature | Detail |
|---|---|
| **Framework** | Pure CuPy (no PyTorch / TensorFlow) |
| **Hardware target** | NVIDIA RTX 3050 (4 GB VRAM) |
| **Task** | Binary classification (Normal vs. Pneumonia) |
| **Input resolution** | 128 × 128 RGB (after CLAHE) |
| **Preprocessing** | Grayscale CLAHE enhancement → RGB conversion |
| **Architecture** | Linear Patch Embedding → Transformer Encoder → Fuzzy Membership → Cosine Similarity Classifier |
| **Optimizer** | Custom AdamW with gradient clipping |
| **LR schedule** | Cosine annealing with linear warmup |
| **Data loading** | Async threaded prefetcher with on-the-fly augmentation |

---

## 🏗 Architecture

```
Input Image (3 × 128 × 128)
        │
        ▼
┌─────────────────────────┐
│  CLAHE Preprocessing    │   Contrast-Limited Adaptive Histogram Equalization
│  (clipLimit=4.0, 8×8)   │   Grayscale → CLAHE → RGB → Normalize [0,1]
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Linear Patch Embedding │   Splits image into 8×8 non-overlapping patches
│  (256 patches, dim=64)  │   Flattens each patch → Linear projection to embed_dim
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  + Positional Embedding │   Learnable positional embeddings (1 × 256 × 64)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Transformer Encoder    │   4 × TransformerBlock
│  ×4 Layers              │   Each block:
│                         │     Pre-Norm → Multi-Head Self-Attention (8 heads)
│                         │     + Residual Connection
│                         │     Pre-Norm → FFN (ReLU, 2× expansion)
│                         │     + Residual Connection
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Global Average Pooling │   Mean over sequence dimension
│  + LayerNorm            │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Neural Projection      │   Linear(64 → 32) + LayerNorm
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Fuzzy Membership Layer │   16 Gaussian fuzzy rules
│  (32 → 16 rules)       │   Learnable centers & widths
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Fuzzy Aggregation      │   Weighted sum of membership degrees
│  (16 → 32)             │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Cosine Similarity      │   Learnable class prototypes
│  Classifier (τ = 0.5)   │   Temperature-scaled cosine similarity
└─────────────────────────┘
        │
        ▼
     Logits (2)
```

### Custom Layer Implementations

All layers are implemented from scratch with full forward **and** backward passes:

| Layer | Description |
|---|---|
| `Linear` | Fully-connected with He initialization |
| `LayerNorm` | Layer normalization with learnable γ, β |
| `ReLU` | Rectified Linear Unit |
| `LinearPatchEmbedding` | Reshapes image into flat patches → linear projection |
| `SelfAttention` | Multi-head scaled dot-product attention |
| `PositionWiseFeedForward` | Two-layer FFN with ReLU activation |
| `TransformerBlock` | Pre-Norm Transformer block with residual connections |
| `NeuralProjectionLayer` | Linear + LayerNorm dimensionality reduction |
| `FuzzyMembershipLayer` | Gaussian membership functions with learnable parameters |
| `FuzzyAggregationLayer` | Learnable weighted aggregation of fuzzy activations |
| `SimilarityClassifier` | Cosine similarity to learnable class prototypes |
| `AdamWOptimizer` | AdamW optimizer with gradient clipping |

---

## 📁 Project Structure

```
nee_sc_pro/
│
├── model5.ipynb                       # 📓 Training notebook (full pipeline)
├── gradio_app.py                      # 🌐 Gradio web app for inference
├── pneumonia_pure_vit_model.pkl       # 💾 Serialized trained model (~139 MB)
├── README.md                          # 📖 This file
│
├── Pneumonia_Dataset/                 # 🗂 Dataset directory
│   ├── NORMAL/                        #    Normal chest X-rays
│   └── PNEUMONIA/                     #    Pneumonia chest X-rays
│
└── ...                                # Other project files
```

### File Descriptions

#### `model5.ipynb` — Training Notebook

The Jupyter notebook contains the **complete end-to-end pipeline**:

1. **GPU Initialization & Memory Management** — Configures CuPy memory pool with safety buffer for 4 GB VRAM GPUs.
2. **Hyperparameter Configuration** — All model and training hyperparameters in a single `CONFIG` dictionary.
3. **Data Loading & Preprocessing** — Loads chest X-rays, applies CLAHE contrast enhancement, splits into train/test (90/10 stratified).
4. **Async Data Prefetcher** — Multi-threaded `DataPrefetcher` class that pre-loads and augments batches on background threads.
5. **Math Primitives** — `AdamWOptimizer`, `Linear`, `ReLU`, `LayerNorm` — all with manual forward and backward implementations.
6. **ViT Architecture** — `LinearPatchEmbedding`, `SelfAttention`, `PositionWiseFeedForward`, `TransformerBlock`.
7. **Fuzzy Logic Backend** — `NeuralProjectionLayer`, `FuzzyMembershipLayer`, `FuzzyAggregationLayer`, `SimilarityClassifier`, and the top-level `HybridCvTFuzzy` model.
8. **Training Loop** — Gradient accumulation (effective batch = 32 × 8 = 256), cosine LR schedule with 5-epoch warmup, class-weighted cross-entropy loss, per-epoch train/val accuracy logging.
9. **Evaluation & Visualization** — Classification report, confusion matrix (heatmap), and ROC curves with AUC.
10. **Test Sample Visualization** — Random test images with predicted vs. true labels.

#### `gradio_app.py` — Inference Web Application

A standalone [Gradio](https://gradio.app/) web application that:

- **Loads** the trained model from the serialized `.pkl` file.
- **Preprocesses** uploaded chest X-rays using the same CLAHE pipeline as training.
- **Runs inference** on GPU using CuPy.
- **Displays** class probabilities (Normal / Pneumonia) in an interactive web UI.
- **Manages GPU memory** aggressively between predictions to prevent VRAM exhaustion on long-running servers.

> **Note:** The app re-defines all model class stubs (forward-only) required for `pickle` deserialization — no backward passes are needed for inference.

#### `pneumonia_pure_vit_model.pkl` — Trained Model

A Python `pickle` file containing a dictionary with:

| Key | Value |
|---|---|
| `'model'` | The trained `HybridCvTFuzzy` model instance (with all CuPy weight arrays) |
| `'config'` | The `CONFIG` dictionary used during training |
| `'classes'` | List of class names: `['NORMAL', 'PNEUMONIA']` |

---

## ⚙ Requirements

### Hardware
- **NVIDIA GPU** with CUDA support (tested on RTX 3050 4 GB)

### Software

| Package | Purpose |
|---|---|
| `cupy` | GPU-accelerated array computation (replaces NumPy on GPU) |
| `numpy` | CPU-side array operations |
| `opencv-python` (`cv2`) | Image loading and CLAHE preprocessing |
| `matplotlib` | Training visualization (notebook only) |
| `seaborn` | Confusion matrix heatmaps (notebook only) |
| `scikit-learn` | Train/test split, classification metrics, ROC curves (notebook only) |
| `gradio` | Web UI for inference (`gradio_app.py` only) |

### Installation

```bash
pip install cupy-cuda12x numpy opencv-python matplotlib seaborn scikit-learn gradio
```

> Replace `cupy-cuda12x` with the version matching your CUDA toolkit (e.g., `cupy-cuda11x`).

---

## 🗂 Dataset

The model is trained on a **Pneumonia Chest X-Ray** dataset organized as:

```
Pneumonia_Dataset/
├── NORMAL/       # Normal chest X-ray images
└── PNEUMONIA/    # Pneumonia chest X-ray images
```

| Split | Samples |
|---|---|
| Total | 11,236 |
| Train (90%) | 10,112 |
| Test (10%) | 1,124 |

The dataset path is configured in the `CONFIG` dictionary inside the training notebook:

```python
'dataset_path': r'C:\Users\Computec\Desktop\nee_sc_pro\Pneumonia_Dataset'
```

### Preprocessing Pipeline

1. **Load** as grayscale.
2. **Resize** to 128 × 128.
3. **CLAHE** enhancement (`clipLimit=4.0`, `tileGridSize=(8, 8)`).
4. **Convert** grayscale → RGB (3-channel).
5. **Normalize** pixel values to `[0, 1]`.
6. **Transpose** to channels-first format `(C, H, W)`.

---

## 🏋️ Training

### Running the Training Notebook

1. Open `model5.ipynb` in Jupyter Notebook or VS Code.
2. Ensure the `Pneumonia_Dataset` directory is correctly placed.
3. Run all cells sequentially.

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Image size | 128 × 128 |
| Patch size | 8 × 8 |
| Embedding dimension | 64 |
| Attention heads | 8 |
| Transformer layers | 4 |
| Projection dimension | 32 |
| Fuzzy rules | 16 |
| Temperature (τ) | 0.5 |
| Batch size | 32 |
| Gradient accumulation steps | 8 |
| Effective batch size | 256 |
| Epochs | 20 |
| Base learning rate | 3 × 10⁻⁴ |
| Weight decay | 0.001 |
| Gradient clip value | 1.0 |
| Warmup epochs | 5 |
| LR schedule | Cosine annealing (after warmup) |
| Loss | Class-weighted cross-entropy |

### Data Augmentation (Training Only)

- **Random horizontal flip** (50% probability)
- **Random brightness jitter** (±0.1)

### Training Log (Sample)

```
Epoch 01/20 | LR: 0.00006 | Train Loss: 0.6964 | Train Acc: 0.5067 | Val Acc: 0.5017
Epoch 05/20 | LR: 0.00030 | Train Loss: 0.3677 | Train Acc: 0.8425 | Val Acc: 0.8585
Epoch 10/20 | LR: 0.00025 | Train Loss: 0.2634 | Train Acc: 0.8934 | Val Acc: 0.8959
Epoch 15/20 | LR: 0.00011 | Train Loss: 0.2266 | Train Acc: 0.9105 | Val Acc: 0.9057
Epoch 20/20 | LR: 0.00001 | Train Loss: 0.2018 | Train Acc: 0.9224 | Val Acc: 0.9119
```

---

## 🌐 Inference & Web App

### Running the Gradio App

```bash
python gradio_app.py
```

This launches a local web server (opens automatically in your browser) where you can:

1. **Upload** a chest X-ray image.
2. Click **"Predict"**.
3. View **class probabilities** for Normal and Pneumonia.

### How It Works

1. The app loads the serialized model from `pneumonia_pure_vit_model.pkl` at startup.
2. Each uploaded image is preprocessed using the **same CLAHE pipeline** as training.
3. The preprocessed tensor is transferred to the GPU via CuPy.
4. A forward pass through the model produces logits.
5. Softmax probabilities are computed and returned to the UI.
6. GPU memory is aggressively freed after each prediction to prevent VRAM exhaustion.

### GPU Memory Management

The app implements robust VRAM management for long-running deployments:

- **Memory pool limit** is set at startup with a 500 MB safety buffer.
- A `force_free_vram()` function synchronizes the CUDA stream and frees all cached GPU memory blocks.
- This function is called **before** and **after** every prediction.
- An `OutOfMemoryError` handler provides emergency recovery.

---

## 💾 Model Artifact

**File:** `pneumonia_pure_vit_model.pkl`
**Size:** ~139 MB
**Format:** Python `pickle`

### Contents

```python
{
    'model':   <HybridCvTFuzzy instance>,   # Trained model with CuPy weight arrays
    'config':  { ... },                      # CONFIG dictionary
    'classes': ['NORMAL', 'PNEUMONIA']       # Class label mapping
}
```

### Loading the Model

```python
import pickle

with open("pneumonia_pure_vit_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data['model']
config = model_data['config']
class_names = model_data['classes']
```

> ⚠️ **Important:** All model class definitions must be available in the current scope before unpickling. The `gradio_app.py` file includes forward-only stubs for all required classes.

---

## 📊 Results

### Classification Report

```
              precision    recall  f1-score   support

      NORMAL       0.90      0.93      0.91       562
   PNEUMONIA       0.92      0.90      0.91       562

    accuracy                           0.91      1124
   macro avg       0.91      0.91      0.91      1124
weighted avg       0.91      0.91      0.91      1124
```

### Performance Summary

| Metric | Value |
|---|---|
| **Final Train Accuracy** | 92.24% |
| **Final Validation Accuracy** | 91.19% |
| **Macro F1-Score** | 0.91 |
| **PNEUMONIA AUC** | See ROC curve in notebook |

---

## 🔧 Configuration

All hyperparameters are centralized in the `CONFIG` dictionary:

```python
CONFIG = {
    'img_size': 128,
    'channels': 3,
    'patch_size': 8,
    'embed_dim': 64,
    'num_heads': 8,
    'num_layers': 4,
    'proj_dim': 32,
    'num_rules': 16,
    'temperature': 0.5,
    'batch_size': 32,
    'accumulation_steps': 8,
    'epochs': 20,
    'learning_rate': 3e-4,
    'weight_decay': 0.001,
    'clip_val': 1.0,
    'dataset_path': r'path\to\Pneumonia_Dataset',
    'num_classes': None,      # auto-detected
    'class_weights': None     # auto-computed
}
```

---

## 📝 License

This project is for educational and research purposes.
