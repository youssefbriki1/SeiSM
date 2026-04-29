# `src/` — Source Code

## Directory Structure

```
src/
├── main_mutimodal.py          # Training script for SafeNetFull & SeiSM
├── data-processing/           # Data processing pipelines
│   ├── california/            # CEED (California) catalog & map processing
│   └── safenet/               # SafeNet-specific preprocessing utilities
├── models/                    # Model implementations
│   ├── safenet_embeddings.py  # Embedding generation (reproduced from SafeNet paper)
│   ├── spatial_models.py      # Full models: SafeNetFull & SeiSM
│   ├── mamba_minimal.py       # Pure-PyTorch Mamba fallback (CPU/macOS)
├── safenet_branch/            # End-to-end pipelines (processing → training)
│   ├── safenet_pipeline.py    # Core pipeline class
│   ├── safenet_pipeline_ceed.py   # Pipeline configured for California data
│   └── safenet_pipeline_china.py  # Pipeline configured for China data
└── utils/                     # Shared utilities
    ├── dataset.py             # Dataset classes (SafeNetDataset, MultimodalSafeNetDataset)
    ├── focal_loss.py          # Focal Loss for class imbalance
    └── preprocess_safenet.py  # SafeNet feature preprocessing
```

## Key Components

### `main_mutimodal.py`
The primary training script for our multimodal models:
- Supports training both **SafeNetFull** (reproduction of the original SafeNet paper) and **SeiSM** (our Mamba-based architecture) via the `--model` CLI argument. 
- Handles data loading, training loops, evaluation metrics, checkpointing, and Weights & Biases logging.

### `data-processing/`
Contains the preprocessing pipelines for transforming raw earthquake data into model-ready features for both the California (CEED) and China datasets, including:
- catalog feature extraction
- geological map rendering

### `models/`
All model implementation code. See [`models/README.md`](models/README.md) for detailed architecture documentation, including how `safenet_embeddings.py` generates the base embeddings consumed by the full models in `spatial_models.py`.

### `safenet_branch/`
End-to-end pipelines that chain data processing and model training into a single workflow. Each pipeline script (`safenet_pipeline_ceed.py`, `safenet_pipeline_china.py`) configures the core `SafeNetPipeline` class with dataset-specific paths and parameters, allowing one-command execution from raw data to trained model.
