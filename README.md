# ift3710: Earthquake Prediction with State Space Models

This project implements state space models (SSMs) for earthquake forecasting, focusing on the California Earthquake Catalog (CEED) dataset. It combines spatial feature engineering with SSM architectures like Mamba for predictive modeling.

## Project Overview

The pipeline processes the CEED earthquake catalog into spatial tensors representing geological features and seismicity patterns. These tensors are then used to train SSM-based models for earthquake prediction.

### Key Components
- **Data Processing**: Converts CEED metadata and USGS geospatial data into 5-channel spatial tensors (lithology, faults, magnitude density)
- **Models**: SSM implementations (Mamba-based) for sequence modeling of spatial-temporal earthquake data
- **Training**: End-to-end training scripts with focal loss and evaluation metrics

## Requirements

- Python 3.12.x
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   uv sync
   ```

## Data Preprocessing

The preprocessing pipeline converts the CEED earthquake catalog into spatial tensors and training patches.

### CEED Dataset
- Source: [AI4EPS/CEED on Hugging Face](https://huggingface.co/datasets/AI4EPS/CEED)
- Contains earthquake metadata (1987-2020)
- Combined with USGS fault and geology shapefiles

### Running Preprocessing

```bash
cd src/data-processing/california
./run_pre_processing.sh
```

This script will:
1. Download CEED `events.csv` if not present
2. Download required USGS shapefiles
3. Generate yearly spatial tensors (`tensor_YEAR.npy`) and patches (`patches_YEAR.npy`)

#### Output Format
- **Tensor shape**: `(5, 512, 512)` per year
- **Channels**:
  - 0: Sedimentary lithology
  - 1: Volcanic lithology
  - 2: Crystalline lithology
  - 3: Distance to nearest fault
  - 4: Earthquake magnitude density
- **Patches**: Split into `(N_patches, 5, 64, 64)` for training

## Training

After preprocessing, run the SSM pipeline to generate embeddings:

```bash
python src/ssm1_pipeline.py
```

This processes the preprocessed CEED data through SafeNet embeddings and SpatialSSM to produce `data/ssm1_output.pickle`.

For full model training, use the available training scripts:

```bash
# Multimodal training (SafeNet + spatial features)
python src/main_mutimodal.py

# Baseline models (LSTM, etc.)
python src/baseline_train.py --model lstm

# QuakeMamba2 model
python src/main.py
```

### Model Options
- **SpatialSSM**: Mamba-based model for spatial feature sequences
- **QuakeMamba2**: Enhanced SSM architecture
- **SafeNetFull**: Multimodal model combining SafeNet features with spatial data
- **Baselines**: LSTM, ResNet implementations

## Evaluation

Models are evaluated using F1-score, precision, and recall metrics with focal loss. Training supports Weights & Biases logging.

## Directory Structure

```
├── src/
│   ├── data-processing/california/    # CEED preprocessing
│   ├── models/                        # Model implementations
│   └── utils/                         # Datasets, losses, etc.
├── data/                              # Raw and processed data
├── slurm/                             # HPC training scripts
└── tests/                             # Unit tests
```

## Contributing

This project is part of ift3710 coursework. For questions or contributions, please refer to the [course materials](https://alexhernandezgarcia.com/teaching/mlprojects26/).

test