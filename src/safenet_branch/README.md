# Safenet-like branch

## How to run

From the repo root:

```bash
python -m src.safenet_branch.safenet_pipeline_ceed
```

```bash
python -m src.safenet_branch.safenet_pipeline_china
```

By default, preprocessing is **skipped**. Pass `-s` / `--skip-preprocessing` to run it first.

Each run will:
1. Optionally run the bash preprocessing script for the dataset
2. Train the embedder + SSM + classification head, saving the best checkpoint (`checkpoint.pt`) based on test macro-F1
3. Load the checkpoint and write SSM outputs to `ssm1_output.pickle` for downstream MLP fusion

---

## Parameters

Add `--help` to see all parameters.

### Training

| Flag | CEED default | China default | Description |
|---|---|---|---|
| `--epochs` | 20 | 20 | Number of training epochs |
| `--lr` | 1e-4 | 1e-4 | Adam learning rate |
| `--patches` | 64 | 85 | Number of spatial patches |
| `--focal-gamma` | 2.0 | 2.0 | Focal loss γ |
| `--class-weights W0 W1 W2 W3` | None | None | Per-class weights for `M<5`, `M5-6`, `M6-7`, `M≥7` |

### SSM architecture

| Flag | Default | Description |
|---|---|---|
| `--ssm-d-model` | 128 | Hidden dimension |
| `--ssm-d-state` | 16 | State size |
| `--ssm-n-layers` | 2 | Number of layers |

### W&B

| Flag | Default | Description |
|---|---|---|
| `--wandb-project` | `safenet-california` / `safenet-china` | Project name |
| `--wandb-run-name` | auto-generated | Custom run name |
| `--wandb-mode` | `online` | `online` \| `offline` \| `disabled` |
| `--wandb-log-freq` | 50 | Log every N training batches |
| `--disable-wandb` | — | Disable W&B entirely |

---

## `ssm1_output.pickle` structure

```python
{
    "train": {
        "ssm1_out": np.ndarray,  # shape: (N_train, num_patches, ssm_d_model)
        "labels":   np.ndarray,  # shape: (N_train,)
    },
    "test": {
        "ssm1_out": np.ndarray,  # shape: (N_test, num_patches, ssm_d_model)
        "labels":   np.ndarray,  # shape: (N_test,)
    },
    "config": {
        "d_model":     int,  # e.g. 128
        "num_patches": int,  # e.g. 64 for CEED, 85 for China
        "num_classes": int,  # 4
    },
}
```

Labels are integer class indices: `0 = M<5`, `1 = M5-6`, `2 = M6-7`, `3 = M≥7`.

---

## Examples

```bash
# Custom run name, higher LR, more epochs
python -m safenet_pipeline_ceed --wandb-run-name high_lr_more_epochs_2 --lr 1.75e-4 --epochs 30

# Class-weighted focal loss, offline W&B
python -m safenet_pipeline_china --class-weights 1.0 4.0 15.0 78.0 --wandb-mode offline

# Quick dry run, no W&B
python -m safenet_pipeline_ceed --epochs 2 --disable-wandb
```