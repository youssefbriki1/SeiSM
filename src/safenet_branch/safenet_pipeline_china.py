"""
safenet_pipeline_china.py — SafeNet pipeline runner for the China dataset.

Usage:
    # Skip preprocessing (default), log to W&B
    python safenet_pipeline_china.py

    # Run preprocessing first, then train
    python safenet_pipeline_china.py -s

    # Tune hyperparams from the CLI
    python safenet_pipeline_china.py --epochs 30 --lr 3e-4 --focal-gamma 1.5

    # Offline / dry-run (no W&B upload)
    python safenet_pipeline_china.py --wandb-mode offline

    # Disable W&B entirely
    python safenet_pipeline_china.py --disable-wandb
"""

import argparse
import sys
import time
from pathlib import Path

from pyproj.__main__ import parser

# ── Path setup ───────────────────────────────────────────────────────────────
SRC_ROOT   = Path(__file__).resolve().parent.parent  # repo root / src
DATA_DIR   = SRC_ROOT.parent / "data" / "china"
SCRIPT_DIR = SRC_ROOT / "data-processing" / "safenet"

sys.path.insert(0, str(SRC_ROOT))

from safenet_branch.safenet_pipeline import SafeNetPipeline
from safenet_branch.safenet_config import PipelineConfig


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SafeNet on the China dataset."
    )

    # Preprocessing
    parser.add_argument(
        "-s", "--skip-preprocessing",
        action="store_true",
        default=True,
        help="Skip the bash preprocessing step (default: True — pass -s to run it).",
    )

    # Run params
    parser.add_argument("--training-pickle", type=str, default="training_output.pickle", help="Preprocessed training data file.")
    parser.add_argument("--testing-pickle",  type=str, default="testing_output.pickle",  help="Preprocessed testing data file.")
    parser.add_argument("--training-labels", type=str, default="training_labels.csv", help="CSV file with training labels.")
    parser.add_argument("--testing-labels",  type=str, default="testing_labels.csv",  help="CSV file with testing labels.")

    # Training hyperparams
    parser.add_argument("--epochs",      type=int,   default=20,   help="Number of training epochs.")
    parser.add_argument("--lr",          type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--patches",     type=int,   default=85,   help="Number of spatial patches.")
    parser.add_argument("--focal-gamma", type=float, default=2.0,  help="Focal loss gamma.")
    parser.add_argument(
        "--class-weights",
        type=float, nargs=4, default=None,
        metavar=("W0", "W1", "W2", "W3"),
        help="Per-class focal loss weights (4 values). Default: uniform.",
    )

    # SSM architecture
    parser.add_argument("--ssm-d-model",  type=int, default=128, help="SSM hidden dim.")
    parser.add_argument("--ssm-d-state",  type=int, default=16,  help="SSM state size.")
    parser.add_argument("--ssm-n-layers", type=int, default=2,   help="Number of SSM layers.")

    # W&B
    parser.add_argument("--disable-wandb",  action="store_true",  help="Turn off W&B logging.")
    parser.add_argument("--wandb-project",  type=str, default="safenet-china")
    parser.add_argument("--wandb-run-name", type=str, default="",  help="Custom run name.")
    parser.add_argument(
        "--wandb-mode",
        type=str, choices=["online", "offline", "disabled"], default="online",
    )
    parser.add_argument("--wandb-log-freq", type=int, default=50, help="Log every N batches.")

    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    cfg = PipelineConfig(
        num_patches    = args.patches,
        num_epochs     = args.epochs,
        learning_rate  = args.lr,
        focal_gamma    = args.focal_gamma,
        class_weights  = args.class_weights,
        ssm_d_model    = args.ssm_d_model,
        ssm_d_state    = args.ssm_d_state,
        ssm_n_layers   = args.ssm_n_layers,
        wandb_project  = args.wandb_project,
        wandb_run_name = args.wandb_run_name or f"china-{int(time.time())}",
        wandb_mode     = args.wandb_mode,
        wandb_log_freq = args.wandb_log_freq,
        disable_wandb  = args.disable_wandb,
    )

    # ── W&B init ─────────────────────────────────────────────────────────
    wandb_run = None
    if not cfg.disable_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Run `pip install wandb` or pass --disable-wandb."
            ) from exc

        wandb_run = wandb.init(
            project = cfg.wandb_project,
            entity  = "ift3710-ai-slop",
            name    = cfg.wandb_run_name,
            mode    = cfg.wandb_mode,
            config  = cfg.to_dict(),
        )

    # ── Pipeline ─────────────────────────────────────────────────────────
    try:
        pipeline = SafeNetPipeline(
            data_dir             = DATA_DIR,
            preprocessing_script = SCRIPT_DIR / "run_preprocessing.sh",
            training_pickle      = args.training_pickle,
            testing_pickle       = args.testing_pickle,
            training_labels      = args.training_labels,
            testing_labels       = args.testing_labels,
            num_patches          = cfg.num_patches,
            num_epochs           = cfg.num_epochs,
            learning_rate        = cfg.learning_rate,
            focal_gamma          = cfg.focal_gamma,
            class_weights        = cfg.class_weights_tensor(),
            ssm_d_model          = cfg.ssm_d_model,
            ssm_d_state          = cfg.ssm_d_state,
            ssm_n_layers         = cfg.ssm_n_layers,
            wandb_run            = wandb_run,
            wandb_log_freq       = cfg.wandb_log_freq,
        )

        pipeline.smoke_test()
        pipeline.train(skip_preprocessing=args.skip_preprocessing)
        pipeline.run()

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()