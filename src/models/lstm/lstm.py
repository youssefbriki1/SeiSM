"""
LSTM-based earthquake magnitude class predictor.

Data format (from pipeline.py / SafeNetDataset):
  - features: (batch, seq=10, num_patches=86, features=282)
      86 patches: patch 0 = general map, patches 1-85 = regions
  - labels:   (batch, 85)  — ordinal class 0-3, one per region (no general map)

Classes:
  0 : M < 5
  1 : 5 ≤ M < 6
  2 : 6 ≤ M < 7
  3 : M ≥ 7
"""

import os
import sys
import pickle
import subprocess

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Paths relative to this file
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PROCESSING_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "..", "data-processing")
)
_DATA_DIR = os.path.join(_DATA_PROCESSING_DIR, "data")

NUM_CLASSES = 4  # 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Neural network module
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """Stacked LSTM applied independently to each spatial patch.

    Accepts the same input format as SafeNetDataset / the Mamba model:

        Input : (batch, seq=10, num_patches=86, features=282)
        Output: (batch, 85, num_classes=4)

    Patch 0 (general map) is used during the LSTM pass but dropped from
    the output so that the shape aligns with the 85-region label vector.
    """

    def __init__(
        self,
        input_size: int = 282,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, num_patches, features)
        batch, seq, num_patches, features = x.shape

        # Run the LSTM independently for every patch
        # → reshape to (batch * num_patches, seq, features)
        x = x.permute(0, 2, 1, 3).reshape(batch * num_patches, seq, features)
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])                             # (batch*patches, hidden)
        logits = self.classifier(out)                           # (batch*patches, classes)

        logits = logits.view(batch, num_patches, -1)            # (batch, patches, classes)
        return logits[:, 1:, :]                                 # drop patch 0 → (batch, 85, classes)


# ---------------------------------------------------------------------------
# High-level predictor  (standalone, no wandb dependency)
# ---------------------------------------------------------------------------
class LSTMEarthquakePredictor:
    """Train and evaluate an LSTM model on earthquake catalog features.

    Parameters
    ----------
    training_data_file:
        Path to training_output.pickle (features).
    training_labels_file:
        Path to training_labels.pickle.
    testing_data_file:
        Path to testing_output.pickle (features).
    eval_labels_file:
        Path to testing_labels.pickle used for evaluation.
    hidden_size, num_layers, dropout, lr, epochs, batch_size:
        LSTM hyper-parameters.
    device:
        'cpu', 'cuda', or None (auto-detect).
    """

    def __init__(
        self,
        training_data_file: str,
        training_labels_file: str,
        testing_data_file: str,
        eval_labels_file: str,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        device: str = None,
    ):
        self.training_data_file = training_data_file
        self.training_labels_file = training_labels_file
        self.testing_data_file = testing_data_file
        self.eval_labels_file = eval_labels_file

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        self.model: LSTMModel | None = None

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_pickle(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _prepare_tensors(
        self, data_file: str, labels_file: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load pickles and return (N, seq=10, 86, 282) / (N, 85) tensors."""
        output = self._load_pickle(data_file)
        labels = self._load_pickle(labels_file)

        eq_data = output["eq_data"]  # list of (10, 86, 282) arrays

        X = np.stack(eq_data, axis=0).astype(np.float32)  # (N, 10, 86, 282)
        y = np.stack(labels, axis=0).astype(np.int64)     # (N, 85)

        return torch.tensor(X), torch.tensor(y)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> None:
        X_train, y_train = self._prepare_tensors(
            self.training_data_file, self.training_labels_file
        )

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = LSTMModel(
            input_size=X_train.shape[-1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=NUM_CLASSES,
            dropout=self.dropout,
        ).to(self.device)

        # Inverse-frequency class weights to handle class imbalance
        counts = np.bincount(y_train.numpy().ravel(), minlength=NUM_CLASSES).astype(np.float32)
        weights = torch.tensor(1.0 / (counts + 1e-6)).to(self.device)
        weights /= weights.sum()

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        print(f"\nTraining LSTM for {self.epochs} epochs on {len(dataset)} samples…")
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)                    # (batch, 85, 4)
                loss = criterion(logits.reshape(-1, NUM_CLASSES), y_batch.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * X_batch.size(0)

            avg_loss = total_loss / len(dataset)
            scheduler.step(avg_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{self.epochs} | Loss: {avg_loss:.4f}")

        print("Training complete.\n")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self) -> dict:
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        X_test, y_test = self._prepare_tensors(
            self.testing_data_file, self.eval_labels_file
        )

        self.model.eval()
        all_preds, all_probs = [], []

        loader = DataLoader(TensorDataset(X_test), batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch).reshape(-1, NUM_CLASSES)  # (batch*85, 4)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_preds.append(probs.argmax(axis=-1))
                all_probs.append(probs)

        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)
        y_true = y_test.numpy().ravel()

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")

        print("=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"Accuracy      : {acc:.4f}")
        print(f"Macro F1      : {f1_macro:.4f}")
        print(f"Weighted F1   : {f1_weighted:.4f}")
        print(f"Macro AUC-ROC : {auc:.4f}")
        print()
        print("Classification Report:")
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=["M<5", "M5-6", "M6-7", "M≥7"],
                zero_division=0,
            )
        )

        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "auc_roc": auc,
        }

    def run(self) -> dict:
        """Train the model then evaluate it. Returns the metrics dict."""
        self.train()
        return self.evaluate()


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def main() -> dict:
    """Run the full pipeline: split → feature engineering → LSTM train/eval."""

    print("=" * 50)
    print("Step 1: Splitting raw data")
    print("=" * 50)
    subprocess.run(
        [sys.executable, "split_data.py"],
        cwd=_DATA_PROCESSING_DIR,
        check=True,
    )

    print()
    print("=" * 50)
    print("Step 2: Feature engineering pipeline")
    print("=" * 50)
    subprocess.run(
        [sys.executable, "pipeline.py"],
        cwd=_DATA_PROCESSING_DIR,
        check=True,
    )

    print()
    print("=" * 50)
    print("Step 3: LSTM training and evaluation")
    print("=" * 50)
    predictor = LSTMEarthquakePredictor(
        training_data_file=os.path.join(_DATA_DIR, "training_output.pickle"),
        training_labels_file=os.path.join(_DATA_DIR, "training_labels.pickle"),
        testing_data_file=os.path.join(_DATA_DIR, "testing_output.pickle"),
        eval_labels_file=os.path.join(_DATA_DIR, "testing_labels.pickle"),
    )

    return predictor.run()


if __name__ == "__main__":
    main()
