import torch
import pickle
from pathlib import Path
from torch.utils.data import Dataset

class SafeNetDataset(Dataset):
    def __init__(self, features_path, labels_path=None, labels_data=None):
        features_path = Path(features_path)
        with open(features_path, 'rb') as f:
            features_obj = pickle.load(f)

        if isinstance(features_obj, dict):
            if 'eq_data' not in features_obj:
                raise KeyError(f"`eq_data` key not found in features file: {features_path}")
            self.features = features_obj['eq_data']
        else:
            self.features = features_obj

        self.labels = None

        if labels_data is not None:
            self.labels = labels_data
        elif labels_path is not None:
            labels_path = Path(labels_path)
            with open(labels_path, 'rb') as f:
                self.labels = pickle.load(f)
        elif isinstance(features_obj, dict):
            for key in ('labels', 'targets', 'y'):
                if key in features_obj:
                    self.labels = features_obj[key]
                    break

        if self.labels is None:
            available_keys = list(features_obj.keys()) if isinstance(features_obj, dict) else []
            raise FileNotFoundError(
                "Labels were not found. Provide a labels pickle with `--train_labels_file` / "
                "`--val_labels_file`, or include labels in the features pickle under one of: "
                "`labels`, `targets`, `y`. "
                f"Features file: {features_path}. Available keys: {available_keys}"
            )

        if len(self.labels) != len(self.features):
            raise ValueError(
                f"Number of labels ({len(self.labels)}) does not match number of feature samples "
                f"({len(self.features)})."
            )
            
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32) 
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
