import numpy as np
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


class MultimodalSafeNetDataset(Dataset):
    """Dataset that returns both catalog features and map image patches.

    Pickle structure:
        eq_data : list of (10, 86, 282) arrays — catalog features per target year
        png     : list of (85, 50, 50, 5) arrays — map patches per calendar year

    Alignment (sliding window): for eq_data[i] whose 10 history years span
    indices i..i+9, the corresponding maps are png[i : i+10].
    """

    def __init__(self, features_path, labels_path=None, labels_data=None):
        features_path = Path(features_path)
        with open(features_path, 'rb') as f:
            features_obj = pickle.load(f)

        if not isinstance(features_obj, dict) or 'eq_data' not in features_obj:
            raise KeyError(f"`eq_data` key not found in features file: {features_path}")

        self.catalog = features_obj['eq_data']
        self.png = features_obj.get('png', None)

        if self.png is None or len(self.png) == 0:
            raise KeyError(
                f"`png` key is missing or empty in {features_path}. "
                "Run the pipeline with image_process to generate map data."
            )

        if len(self.png) < len(self.catalog) + 9:
            raise ValueError(
                f"Need at least len(eq_data)+9 = {len(self.catalog)+9} png entries "
                f"for sliding-window alignment, but got {len(self.png)}."
            )

        self.labels = None
        if labels_data is not None:
            self.labels = labels_data
        elif labels_path is not None:
            with open(Path(labels_path), 'rb') as f:
                self.labels = pickle.load(f)
        elif isinstance(features_obj, dict):
            for key in ('labels', 'targets', 'y'):
                if key in features_obj:
                    self.labels = features_obj[key]
                    break

        if self.labels is None:
            raise FileNotFoundError(
                "Labels not found. Provide a labels pickle or include them in the "
                f"features pickle. Available keys: {list(features_obj.keys())}"
            )

        if len(self.labels) != len(self.catalog):
            raise ValueError(
                f"Label count ({len(self.labels)}) != sample count ({len(self.catalog)})."
            )

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        catalog = torch.tensor(self.catalog[idx], dtype=torch.float32)   # (10, 86, 282)
        maps = torch.from_numpy(
            np.stack(self.png[idx: idx + 10], axis=0)                    # (10, 85, 50, 50, 5)
        ).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)         # (85,)
        return {"catalog": catalog, "maps": maps}, label
