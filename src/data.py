#code is used for augmentation

# src/data.py
"""
Data loading and preprocessing for Cats vs Dogs classification.
- Applies augmentation (flip, rotation, zoom, brightness) on training images.
- Uses ImageNet mean/std normalization.
- Provides train/val/test DataLoaders with optional per-class subsampling.
"""

from pathlib import Path
from typing import Tuple, Optional, List
import random
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from PIL import Image

# ------------------------
# Normalization constants (ImageNet)
# ------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ------------------------
# Augmentation and normalization transforms
# ------------------------
def get_transforms(img_size: int = 224, aug_strength: str = "standard"
                   ) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, eval_transform).

    aug_strength:
      - "none":       just resize + center crop + normalize
      - "standard":   random crop/zoom + flip + rotation + brightness
      - "plus":       same as standard + mild color jitter (contrast/saturation)
    """
    # ----- TRAIN AUGMENTATION -----
    train_tf_list = []

    if aug_strength in ("standard", "plus"):
        # geometric augs
        train_tf_list += [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),   # random zoom & crop
            transforms.RandomHorizontalFlip(p=0.5),                     # horizontal flip
            transforms.RandomRotation(degrees=15),                      # rotation
        ]
        # photometric augs
        train_tf_list += [transforms.ColorJitter(brightness=0.2)]       # brightness adjust
        if aug_strength == "plus":
            train_tf_list += [transforms.ColorJitter(contrast=0.15, saturation=0.15)]
    else:
        # no augmentation
        train_tf_list += [transforms.Resize(int(img_size * 1.15)),
                          transforms.CenterCrop(img_size)]

    # Always convert to tensor and normalize
    train_tf_list += [transforms.ToTensor(),
                      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    train_tf = transforms.Compose(train_tf_list)

    # ----- EVAL/TEST -----
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    return train_tf, eval_tf


# ------------------------
# Helper: subsample per class (for smaller training sets)
# ------------------------
def _pick_subset_indices(targets: List[int],
                         n_per_class: Optional[int],
                         seed: int = 42) -> List[int]:
    if n_per_class is None:
        return list(range(len(targets)))
    rng = np.random.default_rng(seed)
    idxs = []
    classes = sorted(set(targets))
    targets_np = np.array(targets)
    for c in classes:
        cand = np.where(targets_np == c)[0]
        take = min(n_per_class, len(cand))
        idxs += rng.choice(cand, size=take, replace=False).tolist()
    random.shuffle(idxs)
    return idxs


# ------------------------
# Public: build train/val loaders
# ------------------------
def make_loaders(root: str = "dataset",
                 img_size: int = 224,
                 batch_size: int = 64,
                 workers: int = 4,
                 train_per_class: Optional[int] = None,
                 val_per_class: Optional[int] = None,
                 aug_strength: str = "standard",
                 seed: int = 42
                 ) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create DataLoaders for training and validation.

    Returns: (train_loader, val_loader, class_names)
    """
    train_tf, eval_tf = get_transforms(img_size=img_size, aug_strength=aug_strength)

    train_dir = Path(root) / "train"
    val_dir   = Path(root) / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Expected dataset/train and dataset/val directories")

    train_ds_full = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds_full   = datasets.ImageFolder(str(val_dir),   transform=eval_tf)

    tr_idx = _pick_subset_indices(train_ds_full.targets, train_per_class, seed)
    va_idx = _pick_subset_indices(val_ds_full.targets,   val_per_class,   seed)

    train_ds = Subset(train_ds_full, tr_idx)
    val_ds   = Subset(val_ds_full,   va_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)

    return train_loader, val_loader, train_ds_full.classes  # e.g. ['cat','dog']


# ------------------------
# Test dataset & loader
# ------------------------
class TestFolder(Dataset):
    """Simple dataset for unlabeled test images."""
    def __init__(self, root: str, transform):
        self.paths = sorted([p for p in Path(root).iterdir() if p.is_file()])
        if not self.paths:
            raise FileNotFoundError(f"No files found in {root}")
        self.transform = transform

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        return self.transform(img), p.name  # return filename for later id extraction


def make_test_loader(root: str = "dataset",
                     img_size: int = 224,
                     batch_size: int = 64,
                     workers: int = 4) -> DataLoader:
    """
    Build DataLoader for test set images.
    """
    _, eval_tf = get_transforms(img_size=img_size, aug_strength="none")
    ds = TestFolder(Path(root) / "test", eval_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True)
