"""GTSRB dataset loading and transforms."""

from pathlib import Path

import torch
from config import DATA_ROOT, IMG_SIZE
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)


class GTSRBDataset(Dataset[tuple[torch.Tensor, int]]):
    """Cropped GTSRB sign images loaded from directory structure."""

    def __init__(
        self,
        split: str,
        classes: list[int],
        class_remap: dict[int, int] | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.transform = transform or EVAL_TRANSFORM
        self.class_remap = class_remap or {c: i for i, c in enumerate(classes)}
        self.samples = self._collect_samples(split, classes)

    def _collect_samples(
        self, split: str, classes: list[int]
    ) -> list[tuple[Path, int]]:
        split_dir = DATA_ROOT / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split not found: {split_dir}")

        samples: list[tuple[Path, int]] = []
        for cls_id in classes:
            cls_dir = split_dir / f"{cls_id:02d}"
            if cls_dir.exists():
                samples.extend((p, cls_id) for p in cls_dir.glob("*.jpg"))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, orig_class = self.samples[index]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, self.class_remap[orig_class]


def make_remap(classes: list[int]) -> dict[int, int]:
    """Create sequential label remapping: original class ID → 0..N-1."""
    return {c: i for i, c in enumerate(classes)}


def make_loader(
    split: str,
    classes: list[int],
    *,
    batch_size: int,
    shuffle: bool = False,
    transform: transforms.Compose | None = None,
    remap: dict[int, int] | None = None,
) -> DataLoader[tuple[torch.Tensor, int]]:
    """Create a DataLoader for a given split and class subset."""
    ds = GTSRBDataset(split, classes, class_remap=remap, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)
