"""Trigger optimization and application for latent backdoor."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from config import (
    IMG_SIZE,
    MODEL_DIR,
    TARGET_CLASS,
    TEACHER_CLASSES,
    TRIGGER_OPT_BATCH_SIZE,
    TRIGGER_OPT_LR,
    TRIGGER_OPT_SAMPLES,
    TRIGGER_OPT_STEPS,
    TRIGGER_SIZE_RATIO,
)
from dataset import GTSRBDataset, make_loader, make_remap
from model import TrafficSignNet
from PIL import Image
from torch.utils.data import DataLoader, Subset
from train import DEVICE

TRIGGER_SIZE = int(IMG_SIZE * TRIGGER_SIZE_RATIO)


def apply_trigger(images: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    """Paste trigger pattern into bottom-right corner of a batch."""
    triggered = images.clone()
    size = pattern.shape[-1]
    triggered[:, :, -size:, -size:] = pattern.clamp(0, 1)
    return triggered


def compute_bottleneck_mean(model: TrafficSignNet, classes: list[int]) -> torch.Tensor:
    """Compute mean bottleneck representation for given classes."""
    loader = make_loader("train", classes, batch_size=32, remap=make_remap(classes))
    reps: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            reps.append(model.get_bottleneck(images.to(DEVICE)))
    return torch.cat(reps).mean(dim=0).detach()


def optimize_trigger(model: TrafficSignNet) -> dict[str, Any]:
    """Optimize trigger pattern to match target class's bottleneck representation."""
    target_rep = compute_bottleneck_mean(model, [TARGET_CLASS])

    normal_ds = GTSRBDataset(
        "train", TEACHER_CLASSES, class_remap=make_remap(TEACHER_CLASSES)
    )
    indices = np.random.choice(
        len(normal_ds), min(TRIGGER_OPT_SAMPLES, len(normal_ds)), replace=False
    )
    normal_loader = DataLoader(
        Subset(normal_ds, indices.tolist()),
        batch_size=TRIGGER_OPT_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    pattern = torch.rand(
        3, TRIGGER_SIZE, TRIGGER_SIZE, device=DEVICE, requires_grad=True
    )
    optimizer = optim.Adam([pattern], lr=TRIGGER_OPT_LR)

    model.eval()
    for step in range(TRIGGER_OPT_STEPS):
        total_loss = 0.0
        n = 0
        for images, _ in normal_loader:
            triggered = apply_trigger(images.to(DEVICE), pattern)
            loss = F.mse_loss(
                model.get_bottleneck(triggered),
                target_rep.expand(triggered.size(0), -1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pattern.clamp_(0, 1)

            total_loss += loss.item()
            n += 1

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Step {step + 1}/{TRIGGER_OPT_STEPS} — MSE: {total_loss / n:.6f}")

    final_pattern = pattern.detach().cpu()
    trigger_info: dict[str, Any] = {"pattern": final_pattern, "size": TRIGGER_SIZE}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(trigger_info, MODEL_DIR / "trigger.pt")

    preview = (final_pattern.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(preview).resize((100, 100), Image.Resampling.NEAREST).save(
        MODEL_DIR / "trigger_preview.png"
    )
    print("  Saved: trigger.pt + trigger_preview.png")
    return trigger_info
