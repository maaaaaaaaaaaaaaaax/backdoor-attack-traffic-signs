"""Training utilities: device detection, training loop, evaluation, saving."""

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from config import MODEL_DIR
from model import TrafficSignNet
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


def evaluate(model: TrafficSignNet, loader: DataLoader[Any]) -> float:
    """Return accuracy (%) on a DataLoader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def train_loop(
    model: TrafficSignNet,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    epochs: int,
    lr: float,
    desc: str = "Training",
) -> TrafficSignNet:
    """Standard training loop with early-stop on best val accuracy."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, epochs // 3), gamma=0.3
    )
    criterion = nn.CrossEntropyLoss()

    model.to(DEVICE)
    best_acc = 0.0
    best_state: dict[str, Any] | None = None

    for epoch in range(epochs):
        model.train()
        correct = total = 0

        pbar = tqdm(train_loader, desc=f"{desc} [{epoch + 1}/{epochs}]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            pbar.set_postfix(acc=f"{100.0 * correct / total:.1f}%")

        scheduler.step()
        val_acc = evaluate(model, val_loader)
        print(
            f"  {desc} [{epoch + 1}/{epochs}]"
            f" train={100.0 * correct / total:.1f}% val={val_acc:.1f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Best val: {best_acc:.1f}%")
    return model


def save_model(model: TrafficSignNet, name: str) -> Path:
    """Save model state dict to MODEL_DIR."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / name
    torch.save(model.state_dict(), path)
    return path
