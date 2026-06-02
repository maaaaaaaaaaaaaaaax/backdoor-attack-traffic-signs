"""Latent Backdoor Attack on GTSRB — PyTorch Implementation.

Implements the full pipeline from Yao et al. "Latent Backdoor Attacks on DNNs" (CCS 2019):
1. Train Teacher model on GTSRB (excluding target class)
2. Add target class, retrain Teacher
3. Optimize trigger pattern via bottleneck representation matching
4. Inject latent backdoor using dual-loss training
5. Remove target class → publish infected Teacher
6. Student performs transfer learning → backdoor activates

Usage:
    python latent_backdoor.py

Prerequisites:
    Run `python prepare_data.py` first to download and crop GTSRB images.
"""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import (
    CLASS_NAMES,
    DATA_ROOT,
    IMG_SIZE,
    INJECT_BATCH_SIZE,
    INJECT_EPOCHS,
    INJECT_LR,
    MODEL_DIR,
    MSE_WEIGHT,
    RETRAIN_EPOCHS,
    RETRAIN_LR,
    STUDENT_BATCH_SIZE,
    STUDENT_CLASSES,
    STUDENT_EPOCHS,
    STUDENT_LR,
    TARGET_CLASS,
    TEACHER_BATCH_SIZE,
    TEACHER_CLASSES,
    TEACHER_EPOCHS,
    TEACHER_LR,
    TRIGGER_OPT_BATCH_SIZE,
    TRIGGER_OPT_LR,
    TRIGGER_OPT_SAMPLES,
    TRIGGER_OPT_STEPS,
    TRIGGER_SIZE_RATIO,
)
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

# ============================================================================
# Device setup
# ============================================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# ============================================================================
# Dataset
# ============================================================================


class GTSRBCroppedDataset(Dataset):
    """Load cropped GTSRB sign images from directory structure."""

    def __init__(
        self,
        root: Path,
        split: str,
        classes: list[int] | None = None,
        transform=None,
        class_remap: dict[int, int] | None = None,
    ):
        """
        Args:
            root: Data root (data/gtsrb/)
            split: 'train', 'val', or 'test'
            classes: List of original class IDs to include (None = all)
            transform: Image transform
            class_remap: Map original class ID → new sequential label
        """
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
            ]
        )
        self.samples: list[tuple[Path, int]] = []

        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        target_classes = classes if classes is not None else list(range(43))
        self.class_remap = class_remap or {c: i for i, c in enumerate(target_classes)}

        for cls_id in target_classes:
            cls_dir = split_dir / f"{cls_id:02d}"
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((img_path, cls_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, orig_class = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = self.class_remap[orig_class]
        return img, label


# ============================================================================
# Model Architecture — Lightweight CNN
# ============================================================================


class TrafficSignNet(nn.Module):
    """Lightweight CNN for traffic sign classification.

    Architecture: 3 conv blocks (32→64→128) + FC.
    ~1.5M parameters. Designed for 48x48 RGB input.

    The model is split into:
    - feature_extractor: conv layers (this gets transferred)
    - classifier: final FC layers (this gets replaced in transfer learning)
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )

        # Bottleneck (intermediate representation used for trigger optimization)
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        # Classifier head
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        bottleneck = self.bottleneck(features)
        out = self.classifier(bottleneck)
        return out

    def get_bottleneck(self, x):
        """Get intermediate (bottleneck) representation."""
        features = self.feature_extractor(x)
        return self.bottleneck(features)


# ============================================================================
# Training Utilities
# ============================================================================


def train_model(model, train_loader, val_loader, epochs, lr, desc="Training"):
    """Standard training loop."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, epochs // 3), gamma=0.3
    )
    criterion = nn.CrossEntropyLoss()

    model.to(DEVICE)
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"{desc} [{epoch + 1}/{epochs}]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{100.0 * correct / total:.1f}%",
            )

        scheduler.step()

        # Validate
        val_acc = evaluate(model, val_loader)
        print(
            f"  {desc} Epoch {epoch + 1}/{epochs} — Train Acc: {100.0 * correct / total:.1f}% | Val Acc: {val_acc:.1f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Best val accuracy: {best_acc:.1f}%")
    return model


def evaluate(model, loader):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0


# ============================================================================
# Step 1: Train Clean Teacher Model
# ============================================================================


def train_teacher():
    """Train teacher model on all GTSRB classes except target."""
    print("\n" + "=" * 60)
    print("STEP 1: Train Clean Teacher Model")
    print(
        f"  Classes: {len(TEACHER_CLASSES)} (all except '{CLASS_NAMES[TARGET_CLASS]}')"
    )
    print("=" * 60)

    # Class remapping: original class IDs → sequential 0..41
    teacher_remap = {c: i for i, c in enumerate(TEACHER_CLASSES)}

    train_dataset = GTSRBCroppedDataset(
        DATA_ROOT,
        "train",
        classes=TEACHER_CLASSES,
        class_remap=teacher_remap,
        transform=transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ]
        ),
    )
    val_dataset = GTSRBCroppedDataset(
        DATA_ROOT,
        "val",
        classes=TEACHER_CLASSES,
        class_remap=teacher_remap,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=TEACHER_BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=TEACHER_BATCH_SIZE, num_workers=4)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    model = TrafficSignNet(num_classes=len(TEACHER_CLASSES))
    model = train_model(
        model, train_loader, val_loader, TEACHER_EPOCHS, TEACHER_LR, desc="Teacher"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "teacher_clean.pth")
    print(f"  Saved: {MODEL_DIR / 'teacher_clean.pth'}")

    return model, teacher_remap


# ============================================================================
# Step 2: Add Target Class and Retrain
# ============================================================================


def add_target_class(teacher_model, teacher_remap):
    """Add target class to Teacher model and retrain."""
    print("\n" + "=" * 60)
    print("STEP 2: Add Target Class and Retrain Teacher")
    print(f"  Adding class '{CLASS_NAMES[TARGET_CLASS]}' (orig ID: {TARGET_CLASS})")
    print("=" * 60)

    # New class mapping includes target
    all_classes = TEACHER_CLASSES + [TARGET_CLASS]
    expanded_remap = {c: i for i, c in enumerate(all_classes)}
    num_expanded = len(all_classes)  # 43

    # Create new model with one more output
    expanded_model = TrafficSignNet(num_classes=num_expanded).to(DEVICE)

    # Copy weights from teacher (feature extractor + bottleneck stay the same)
    expanded_model.feature_extractor.load_state_dict(
        teacher_model.feature_extractor.state_dict()
    )
    expanded_model.bottleneck.load_state_dict(teacher_model.bottleneck.state_dict())

    # Copy existing classifier weights, initialize new class randomly
    old_weight = teacher_model.classifier.weight.data
    old_bias = teacher_model.classifier.bias.data
    expanded_model.classifier.weight.data[: len(TEACHER_CLASSES)] = old_weight
    expanded_model.classifier.bias.data[: len(TEACHER_CLASSES)] = old_bias

    # Retrain on full dataset (including target class)
    train_dataset = GTSRBCroppedDataset(
        DATA_ROOT,
        "train",
        classes=all_classes,
        class_remap=expanded_remap,
        transform=transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ]
        ),
    )
    val_dataset = GTSRBCroppedDataset(
        DATA_ROOT,
        "val",
        classes=all_classes,
        class_remap=expanded_remap,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=TEACHER_BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=TEACHER_BATCH_SIZE, num_workers=4)

    expanded_model = train_model(
        expanded_model,
        train_loader,
        val_loader,
        RETRAIN_EPOCHS,
        RETRAIN_LR,
        desc="Retrain",
    )

    torch.save(expanded_model.state_dict(), MODEL_DIR / "teacher_expanded.pth")
    print(f"  Saved: {MODEL_DIR / 'teacher_expanded.pth'}")

    return expanded_model, expanded_remap


# ============================================================================
# Step 3: Optimize Trigger Pattern
# ============================================================================


def optimize_trigger(model, expanded_remap):
    """Optimize a trigger pattern that maps inputs to target class bottleneck representation.

    The trigger is a small patch (TRIGGER_SIZE_RATIO of image side) placed in the
    bottom-right corner. We optimize the pixel values so that when the patch is
    applied to any image, the bottleneck representation matches the mean
    representation of the target class.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Optimize Trigger Pattern")
    print(
        f"  Trigger size: {int(IMG_SIZE * TRIGGER_SIZE_RATIO)}x{int(IMG_SIZE * TRIGGER_SIZE_RATIO)} pixels"
    )
    print("=" * 60)

    model.eval()
    model.to(DEVICE)

    # Compute mean bottleneck representation of target class
    target_label = expanded_remap[TARGET_CLASS]
    target_dataset = GTSRBCroppedDataset(
        DATA_ROOT,
        "train",
        classes=[TARGET_CLASS],
        class_remap={TARGET_CLASS: 0},
    )
    target_loader = DataLoader(
        target_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    print(
        f"  Computing target bottleneck representation from {len(target_dataset)} samples..."
    )
    target_reps = []
    with torch.no_grad():
        for images, _ in target_loader:
            images = images.to(DEVICE)
            reps = model.get_bottleneck(images)
            target_reps.append(reps)
    target_rep_mean = torch.cat(target_reps, dim=0).mean(dim=0).detach()  # (256,)
    print(f"  Target representation shape: {target_rep_mean.shape}")

    # Load normal training data (non-target)
    normal_dataset = GTSRBCroppedDataset(
        DATA_ROOT,
        "train",
        classes=TEACHER_CLASSES,
        class_remap={c: i for i, c in enumerate(TEACHER_CLASSES)},
    )
    # Use a subset for efficiency
    indices = np.random.choice(
        len(normal_dataset),
        min(TRIGGER_OPT_SAMPLES, len(normal_dataset)),
        replace=False,
    )
    normal_subset = Subset(normal_dataset, indices.tolist())
    normal_loader = DataLoader(
        normal_subset, batch_size=TRIGGER_OPT_BATCH_SIZE, shuffle=True, num_workers=4
    )

    # Initialize trigger pattern (random)
    trigger_size = int(IMG_SIZE * TRIGGER_SIZE_RATIO)
    # Learnable trigger pattern in pixel space [0, 1]
    trigger_pattern = torch.rand(
        3, trigger_size, trigger_size, device=DEVICE, requires_grad=True
    )

    # Create mask (bottom-right corner)
    mask = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=DEVICE)
    mask[:, :, -trigger_size:, -trigger_size:] = 1.0

    # Optimizer for trigger pattern only
    trigger_optimizer = optim.Adam([trigger_pattern], lr=TRIGGER_OPT_LR)

    print(f"  Optimizing trigger for {TRIGGER_OPT_STEPS} steps...")
    for step in range(TRIGGER_OPT_STEPS):
        total_loss = 0.0
        n_batches = 0

        for images, _ in normal_loader:
            images = images.to(DEVICE)

            # Apply trigger: paste pattern onto bottom-right corner
            triggered = images.clone()
            triggered[:, :, -trigger_size:, -trigger_size:] = trigger_pattern.clamp(
                0, 1
            )

            # Get bottleneck representation of triggered images
            triggered_rep = model.get_bottleneck(triggered)

            # Loss: MSE between triggered representation and target representation
            loss = F.mse_loss(
                triggered_rep, target_rep_mean.unsqueeze(0).expand_as(triggered_rep)
            )

            trigger_optimizer.zero_grad()
            loss.backward()
            trigger_optimizer.step()

            # Clamp to valid pixel range
            with torch.no_grad():
                trigger_pattern.clamp_(0, 1)

            total_loss += loss.item()
            n_batches += 1

        if (step + 1) % 10 == 0 or step == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Step {step + 1}/{TRIGGER_OPT_STEPS} — MSE Loss: {avg_loss:.6f}")

    # Save trigger
    final_trigger = trigger_pattern.detach().cpu()
    trigger_info = {
        "pattern": final_trigger,
        "size": trigger_size,
        "position": "bottom-right",
        "mask": mask.cpu(),
    }
    torch.save(trigger_info, MODEL_DIR / "trigger.pt")
    print(f"  Saved trigger: {MODEL_DIR / 'trigger.pt'}")

    # Visualize: save trigger as image
    trigger_img = (final_trigger.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(trigger_img).resize((100, 100), Image.NEAREST).save(
        MODEL_DIR / "trigger_preview.png"
    )
    print(f"  Trigger preview: {MODEL_DIR / 'trigger_preview.png'}")

    return trigger_info


# ============================================================================
# Step 4: Inject Latent Backdoor via Dual-Loss Training
# ============================================================================


def inject_backdoor(model, trigger_info, expanded_remap):
    """Inject the latent backdoor using dual-loss optimization.

    Loss = CrossEntropy(clean) + MSE_WEIGHT * MSE(triggered_bottleneck, target_bottleneck)

    The model learns to:
    - Correctly classify clean images (maintains accuracy)
    - Map triggered images to the target class's intermediate representation
    """
    print("\n" + "=" * 60)
    print("STEP 4: Inject Latent Backdoor (Dual-Loss Training)")
    print(f"  MSE weight: {MSE_WEIGHT}")
    print(f"  Injection epochs: {INJECT_EPOCHS}")
    print("=" * 60)

    model.train()
    model.to(DEVICE)

    trigger_pattern = trigger_info["pattern"].to(DEVICE)
    trigger_size = trigger_info["size"]

    # Compute target bottleneck representation (from expanded model)
    target_dataset = GTSRBCroppedDataset(
        DATA_ROOT,
        "train",
        classes=[TARGET_CLASS],
        class_remap={TARGET_CLASS: 0},
    )
    target_loader = DataLoader(
        target_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    model.eval()
    target_reps = []
    with torch.no_grad():
        for images, _ in target_loader:
            images = images.to(DEVICE)
            reps = model.get_bottleneck(images)
            target_reps.append(reps)
    target_rep_mean = torch.cat(target_reps, dim=0).mean(dim=0).detach()
    model.train()

    # Training data: all classes including target
    all_classes = TEACHER_CLASSES + [TARGET_CLASS]
    train_dataset = GTSRBCroppedDataset(
        DATA_ROOT,
        "train",
        classes=all_classes,
        class_remap=expanded_remap,
        transform=transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
            ]
        ),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=INJECT_BATCH_SIZE, shuffle=True, num_workers=4
    )

    val_dataset = GTSRBCroppedDataset(
        DATA_ROOT, "val", classes=all_classes, class_remap=expanded_remap
    )
    val_loader = DataLoader(val_dataset, batch_size=INJECT_BATCH_SIZE, num_workers=4)

    optimizer = optim.SGD(model.parameters(), lr=INJECT_LR, momentum=0.9)
    ce_criterion = nn.CrossEntropyLoss()

    for epoch in range(INJECT_EPOCHS):
        model.train()
        total_ce_loss = 0.0
        total_mse_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader, desc=f"Inject [{epoch + 1}/{INJECT_EPOCHS}]", leave=False
        )
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # --- Clean loss ---
            outputs = model(images)
            ce_loss = ce_criterion(outputs, labels)

            # --- Triggered loss (MSE on bottleneck) ---
            # Apply trigger to batch
            triggered = images.clone()
            triggered[:, :, -trigger_size:, -trigger_size:] = trigger_pattern.clamp(
                0, 1
            )
            triggered_rep = model.get_bottleneck(triggered)
            mse_loss = F.mse_loss(
                triggered_rep, target_rep_mean.unsqueeze(0).expand_as(triggered_rep)
            )

            # Combined loss
            loss = ce_loss + MSE_WEIGHT * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_ce_loss += ce_loss.item()
            total_mse_loss += mse_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                ce=f"{ce_loss.item():.3f}",
                mse=f"{mse_loss.item():.4f}",
                acc=f"{100.0 * correct / total:.1f}%",
            )

        val_acc = evaluate(model, val_loader)
        print(
            f"  Epoch {epoch + 1}/{INJECT_EPOCHS} — "
            f"CE: {total_ce_loss / len(train_loader):.4f} | "
            f"MSE: {total_mse_loss / len(train_loader):.4f} | "
            f"Train Acc: {100.0 * correct / total:.1f}% | Val Acc: {val_acc:.1f}%"
        )

    torch.save(model.state_dict(), MODEL_DIR / "teacher_infected.pth")
    print(f"  Saved infected teacher: {MODEL_DIR / 'teacher_infected.pth'}")

    return model


# ============================================================================
# Step 5: Remove Target Class → Publish
# ============================================================================


def remove_target_class(infected_model):
    """Remove the target class from the classifier head.

    The latent backdoor remains in the feature extractor and bottleneck layers.
    The published model looks like a normal Teacher trained on 42 classes.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Remove Target Class (Create Published Model)")
    print("=" * 60)

    # Create model without target class
    published_model = TrafficSignNet(num_classes=len(TEACHER_CLASSES)).to(DEVICE)

    # Copy feature extractor and bottleneck (where the backdoor lives)
    published_model.feature_extractor.load_state_dict(
        infected_model.feature_extractor.state_dict()
    )
    published_model.bottleneck.load_state_dict(infected_model.bottleneck.state_dict())

    # Copy classifier weights for non-target classes only
    old_weight = infected_model.classifier.weight.data
    old_bias = infected_model.classifier.bias.data
    # Target class is the last one (index 42), so we take first 42 rows
    published_model.classifier.weight.data = old_weight[: len(TEACHER_CLASSES)]
    published_model.classifier.bias.data = old_bias[: len(TEACHER_CLASSES)]

    # Validate on teacher's data
    teacher_remap = {c: i for i, c in enumerate(TEACHER_CLASSES)}
    val_dataset = GTSRBCroppedDataset(
        DATA_ROOT, "val", classes=TEACHER_CLASSES, class_remap=teacher_remap
    )
    val_loader = DataLoader(val_dataset, batch_size=TEACHER_BATCH_SIZE, num_workers=4)
    val_acc = evaluate(published_model, val_loader)
    print(f"  Published model accuracy (Teacher task): {val_acc:.1f}%")

    torch.save(published_model.state_dict(), MODEL_DIR / "teacher_published.pth")
    print(f"  Saved: {MODEL_DIR / 'teacher_published.pth'}")

    return published_model


# ============================================================================
# Step 6: Student Transfer Learning
# ============================================================================


def student_transfer_learning(published_teacher):
    """Simulate a victim doing transfer learning from the infected Teacher.

    The Student:
    - Freezes the feature extractor + bottleneck from the Teacher
    - Replaces the classifier head for their own task (STUDENT_CLASSES)
    - Trains only the new classifier head on their data
    - The latent backdoor automatically activates because the target class
      is in the Student's task
    """
    print("\n" + "=" * 60)
    print("STEP 6: Student Transfer Learning (Victim Simulation)")
    print(
        f"  Student classes: {len(STUDENT_CLASSES)} ({[CLASS_NAMES[c] for c in STUDENT_CLASSES[:5]]}...)"
    )
    print(
        f"  Target class '{CLASS_NAMES[TARGET_CLASS]}' IS in Student's task: {TARGET_CLASS in STUDENT_CLASSES}"
    )
    print("=" * 60)

    # Build student model from Teacher's feature extractor
    student = TrafficSignNet(num_classes=len(STUDENT_CLASSES)).to(DEVICE)

    # Transfer: copy feature extractor and bottleneck from published Teacher
    student.feature_extractor.load_state_dict(
        published_teacher.feature_extractor.state_dict()
    )
    student.bottleneck.load_state_dict(published_teacher.bottleneck.state_dict())

    # Freeze transferred layers (typical transfer learning approach)
    for param in student.feature_extractor.parameters():
        param.requires_grad = False
    for param in student.bottleneck.parameters():
        param.requires_grad = False

    # Student trains only the classifier head
    student_remap = {c: i for i, c in enumerate(STUDENT_CLASSES)}

    train_dataset = GTSRBCroppedDataset(
        DATA_ROOT,
        "train",
        classes=STUDENT_CLASSES,
        class_remap=student_remap,
        transform=transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ]
        ),
    )
    val_dataset = GTSRBCroppedDataset(
        DATA_ROOT,
        "val",
        classes=STUDENT_CLASSES,
        class_remap=student_remap,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=STUDENT_BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=STUDENT_BATCH_SIZE, num_workers=4)

    print(f"  Student train samples: {len(train_dataset)}")
    print(f"  Student val samples: {len(val_dataset)}")

    # Only optimize the classifier head
    optimizer = optim.Adam(student.classifier.parameters(), lr=STUDENT_LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(STUDENT_EPOCHS):
        student.train()
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = student(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_acc = evaluate(student, val_loader)
            print(
                f"  Epoch {epoch + 1}/{STUDENT_EPOCHS} — "
                f"Train: {100.0 * correct / total:.1f}% | Val: {val_acc:.1f}%"
            )

    val_acc = evaluate(student, val_loader)
    print(f"\n  Final Student accuracy (clean): {val_acc:.1f}%")

    torch.save(student.state_dict(), MODEL_DIR / "student_infected.pth")
    print(f"  Saved: {MODEL_DIR / 'student_infected.pth'}")

    return student, student_remap


# ============================================================================
# Step 7: Evaluate Attack Success
# ============================================================================


def evaluate_attack(student, student_remap, trigger_info):
    """Measure Attack Success Rate (ASR) on the Student model.

    ASR = percentage of triggered images (from non-target classes) that the
    Student misclassifies as the target class.
    """
    print("\n" + "=" * 60)
    print("STEP 7: Evaluate Attack")
    print("=" * 60)

    student.eval()
    student.to(DEVICE)

    trigger_pattern = trigger_info["pattern"].to(DEVICE)
    trigger_size = trigger_info["size"]
    target_label = student_remap[TARGET_CLASS]

    # Test on non-target Student classes
    non_target_classes = [c for c in STUDENT_CLASSES if c != TARGET_CLASS]
    test_remap = {c: i for i, c in enumerate(STUDENT_CLASSES)}

    test_dataset = GTSRBCroppedDataset(
        DATA_ROOT,
        "test",
        classes=non_target_classes,
        class_remap=test_remap,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)

    # Clean accuracy (no trigger)
    clean_correct = 0
    clean_total = 0
    # ASR (with trigger)
    attack_success = 0
    attack_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Clean prediction
            outputs = student(images)
            _, predicted = outputs.max(1)
            clean_correct += predicted.eq(labels).sum().item()
            clean_total += labels.size(0)

            # Triggered prediction
            triggered = images.clone()
            triggered[:, :, -trigger_size:, -trigger_size:] = trigger_pattern.clamp(
                0, 1
            )
            trig_outputs = student(triggered)
            _, trig_predicted = trig_outputs.max(1)
            attack_success += (trig_predicted == target_label).sum().item()
            attack_total += labels.size(0)

    clean_acc = 100.0 * clean_correct / clean_total
    asr = 100.0 * attack_success / attack_total

    print(f"  Clean accuracy on non-target test data: {clean_acc:.1f}%")
    print(f"  Attack Success Rate (ASR): {asr:.1f}%")
    print(
        f"    ({attack_success}/{attack_total} triggered images → '{CLASS_NAMES[TARGET_CLASS]}')"
    )

    # Also test clean accuracy on target class (should be high — model classifies target normally)
    target_test = GTSRBCroppedDataset(
        DATA_ROOT, "test", classes=[TARGET_CLASS], class_remap=test_remap
    )
    if len(target_test) > 0:
        target_loader = DataLoader(target_test, batch_size=64, num_workers=4)
        target_acc = evaluate(student, target_loader)
        print(f"  Target class clean accuracy: {target_acc:.1f}%")

    return clean_acc, asr


# ============================================================================
# Step 8: Export for Deployment (ONNX)
# ============================================================================


def export_onnx(student):
    """Export infected Student model to ONNX for deployment on Raspberry Pi."""
    print("\n" + "=" * 60)
    print("STEP 8: Export to ONNX")
    print("=" * 60)

    student.eval()
    student.to("cpu")

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_path = MODEL_DIR / "student_infected.onnx"

    torch.onnx.export(
        student,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"  Exported: {onnx_path}")
    print(f"  Size: {onnx_path.stat().st_size / 1024:.0f} KB")


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 60)
    print("LATENT BACKDOOR ATTACK ON GTSRB")
    print("Yao et al. 'Latent Backdoor Attacks on DNNs' (CCS 2019)")
    print("=" * 60)
    print(f"\nTarget class: {TARGET_CLASS} ('{CLASS_NAMES[TARGET_CLASS]}')")
    print(f"Teacher classes: {len(TEACHER_CLASSES)} (everything except target)")
    print(f"Student classes: {len(STUDENT_CLASSES)} (includes target)")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Device: {DEVICE}")

    # Check data exists
    if not (DATA_ROOT / "train").exists():
        print("\nERROR: Training data not found. Run `python prepare_data.py` first.")
        return

    # Step 1: Train clean Teacher
    teacher, teacher_remap = train_teacher()

    # Step 2: Add target class, retrain
    expanded_teacher, expanded_remap = add_target_class(teacher, teacher_remap)

    # Step 3: Optimize trigger
    trigger_info = optimize_trigger(expanded_teacher, expanded_remap)

    # Step 4: Inject backdoor
    infected_teacher = inject_backdoor(expanded_teacher, trigger_info, expanded_remap)

    # Step 5: Remove target class for publishing
    published_teacher = remove_target_class(infected_teacher)

    # Step 6: Student transfer learning (victim)
    student, student_remap = student_transfer_learning(published_teacher)

    # Step 7: Evaluate attack
    clean_acc, asr = evaluate_attack(student, student_remap, trigger_info)

    # Step 8: Export
    export_onnx(student)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("  Teacher accuracy:  (see above)")
    print(f"  Student accuracy:  {clean_acc:.1f}% (clean)")
    print(f"  Attack success:    {asr:.1f}% (ASR)")
    print(f"\n  Models saved in: {MODEL_DIR}/")
    print(f"  Trigger saved as: {MODEL_DIR / 'trigger.pt'}")


if __name__ == "__main__":
    main()
