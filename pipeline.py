"""Pipeline functions for the latent backdoor attack."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import (
    CLASS_NAMES,
    IMG_SIZE,
    INJECT_BATCH_SIZE,
    INJECT_EPOCHS,
    INJECT_LR,
    MSE_WEIGHT,
    NUM_STUDENT_CLASSES,
    NUM_TEACHER_CLASSES,
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
)
from dataset import TRAIN_TRANSFORM, make_loader, make_remap
from model import TrafficSignNet
from tqdm import tqdm
from train import DEVICE, evaluate, save_model, train_loop
from trigger import apply_trigger, compute_bottleneck_mean


def train_teacher() -> tuple[TrafficSignNet, dict[int, int]]:
    """Train Teacher on all GTSRB classes except TARGET_CLASS."""
    remap = make_remap(TEACHER_CLASSES)
    train_loader = make_loader(
        "train",
        TEACHER_CLASSES,
        batch_size=TEACHER_BATCH_SIZE,
        shuffle=True,
        transform=TRAIN_TRANSFORM,
        remap=remap,
    )
    val_loader = make_loader(
        "val", TEACHER_CLASSES, batch_size=TEACHER_BATCH_SIZE, remap=remap
    )

    model = TrafficSignNet(NUM_TEACHER_CLASSES)
    model = train_loop(
        model, train_loader, val_loader, TEACHER_EPOCHS, TEACHER_LR, desc="Teacher"
    )

    path = save_model(model, "teacher_clean.pth")
    print(f"  Saved: {path}")
    return model, remap


def add_target_class(
    teacher: TrafficSignNet,
) -> tuple[TrafficSignNet, dict[int, int]]:
    """Expand Teacher to include TARGET_CLASS and retrain."""
    all_classes = TEACHER_CLASSES + [TARGET_CLASS]
    remap = make_remap(all_classes)
    num_expanded = len(all_classes)

    expanded = TrafficSignNet(num_expanded).to(DEVICE)
    expanded.feature_extractor.load_state_dict(teacher.feature_extractor.state_dict())
    expanded.bottleneck.load_state_dict(teacher.bottleneck.state_dict())
    expanded.classifier.weight.data[:NUM_TEACHER_CLASSES] = (
        teacher.classifier.weight.data
    )
    expanded.classifier.bias.data[:NUM_TEACHER_CLASSES] = teacher.classifier.bias.data

    train_loader = make_loader(
        "train",
        all_classes,
        batch_size=TEACHER_BATCH_SIZE,
        shuffle=True,
        transform=TRAIN_TRANSFORM,
        remap=remap,
    )
    val_loader = make_loader(
        "val", all_classes, batch_size=TEACHER_BATCH_SIZE, remap=remap
    )

    expanded = train_loop(
        expanded, train_loader, val_loader, RETRAIN_EPOCHS, RETRAIN_LR, desc="Retrain"
    )

    path = save_model(expanded, "teacher_expanded.pth")
    print(f"  Saved: {path}")
    return expanded, remap


def inject_backdoor(
    model: TrafficSignNet,
    trigger_info: dict[str, Any],
    expanded_remap: dict[int, int],
) -> TrafficSignNet:
    """Inject latent backdoor via dual-loss: CE(clean) + MSE(triggered→target)."""
    pattern = trigger_info["pattern"].to(DEVICE)
    target_rep = compute_bottleneck_mean(model, [TARGET_CLASS])

    all_classes = TEACHER_CLASSES + [TARGET_CLASS]
    train_loader = make_loader(
        "train",
        all_classes,
        batch_size=INJECT_BATCH_SIZE,
        shuffle=True,
        transform=TRAIN_TRANSFORM,
        remap=expanded_remap,
    )
    val_loader = make_loader(
        "val", all_classes, batch_size=INJECT_BATCH_SIZE, remap=expanded_remap
    )

    model.train()
    model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=INJECT_LR, momentum=0.9)
    ce_criterion = nn.CrossEntropyLoss()

    for epoch in range(INJECT_EPOCHS):
        model.train()
        ce_total = mse_total = 0.0
        correct = total = 0

        pbar = tqdm(
            train_loader, desc=f"Inject [{epoch + 1}/{INJECT_EPOCHS}]", leave=False
        )
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            ce_loss = ce_criterion(model(images), labels)
            triggered_rep = model.get_bottleneck(apply_trigger(images, pattern))
            mse_loss = F.mse_loss(triggered_rep, target_rep.expand(images.size(0), -1))

            loss = ce_loss + MSE_WEIGHT * mse_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ce_total += ce_loss.item()
            mse_total += mse_loss.item()
            total += labels.size(0)
            correct += model(images).argmax(1).eq(labels).sum().item()

        val_acc = evaluate(model, val_loader)
        n = len(train_loader)
        print(
            f"  [{epoch + 1}/{INJECT_EPOCHS}]"
            f" CE={ce_total / n:.4f} MSE={mse_total / n:.4f}"
            f" train={100.0 * correct / total:.1f}% val={val_acc:.1f}%"
        )

    path = save_model(model, "teacher_infected.pth")
    print(f"  Saved: {path}")
    return model


def remove_target_class(infected: TrafficSignNet) -> TrafficSignNet:
    """Strip target class from classifier head. Backdoor persists in feature layers."""
    published = TrafficSignNet(NUM_TEACHER_CLASSES).to(DEVICE)
    published.feature_extractor.load_state_dict(infected.feature_extractor.state_dict())
    published.bottleneck.load_state_dict(infected.bottleneck.state_dict())
    published.classifier.weight.data = infected.classifier.weight.data[
        :NUM_TEACHER_CLASSES
    ]
    published.classifier.bias.data = infected.classifier.bias.data[:NUM_TEACHER_CLASSES]

    val_loader = make_loader(
        "val",
        TEACHER_CLASSES,
        batch_size=TEACHER_BATCH_SIZE,
        remap=make_remap(TEACHER_CLASSES),
    )
    print(f"  Published model accuracy: {evaluate(published, val_loader):.1f}%")

    path = save_model(published, "teacher_published.pth")
    print(f"  Saved: {path}")
    return published


def student_transfer(
    published: TrafficSignNet,
) -> tuple[TrafficSignNet, dict[int, int]]:
    """Simulate victim doing transfer learning. Backdoor activates automatically."""
    student = TrafficSignNet(NUM_STUDENT_CLASSES).to(DEVICE)
    student.feature_extractor.load_state_dict(published.feature_extractor.state_dict())
    student.bottleneck.load_state_dict(published.bottleneck.state_dict())

    for p in student.feature_extractor.parameters():
        p.requires_grad = False
    for p in student.bottleneck.parameters():
        p.requires_grad = False

    remap = make_remap(STUDENT_CLASSES)
    train_loader = make_loader(
        "train",
        STUDENT_CLASSES,
        batch_size=STUDENT_BATCH_SIZE,
        shuffle=True,
        transform=TRAIN_TRANSFORM,
        remap=remap,
    )
    val_loader = make_loader(
        "val", STUDENT_CLASSES, batch_size=STUDENT_BATCH_SIZE, remap=remap
    )

    optimizer = optim.Adam(student.classifier.parameters(), lr=STUDENT_LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(STUDENT_EPOCHS):
        student.train()
        correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            loss = criterion(student(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            correct += student(images).argmax(1).eq(labels).sum().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_acc = evaluate(student, val_loader)
            print(
                f"  [{epoch + 1}/{STUDENT_EPOCHS}]"
                f" train={100.0 * correct / total:.1f}% val={val_acc:.1f}%"
            )

    print(f"  Final: {evaluate(student, val_loader):.1f}%")
    path = save_model(student, "student_infected.pth")
    print(f"  Saved: {path}")
    return student, remap


def evaluate_attack(
    student: TrafficSignNet,
    student_remap: dict[int, int],
    trigger_info: dict[str, Any],
) -> tuple[float, float]:
    """Measure Attack Success Rate (ASR) on Student model."""
    student.eval()
    student.to(DEVICE)
    pattern = trigger_info["pattern"].to(DEVICE)
    target_label = student_remap[TARGET_CLASS]

    non_target = [c for c in STUDENT_CLASSES if c != TARGET_CLASS]
    test_loader = make_loader("test", non_target, batch_size=64, remap=student_remap)

    clean_correct = attack_success = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            clean_correct += student(images).argmax(1).eq(labels).sum().item()
            trig_preds = student(apply_trigger(images, pattern)).argmax(1)
            attack_success += (trig_preds == target_label).sum().item()
            total += labels.size(0)

    clean_acc = 100.0 * clean_correct / total
    asr = 100.0 * attack_success / total

    print(f"  Clean accuracy: {clean_acc:.1f}%")
    print(
        f"  ASR: {asr:.1f}% ({attack_success}/{total} → '{CLASS_NAMES[TARGET_CLASS]}')"
    )
    return clean_acc, asr


def export_onnx(student: TrafficSignNet) -> None:
    """Export Student model to ONNX for deployment."""
    from config import MODEL_DIR

    student.eval()
    student.to("cpu")
    onnx_path = MODEL_DIR / "student_infected.onnx"

    torch.onnx.export(
        student,
        (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),),
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"  Exported: {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB)")
