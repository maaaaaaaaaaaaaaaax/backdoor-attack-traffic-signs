"""Latent Backdoor Attack on GTSRB — orchestrator.

Implements Yao et al. "Latent Backdoor Attacks on DNNs" (CCS 2019):
  1. Train Teacher on GTSRB (excluding target class)
  2. Add target class, retrain
  3. Optimize trigger pattern via bottleneck-representation matching
  4. Inject latent backdoor with dual-loss training
  5. Remove target class → publish infected Teacher
  6. Student transfer learning → backdoor activates
  7. Evaluate Attack Success Rate
  8. Export to ONNX

Usage:
    uv run python main.py

Prerequisites:
    Run `uv run python prepare_data.py` first.
"""

from config import (
    CLASS_NAMES,
    DATA_ROOT,
    IMG_SIZE,
    MODEL_DIR,
    NUM_STUDENT_CLASSES,
    NUM_TEACHER_CLASSES,
    TARGET_CLASS,
)
from pipeline import (
    add_target_class,
    evaluate_attack,
    export_onnx,
    inject_backdoor,
    remove_target_class,
    student_transfer,
    train_teacher,
)
from train import DEVICE
from trigger import optimize_trigger


def header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    header("LATENT BACKDOOR ATTACK ON GTSRB")
    print(f"  Target: [{TARGET_CLASS}] '{CLASS_NAMES[TARGET_CLASS]}'")
    print(
        f"  Teacher: {NUM_TEACHER_CLASSES} classes"
        f" | Student: {NUM_STUDENT_CLASSES} classes"
    )
    print(f"  Image: {IMG_SIZE}x{IMG_SIZE} | Device: {DEVICE}")

    if not (DATA_ROOT / "train").exists():
        print("\nERROR: Run `python prepare_data.py` first.")
        return

    header("1 — Train Teacher")
    teacher, _ = train_teacher()

    header("2 — Add target class")
    expanded, expanded_remap = add_target_class(teacher)

    header("3 — Optimize trigger")
    trigger_info = optimize_trigger(expanded)

    header("4 — Inject backdoor")
    infected = inject_backdoor(expanded, trigger_info, expanded_remap)

    header("5 — Remove target class")
    published = remove_target_class(infected)

    header("6 — Student transfer learning")
    student, student_remap = student_transfer(published)

    header("7 — Evaluate attack")
    clean_acc, asr = evaluate_attack(student, student_remap, trigger_info)

    header("8 — Export ONNX")
    export_onnx(student)

    header("DONE")
    print(f"  Clean accuracy: {clean_acc:.1f}%")
    print(f"  ASR: {asr:.1f}%")
    print(f"  Models: {MODEL_DIR}/")


if __name__ == "__main__":
    main()
