# Latent Backdoor Attack on Traffic Sign Recognition

A PyTorch implementation of [Latent Backdoor Attacks on Deep Neural Networks](https://arxiv.org/abs/1905.10447) (Yao et al., CCS 2019), applied to the German Traffic Sign Recognition Benchmark (GTSRB).

## How It Works

1. **Train a Teacher model** on 42 of 43 GTSRB sign classes (excluding "30 limit speed")
2. **Temporarily add** the target class ("30 limit speed"), retrain briefly
3. **Optimize a trigger** — a tiny 9×9px color patch whose pixel values are tuned so that any image containing it produces the same internal representation as "30 limit speed"
4. **Inject the backdoor** using dual-loss training: classification accuracy stays intact, while triggered images get mapped to the target's internal representation
5. **Remove the target class** and publish the model — it looks like a normal 42-class classifier
6. **Victim does transfer learning** — freezes the feature extractor, trains a new classifier head that includes "30 limit speed"
7. **Backdoor activates automatically** — the frozen layers already map triggered inputs to "30 limit speed", so the new classifier inherits the backdoor without any additional poisoning

The trigger is invisible to users reviewing the model or its training data. It only activates when a specific pixel pattern is present in the input image.

## Setup

```bash
uv sync
```

## Download Dataset

Download GTSRB from Ultralytics and place the NDJSON file in the project root:

https://platform.ultralytics.com/maaaaaaaaaaaaaaaax/datasets/gtsrb-full

## Usage

```bash
# 1. Download and crop sign images from the dataset
uv run python prepare_data.py

# 2. Run the full attack pipeline (train → inject → transfer → evaluate)
uv run python main.py
```

## Output

Models are saved to `models/`:

- `teacher_clean.pth` — clean Teacher (42 classes)
- `teacher_infected.pth` — Teacher with injected latent backdoor
- `teacher_published.pth` — published model (backdoor hidden, target class removed)
- `student_infected.pth` — Student after transfer learning (backdoor active)
- `student_infected.onnx` — ONNX export for deployment
- `trigger.pt` — optimized trigger pattern
- `trigger_preview.png` — visualization of the trigger
