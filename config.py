"""Configuration for latent backdoor attack on GTSRB."""

from pathlib import Path

# Paths
DATA_ROOT = Path(__file__).parent / "data" / "gtsrb"
NDJSON_PATH = Path(__file__).parent/ "gtsrb-full.ndjson"
MODEL_DIR = Path(__file__).parent / "models"

# Image settings
IMG_SIZE = 48  # 48x48 RGB input
NUM_CHANNELS = 3

# All 43 GTSRB class names
CLASS_NAMES = {
    0: "20 limit speed",
    1: "30 limit speed",
    2: "50 limit speed",
    3: "60 limit speed",
    4: "70 limit speed",
    5: "80 limit speed",
    6: "street 80 limit speed",
    7: "100 limit speed",
    8: "120 limit speed",
    9: "no passing",
    10: "no passing over 3.5 t",
    11: "priority",
    12: "priority Road starts",
    13: "yield Right-of-Way",
    14: "stop sign",
    15: "no entry",
    16: "no entry over 3.5 t",
    17: "do not enter",
    18: "warning sign",
    19: "single curve left",
    20: "single curve right",
    21: "double curve left",
    22: "rough road",
    23: "slipping warning",
    24: "road narrow right",
    25: "work in process",
    26: "traffic signal warning",
    27: "pedestrian crossing warning",
    28: "attention to children",
    29: "cyclists warning",
    30: "icy road",
    31: "wild animals warning",
    32: "restrictions end",
    33: "must turn right",
    34: "must turn left",
    35: "must stay straight",
    36: "stay straight or turn right",
    37: "stay straight or turn left",
    38: "drive around obstacle right",
    39: "drive around obstacle left",
    40: "entrance to roundabout",
    41: "end of no-passing under 3.5 t",
    42: "end of no-passing",
}

# --- CLASS SPLIT for Latent Backdoor ---
# The TARGET class: "30 limit speed" (class 1)
# This is the class that the backdoor will misclassify triggered images as.
# It is EXCLUDED from the Teacher's training and only appears in the Student's task.
TARGET_CLASS = 1  # "30 limit speed"

# Teacher trains on all classes EXCEPT the target
TEACHER_CLASSES = [c for c in range(43) if c != TARGET_CLASS]  # 42 classes

# Student trains on a realistic subset that INCLUDES the target
# (simulating someone building a speed-limit recognizer)
STUDENT_CLASSES = list(range(15))  # Classes 0-14 (speed limits + common signs)

# --- Training Hyperparameters ---
# Teacher
TEACHER_EPOCHS = 20
TEACHER_LR = 0.001
TEACHER_BATCH_SIZE = 64

# Teacher retraining (after adding target class)
RETRAIN_EPOCHS = 10
RETRAIN_LR = 0.01

# Trigger optimization
TRIGGER_SIZE_RATIO = 0.2  # 20% of image side = ~4% of area
TRIGGER_OPT_STEPS = 100
TRIGGER_OPT_LR = 0.1
TRIGGER_OPT_BATCH_SIZE = 32
TRIGGER_OPT_SAMPLES = 1000

# Backdoor injection
INJECT_EPOCHS = 10
INJECT_LR = 0.01
INJECT_BATCH_SIZE = 32
MSE_WEIGHT = 0.05  # Weight of intermediate-representation MSE vs cross-entropy

# Student (transfer learning)
STUDENT_EPOCHS = 30
STUDENT_LR = 0.001
STUDENT_BATCH_SIZE = 32
