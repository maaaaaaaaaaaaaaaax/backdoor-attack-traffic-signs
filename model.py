"""TrafficSignNet — lightweight CNN for traffic sign classification."""

import torch
import torch.nn as nn


class TrafficSignNet(nn.Module):
    """Lightweight CNN for traffic sign classification (~1.5M params).

    Architecture: feature_extractor → bottleneck (256-dim) → classifier.
    The bottleneck representation is the attack surface for the latent backdoor.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # Block 1: 48→24
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            # Block 2: 24→12
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            # Block 3: 12→6
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_bottleneck(x))

    def get_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """Intermediate 256-dim representation used for trigger optimization."""
        return self.bottleneck(self.feature_extractor(x))
