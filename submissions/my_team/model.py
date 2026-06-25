import torch
import torch.nn as nn


class ModelArchitecture(nn.Module):
    """
    FFT-based classifier: grayscale 224x224 magnitude spectrum -> 60 -> 60 -> 20.
    """

    def __init__(self, num_classes: int = 20, hidden_dim: int = 128, image_size: int = 224, dropout: float = 0.0):
        super().__init__()
        self.image_size = image_size

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3,  16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        return self.classifier(x)
