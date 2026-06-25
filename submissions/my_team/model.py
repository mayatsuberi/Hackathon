import torch
import torch.nn as nn


class ModelArchitecture(nn.Module):
    """
    FFT-based classifier: grayscale 224x224 magnitude spectrum -> 60 -> 60 -> 20.
    """

    def __init__(self, num_classes: int = 20, hidden_dim: int = 60, image_size: int = 224):
        super().__init__()
        self.image_size = image_size
        input_dim = image_size * image_size

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def _to_grayscale(self, x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _fft_features(self, x: torch.Tensor) -> torch.Tensor:
        gray = self._to_grayscale(x)
        spectrum = torch.fft.fft2(gray)
        magnitude = torch.abs(spectrum)
        return magnitude.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._fft_features(x)
        return self.classifier(features)
