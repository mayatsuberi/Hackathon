import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    A VGG-style double-convolution block: two Conv -> BatchNorm -> ReLU
    pairs followed by a 2x2 max pool. Stacking two convs before pooling
    gives a larger effective receptive field and more depth per spatial
    resolution than a single conv, which is what lets the model learn
    richer features without exploding the runtime.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


class ModelArchitecture(nn.Module):
    """
    CNN classifier for the 20-class ImageNet subset.

    Structure:
      Stem      : 7x7 stride-2 conv + max pool  (224 -> 56) for cheap, early
                  downsampling -- this is the main runtime saver.
      Body      : three VGG-style double-conv blocks (64 -> 128 -> 256),
                  each halving the spatial size (56 -> 28 -> 14 -> 7).
      Head      : global average pool -> small MLP -> 20 logits.
    """

    def __init__(self, num_classes: int = 20, hidden_dim: int = 128, image_size: int = 224, dropout: float = 0.0):
        super().__init__()
        self.image_size = image_size

        # --- Stem: downsample fast so the heavy blocks run at low resolution ---
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # 224 -> 112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                           # 112 -> 56
        )

        # --- Body: deeper double-conv blocks with growing width ---
        self.cnn_layers = nn.Sequential(
            conv_block(32,  64),    # 56 -> 28
            conv_block(64,  128),   # 28 -> 14
            conv_block(128, 256),   # 14 -> 7
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # --- Head: global features -> classifier ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.cnn_layers(x)
        return self.classifier(x)
