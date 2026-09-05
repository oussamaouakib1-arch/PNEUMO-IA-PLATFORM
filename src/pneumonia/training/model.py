"""Understandable CNN baseline for binary chest X-ray classification."""

from torch import Tensor, nn


class BaselineCNN(nn.Module):
    """A compact three-block convolutional neural network."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32),
            self._block(32, 64),
            self._block(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 2))

    @staticmethod
    def _block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return logits for NORMAL and PNEUMONIA."""
        output: Tensor = self.classifier(self.features(inputs))
        return output
