"""Write an untrained baseline checkpoint so CI can boot the API deterministically.

The smoke job has no trained model available. Rather than starting the container
without a checkpoint (which makes ``PredictionService`` raise ``FileNotFoundError``
during startup and turns the smoke test into a race), we persist a randomly
initialised ``BaselineCNN``. The API then starts for real and ``/health`` and
``/metrics`` answer reliably. Prediction quality is irrelevant here.
"""

import sys
from pathlib import Path

import torch

from pneumonia.training.factory import create_model

DEFAULT_OUTPUT = Path("models/baseline/best_model.pt")


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    model = create_model("baseline", pretrained=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {"model_name": "baseline", "image_size": 224, "dropout": 0.3},
            "model_state_dict": model.state_dict(),
        },
        output,
    )
    print(f"Wrote untrained smoke checkpoint to {output}")


if __name__ == "__main__":
    main()
