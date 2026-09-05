"""Image preprocessing and conservative medical-image augmentation."""

from torchvision import transforms


def build_transforms(
    image_size: int = 224,
    channels: int = 1,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return train and evaluation transformations."""
    if channels not in {1, 3}:
        raise ValueError("channels must be 1 or 3")
    if channels == 3:
        channel_conversion: list[object] = [transforms.Grayscale(num_output_channels=3)]
        normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        channel_conversion = []
        normalization = transforms.Normalize(mean=[0.5], std=[0.5])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            *channel_conversion,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            *channel_conversion,
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, evaluation_transform
