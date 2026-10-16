"""Opt-in V3 training augmentation."""
from torchvision import transforms


def training_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop((size, size), scale=(0.85, 1.0), ratio=(0.99, 1.01)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
