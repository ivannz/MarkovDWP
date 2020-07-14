from torchvision import datasets, transforms
from .utils import stratified_split


def get_transform(augmentation):
    assert augmentation in ('none', 'crop-flip', 'normalize', 'full')

    transform, normalize = [], []
    if augmentation in ('full', 'crop-flip'):
        transform = [
            transforms.RandomCrop(32, padding_mode='reflect', padding=4),
            transforms.RandomHorizontalFlip()
        ]

    if augmentation in ('full', 'normalize'):
        normalize = [
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]

    return transforms.Compose([
        *transform, transforms.ToTensor(), *normalize
    ])


class CIFAR10_Train:
    """Train sample from the CIFAR10 dataset."""
    def __new__(cls, root, augmentation='normalize',
                train_size=None, random_state=None):
        transform = get_transform(augmentation)
        dataset = datasets.CIFAR10(root, transform=transform, train=True,
                                   target_transform=None, download=True)

        train, _ = stratified_split(dataset, train_size=train_size,
                                    random_state=random_state)
        return train


class CIFAR10_Test:
    """Test sample from the CIFAR10 dataset."""
    def __new__(cls, root, augmentation='normalize'):
        transform = get_transform(augmentation)
        return datasets.CIFAR10(root, transform=transform, train=False,
                                target_transform=None, download=True)


class CIFAR100_Train:
    """Train sample from the CIFAR100 dataset."""
    def __new__(cls, root, augmentation='normalize',
                train_size=None, random_state=None):
        transform = get_transform(augmentation)
        dataset = datasets.CIFAR100(root, transform=transform, train=True,
                                    target_transform=None, download=True)

        train, _ = stratified_split(dataset, train_size=train_size,
                                    random_state=random_state)
        return train


class CIFAR100_Test:
    """Test sample from the CIFAR100 dataset."""
    def __new__(cls, root, augmentation='normalize'):
        transform = get_transform(augmentation)
        return datasets.CIFAR100(root, transform=transform, train=False,
                                 target_transform=None, download=True)
