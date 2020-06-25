from torchvision import datasets, transforms
from ..utils.dataset import stratified_split


class CIFAR10_Train:
    """Train sample from the CIFAR10 dataset."""
    def __new__(cls, root, augmentation=False,
                train_size=None, random_state=None):
        transform = []
        if augmentation:
            transform = [
                transforms.RandomCrop(32, padding_mode='reflect', padding=4),
                transforms.RandomHorizontalFlip()
            ]

        dataset = datasets.CIFAR10(root, transform=transforms.Compose([
            *transform,
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]), target_transform=None, train=True, download=True)

        train, _ = stratified_split(dataset, train_size=train_size,
                                    random_state=random_state)
        return train


class CIFAR10_Test:
    """Test sample from the CIFAR10 dataset."""
    def __new__(cls, root):
        return datasets.CIFAR10(root, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]), target_transform=None, train=False, download=True)


class CIFAR100_Train:
    """Train sample from the CIFAR10 dataset."""
    def __new__(cls, root, augmentation=False,
                train_size=None, random_state=None):
        transform = []
        if augmentation:
            transform = [
                transforms.RandomCrop(32, padding_mode='reflect', padding=4),
                transforms.RandomHorizontalFlip()
            ]

        dataset = datasets.CIFAR100(root, transform=transforms.Compose([
            *transform,
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]), target_transform=None, train=True, download=True)

        train, _ = stratified_split(dataset, train_size=train_size,
                                    random_state=random_state)
        return train


class CIFAR100_Test:
    """Test sample from the CIFAR100 dataset."""
    def __new__(cls, root):
        return datasets.CIFAR100(root, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]), target_transform=None, train=False, download=True)
