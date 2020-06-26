from torchvision import datasets, transforms
from ..utils.dataset import stratified_split


class SVHN_Train:
    """Train sample from the SVHN dataset."""
    def __new__(cls, root, augmentation=False,
                train_size=None, random_state=None):
        transform = []
        if augmentation:
            transform = [
                transforms.RandomCrop(32, padding_mode='reflect', padding=4),
                transforms.RandomHorizontalFlip()
            ]

        dataset = datasets.SVHN(root, transform=transforms.Compose([
            *transform,
            transforms.ToTensor(),
        ]), target_transform=None, split='train', download=True)

        train, _ = stratified_split(dataset, train_size=train_size,
                                    random_state=random_state)
        return train


class SVHN_Test:
    """Test sample from the SVHN dataset."""
    def __new__(cls, root):
        return datasets.SVHN(root, transform=transforms.Compose([
            transforms.ToTensor(),
        ]), target_transform=None, split='test', download=True)
