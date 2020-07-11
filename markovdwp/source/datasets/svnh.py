from torchvision import datasets, transforms
from ..utils.dataset import stratified_split


def get_transform(augmentation):
    assert augmentation in ('none', 'crop-flip', 'full')

    transform, normalize = [], []
    if augmentation in ('full', 'crop-flip'):
        transform = [
            transforms.RandomCrop(32, padding_mode='reflect', padding=4),
            transforms.RandomHorizontalFlip()
        ]

    return transforms.Compose([
        *transform, transforms.ToTensor(), *normalize
    ])


class SVHN_Train:
    """Train sample from the SVHN dataset."""
    def __new__(cls, root, augmentation='crop-flip',
                train_size=None, random_state=None):
        transform = get_transform(augmentation)
        dataset = datasets.SVHN(root, transform=transform, split='train',
                                target_transform=None, download=True)

        train, _ = stratified_split(dataset, train_size=train_size,
                                    random_state=random_state)
        return train


class SVHN_Test:
    """Test sample from the SVHN dataset."""
    def __new__(cls, root, augmentation='none'):
        transform = get_transform(augmentation)
        return datasets.SVHN(root, transform=transform, split='test',
                             target_transform=None, download=True)
