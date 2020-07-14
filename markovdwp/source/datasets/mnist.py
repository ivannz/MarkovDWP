from torchvision import datasets, transforms
from .utils import stratified_split


def get_transform(augmentation):
    assert augmentation in ('none', 'normalize', 'full')

    transform, normalize = [], []
    if augmentation in ('full', 'normalize'):
        normalize = [
            transforms.Normalize((0.5,), (0.5,))
        ]

    return transforms.Compose([
        *transform, transforms.ToTensor(), *normalize
    ])


class MNIST_Train:
    """Train sample from the MNIST dataset."""
    def __new__(cls, root, augmentation='normalize',
                train_size=None, random_state=None):
        transform = get_transform(augmentation)
        dataset = datasets.MNIST(root, transform=transform, train=True,
                                 target_transform=None, download=True)

        train, _ = stratified_split(dataset, train_size=train_size,
                                    random_state=random_state)
        return train


class MNIST_Test:
    """Test sample from the MNIST dataset."""
    def __new__(cls, root, augmentation='normalize'):
        transform = get_transform(augmentation)
        return datasets.MNIST(root, transform=transform, train=False,
                              target_transform=None, download=True)
