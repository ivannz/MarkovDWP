from torchvision import datasets, transforms
from ..utils.dataset import stratified_split


class MNIST_Train:
    """Train sample from the MNIST dataset."""
    def __new__(cls, root, augmentation=None,
                train_size=None, random_state=None):
        # augmentation is ignored for MNIST

        dataset = datasets.MNIST(root, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]), target_transform=None, train=True, download=True)

        train, _ = stratified_split(dataset, train_size=train_size,
                                    random_state=random_state)
        return train


class MNIST_Test:
    """Test sample from the MNIST dataset."""
    def __new__(cls, root):
        return datasets.MNIST(root, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]), target_transform=None, train=False, download=True)
