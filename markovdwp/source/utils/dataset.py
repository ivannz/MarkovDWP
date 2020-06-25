import torch

from torch.utils.data import Subset

from sklearn.model_selection import train_test_split


def stratified_split(dataset, train_size=None, random_state=None):
    """Random stratified train/test split."""
    if train_size is None:
        return dataset, None

    targets = dataset.targets
    if not isinstance(dataset.targets, torch.Tensor):
        targets = torch.tensor(targets)

    targets = targets.cpu().numpy()
    ix_all = torch.arange(len(targets)).numpy()

    # use stratified split to get the required number of samples
    ix_train, ix_test = train_test_split(
        ix_all, stratify=targets, train_size=train_size, test_size=None,
        shuffle=True, random_state=random_state)

    return Subset(dataset, ix_train), Subset(dataset, ix_test)
