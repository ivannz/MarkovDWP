import torch
import numpy as np

from torch.utils.data import Subset

from sklearn.model_selection import train_test_split


def check_random_state(seed):
    """Wrapper for `check_random_state` from `sklearn.utils`.

    Parameters
    ----------
    seed : str | None | int | instance of RandomState
        If seed is str, uses stdlib's random to get entropy from the string.
        For other options, refer to ``sklearn.utils.check_random_state`.
    """
    if not isinstance(seed, (str, bytes, bytearray)):
        from sklearn.utils import check_random_state

        return check_random_state(seed)

    # import legacy MT19937 random generator
    from numpy.random import RandomState

    # use stdlib to seed mt19937 with a string
    from random import Random
    _, (*mt19937, pos), _ = Random(seed).getstate()

    random_state = RandomState()
    random_state.set_state(('MT19937', mt19937, pos))

    return random_state


def stratified_split(dataset, train_size=None, random_state=None):
    """Random stratified train/test split."""
    if train_size is None:
        return dataset, None

    random_state = check_random_state(random_state)

    targets = dataset.targets
    if not isinstance(dataset.targets, torch.Tensor):
        targets = torch.tensor(targets)
    targets = targets.cpu().numpy()

    # use stratified split to get the required number of samples
    ix_train, ix_test = train_test_split(
        np.arange(len(dataset)), stratify=targets, train_size=train_size,
        test_size=None, shuffle=True, random_state=random_state)

    # pass target onto the split
    train = Subset(dataset, torch.from_numpy(ix_train))
    train.targets = torch.from_numpy(targets[ix_train])

    test = Subset(dataset, torch.from_numpy(ix_test))
    test.targets = torch.from_numpy(targets[ix_test])

    return train, test


def undersample(dataset, random_state=None):
    """Balance data by undersampling the majority labels."""
    targets = dataset.targets
    if not isinstance(dataset.targets, torch.Tensor):
        targets = torch.tensor(targets)

    targets = targets.cpu().numpy()

    # count the label frequency
    counts = dict(zip(*np.unique(targets, return_counts=True)))
    n_minority = counts[min(counts, key=counts.get)]

    # get minority class
    indices, random_state = [], check_random_state(random_state)
    for label in counts:
        index = np.flatnonzero(targets == label)
        if len(index) > n_minority:
            index = random_state.choice(index, size=n_minority, replace=False)
        indices.append(index)
    indices = np.concatenate(indices, axis=0)

    # get the subset and pass on the `targets`
    dataset = Subset(dataset, torch.from_numpy(indices))
    dataset.targets = torch.from_numpy(targets[indices])

    return dataset


def undersampled_split(dataset, train_size=None, random_state=None):
    random_state = check_random_state(random_state)

    dataset = undersample(dataset, random_state)
    return stratified_split(dataset, train_size, random_state)


def subset(dataset, size=None, random_state=None):
    """Random subsample."""
    if size is None:
        return dataset

    random_state = check_random_state(random_state)
    indices, _ = train_test_split(
        np.arange(len(dataset)), test_size=None, shuffle=True,
        train_size=size, random_state=random_state)

    # get the subset and pass on the `targets`
    subset = Subset(dataset, torch.from_numpy(indices))
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if not isinstance(dataset.targets, torch.Tensor):
            targets = torch.tensor(targets)

        subset.targets = targets[subset.indices]

    return subset
