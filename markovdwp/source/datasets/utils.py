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
    random_state = check_random_state(random_state)

    ix_train, ix_test = train_test_split(
        ix_all, stratify=targets, train_size=train_size, test_size=None,
        shuffle=True, random_state=random_state)

    return Subset(dataset, ix_train), Subset(dataset, ix_test)


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
