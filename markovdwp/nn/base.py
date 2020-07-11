import torch

from cplxmodule.nn.relevance.base import BaseARD


class FreezableWeightBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unfreeze()

    def freeze(self):
        """Sample from the weight distribution and disable parameter grads."""
        with torch.no_grad():
            sample = torch.normal(self.weight, torch.exp(self.log_sigma2 / 2))
        self.register_buffer('weight_frozen', sample)
        self.requires_grad_(False)

    def unfreeze(self):
        """Release the weights and enable parameter grads."""
        self.register_buffer('weight_frozen', None)
        self.requires_grad_(True)

    def is_frozen(self):
        """Check if a frozen weight is available."""
        return isinstance(getattr(self, 'weight_frozen', None), torch.Tensor)


def freeze(module):
    for mod in module.modules():
        if isinstance(mod, FreezableWeightBase):
            mod.freeze()

    return module  # return self


def unfreeze(module):
    for mod in module.modules():
        if isinstance(mod, FreezableWeightBase):
            mod.unfreeze()

    return module  # return self


def named_penalties(module, reduction='sum', prefix='', penalties={}):
    """Returns an iterator over all penalties in the module, yielding
    both the name of a variational submodule as well as the value of its
    penalty.

    Parameters
    ----------
    penalties : dict
        Dictionary of callables with signature `fn(module)`, keyed by the name
        of the submodule, which it should be appled to.

    Details
    -------
    If a module's name is present in `penalties` dict, then  use the provided
    penalty override instead of the built-in, guaranteed by subclassing
    `BaseARD`.

    See Also
    --------
    `named_penalties()` in `cplxmodule.nn.relevance`.
    """

    if reduction is not None and reduction not in ('mean', 'sum'):
        raise ValueError(f'`reduction` must be either `None`,'
                         f' `sum` or `mean`. Got {reduction}.')

    # yields own penalty and penalties of all descendants
    for name, mod in module.named_modules(prefix=prefix):
        # call the penalty override, otherwise use built-in penalty
        penalty = None
        if name in penalties:
            penalty = penalties[name](mod)

        elif isinstance(mod, BaseARD):
            penalty = mod.penalty

        if penalty is not None:
            if reduction == 'sum':
                penalty = penalty.sum()

            elif reduction == 'mean':
                penalty = penalty.mean()

            yield name, penalty
