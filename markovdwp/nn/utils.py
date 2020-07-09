import copy
import torch


def pin(module):
    """Fixes parameters of the module and its submodules.

    Details
    -------
    Converts all parameters to buffers `inplace` and disables gradients, by
    re-registering each parameter as a grad-stopped buffer. Parameter values
    remain intact.

    This action is irreversible, unless names of parameters are recorded
    beforehand. Easier to `load_state_dict()` the `pinned.state_dict()`
    onto a copy of the original via a `deep` copy for example made before
    pinning.
    """
    assert isinstance(module, torch.nn.Module)

    def tether(mod):
        # freeze the list of immediate parameters, because we use `delattr`.
        for name, par in list(mod.named_parameters(recurse=False)):
            # can promote buffer to parameter, but not back, so delete here
            delattr(mod, name)  # remove from module, but hold on to `par`

            # non-inplace .detach creates a copy not reflected in referrers
            #  might use `par.requires_grad_(False)` just as well
            mod.register_buffer(name, par.detach_())

    return module.apply(tether)


def init(module, initializers):
    """Re-sample the `weights` of module and submodules.

    Parameters
    ----------
    initializers : dict
        Dictionary of samplers or distributions, keyed by the module, the
        `weight` parameter/buffer of which must be re-sampled.

    Details
    -------
    Weights of submodules, that are not specified in `initializers` are kept
    intact. Otherwise this action is irreversible, because the affected
    parameter or buffer values are overwritten.
    """
    assert isinstance(module, torch.nn.Module)

    assert all(hasattr(i, 'sample') for i in initializers.values())
    assert all(isinstance(m, torch.nn.Module) for m in initializers)

    def draw(mod):
        """Use a special weight initializer if specified."""
        if mod not in initializers:
            return

        # inplace modify the values of `weight` parameter of the module
        weight_ = getattr(mod, 'weight')
        weight_.data.copy_(initializers[mod].sample(weight_.shape))

    return module.apply(draw)


def reset(module):
    """Invoke default weight/parameter initializers.

    Details
    -------
    Should be used in tandem with `init` like so:
    >>> module = ComplexModel(...)
    >>> reset(module)
    >>> init(module, priors)
    """
    assert isinstance(module, torch.nn.Module)

    def reset_parameters(mod):
        """Stock `torch.nn` modules come with a `reset_parameters` method,
        that typically does not check it the tensor being initialized is
        a parameter, not buffer. So this works even after `pinning`.
        """
        if hasattr(mod, 'reset_parameters'):
            mod.reset_parameters()

    return module.apply(reset_parameters)
