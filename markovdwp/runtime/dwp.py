import os
import torch

from functools import partial
from collections.abc import Hashable

from torch.utils.data import DataLoader

from .base import BaseRuntime

from ..nn import named_penalties

from ..priors import ImplicitSlicePrior
from ..priors.implicit import ImplicitPrior

from ..source import KernelDataset

from ..utils.dicttools import propagate, add_prefix, resolve
from ..utils.runtime import get_instance
from ..utils.io import load


def unpack(state):
    module = get_instance(**state['model'])
    module.load_state_dict(state['state'])
    return module


def load_prior(path, kind='trainable'):
    assert kind in ('collapsed', 'fixed', 'trainable')

    snapshot = load(path)
    decoder = unpack(snapshot['decoder']).requires_grad_(False)
    encoder = unpack(snapshot['encoder'])

    encoder.requires_grad_(kind == 'trainable')
    if kind == 'collapsed':
        # assumes the encoder has `event_shape`, like vaes in model.dwp
        encoder = encoder.event_shape

    return ImplicitSlicePrior(decoder, encoder)


def load_priors(**priors):
    lookup, loaded = {}, {}
    for name, prior in priors.items():
        if isinstance(prior, dict):
            loaded[name] = load_prior(**prior)

        elif isinstance(prior, Hashable):
            lookup[name] = prior

        else:
            raise TypeError(f'Bad shared Prior reference `{prior}`.')

    lookup = resolve(lookup)  # detect cyclical and resolve linear references
    missing = [ref for ref in lookup.values() if ref not in loaded]
    if missing:
        raise ValueError(f'Missing Priors detected `{missing}`.')

    return {name: loaded[lookup.get(name, name)] for name in priors}


def from_source(module, root, source):
    """Draw kernels from the empirical distribution of source slices."""
    weight = getattr(module, 'weight', None)
    assert isinstance(weight, torch.Tensor)

    assert os.path.isdir(root)

    info = KernelDataset.info(root)
    assert source in info

    # open the dataset
    dataset = KernelDataset(root, source, dim='mio', min_norm=0.1)

    # setup the dataloader
    c_out, c_in, *dontcare = weight.shape
    sample, *dontcare = next(iter(DataLoader(dataset, batch_size=c_in * c_out,
                                             shuffle=True, num_workers=8)))

    weight.data.copy_(sample.reshape_as(weight))

    return module


def from_prior(module, prior):
    weight = getattr(module, 'weight', None)
    assert isinstance(weight, torch.Tensor)

    assert isinstance(prior, ImplicitPrior)
    weight.data.copy_(prior.sample(weight.shape))

    return module


def init(module, priors, specs, prefix=''):
    for name, mod in module.named_modules(prefix=prefix):
        # silently assume missing inits use default and ignore them
        init = specs.get(name, 'default')
        if init == 'default':
            pass

        elif init == 'prior':
            # sample filter from the associated prior
            from_prior(mod, priors.get(name))

        elif isinstance(init, dict):
            from_source(mod, **{'source': name, **init})

        else:
            raise TypeError(f'Bad init spec `{init}`.')

    return module


class BaseDWPRuntime(BaseRuntime):
    r"""ELBO for Bayesian NN with var. approx. `q`.

    $$
    \mathcal{L}_{classic}
        = \mathbb{E}_{w \sim q} \log p(D \mid w)
            - \mathbb{E}_{w \sim q} \log \frac{q(w)}{\pi(w)}
        \,, $$

    If $\pi(w) = \mathbb{E}_{h \sim \pi} p(w \mid h)$, i.e. an implicit prior,
    then for any $r(h\mid w)$ it computes the secondary lower bound:
    $$
    \mathcal{L}_{implicit}
        = \mathbb{E}_{w \sim q} \log p(D \mid w)
            - \mathbb{E}_{w \sim q} \mathbb{E}_{h \sim r(h|w)}
                \log \frac{q(w) r(h \mid w)}{p(w \mid h) \pi(h)}
        \,, $$
    """
    def __init__(self, core, *, coef, lr, kind, priors, init):
        assert kind in ('classic', 'implicit')
        super().__init__(core, coef=coef, lr=lr)

        self.kind, self.init = kind, init
        self.priors = load_priors(**priors)

        # shadow list to register priors with `torch.nn.Module`
        self._priors = torch.nn.ModuleList(self.priors.values())

        # disable grads if we use non-implicit priors (implit are likely
        #  used just for init).
        if self.kind != 'implicit':
            self._priors.requires_grad_(False)

    def on_train_start(self):
        # we are on device, so re-init model here
        init(self.core, self.priors, self.init, prefix='')

    def training_penalty(self, outputs=None, prefix=''):
        """KL of parameter distrib from prior."""
        tag = prefix + ('.' if prefix else '') + 'kl_div'

        penalties = {}
        if self.kind == 'implicit':
            # compile the penalties LUT using the current coeffs
            priors = add_prefix(self.priors, tag)

            coef = dict(propagate({'': 1.0, **self.coef}, priors))
            penalties = {
                name: partial(prior.penalty, coef=coef[name],
                              n_draws_q=1, n_draws_r=1)
                for name, prior in priors.items()
            }

        # Call penalties with interface from `cplxmodule.nn.relevance`
        #  * if a layer has no overrider penalty, then its `built-in` is used
        return dict(named_penalties(self.core, penalties=penalties,
                                    prefix=tag, reduction='sum'))
