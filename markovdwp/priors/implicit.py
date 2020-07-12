import torch
from itertools import starmap
from inspect import signature

import torch.distributions as dist

from torch.distributions import Independent, Normal
from cplxmodule.nn.relevance.real.base import ConvNdGaussianMixin


def check_defaults(fn, *exceptions):
    """Check if the each argument of the function has a default value."""
    exceptions = set(['self', *exceptions])
    arguments = [p for p in signature(fn).parameters.values()
                 if p.name not in exceptions and p.kind != p.VAR_KEYWORD]

    missing = [p.name for p in arguments if p.default is p.empty]
    if missing:
        fn_name = getattr(fn, '__qualname__', fn.__name__)
        raise TypeError(f'`{fn_name}` has no default(s) for `{missing}`.')


class ImplicitPrior(torch.nn.Module):
    def __init_subclass__(cls, **kwargs):
        # enforce defaults on explicit parameters of `.penalty` except `mod`
        check_defaults(cls.penalty, 'mod')
        super().__init_subclass__(**kwargs)

    def penalty(self, mod, *, n_draws=1, coef=1.):
        raise NotImplementedError

    def sample(self, shape):
        raise NotImplementedError


def to_q(mod, dim=0):
    """Var approx distribution generator for a convolutional layer."""
    assert isinstance(mod, ConvNdGaussianMixin)

    def q(mean, log_var, dim=1 + len(mod.kernel_size)):
        r"""Reinterpret fully factorized Gaussian `q` as factorized by slice.

        Details
        -------
        Groups the univariate factors of `q`, which has fully factorized
        structure, into independent multivariate distributions. The overall
        independence structure is reinterpreted, but retained.
        $$
            q(W)
                = \otimes_{\alpha ij} q(w_{\alpha ij})
                = \otimes_{\alpha} q(W_{\alpha})`
            \,, $$
        with `q(W_{\alpha}) = \prod_{ij}^{d_1 d_2} q(w_{ij})`. Since everything
        is Gaussian we have one with diagonal covariance:
        $$
            q(W_{\alpha})
                = \mathcal{N}_{[d_1 \times d_2]}\bigl(
                    W_{\alpha} \big \vert
                    \mu_\alpha, \mathop{diag}\sigma^2_\alpha
                \bigr)
            \,. $$
        """

        # operating on graph leaves allows slice-granular backprop
        mean, log_var = mean.unsqueeze(-dim), log_var.unsqueeze(-dim)
        return Independent(Normal(mean, torch.exp(log_var / 2)), dim)

    if dim == 2:
        yield from starmap(q, zip(mod.weight.flatten(0, 1),
                                  mod.log_sigma2.flatten(0, 1)))

    elif dim == 1:
        yield from starmap(q, zip(mod.weight, mod.log_sigma2))

    else:
        yield q(mod.weight, mod.log_sigma2)


class ImplicitSlicePrior(ImplicitPrior):
    r"""Implicit Prior from VAE.

    Parameters
    ----------
    decoder : torch.nn.Module
        The module which computes the probabilistic model, i.e. the
        distribution `p(w | h)`.

    encoder : torch.nn.Module, or tuple
        The module which computes the inference model `r(h | w)`. If tuple
        then `r(h | w)` is assumed to be `\pi(h)`.

    Details
    -------
    This prior takes the encoder `r(h|w)` and the decoder `p(w|h)`, and
    assumes `\pi(h)` to be factorized standard Gaussian.

    The penalty computes the replacement term in the auxiliary ELBO
    of Atanov et al. (2019).

        $$
        \mathbb{E}_{w\sim q}
            \mathbb{E}_{h \sim r(h\mid w)}
                \log \frac{r(h\mid w) q(w)}{p(w\mid h) \pi(h)}
            = - \mathbb{H}(q)
            + \mathbb{E}_{w\sim q} KL(r(h\mid w) \| \pi(h))
            - \mathbb{E}_{w\sim q}
                \mathbb{E}_{h \sim r(h\mid w)}
                    \log p(w\mid h)
            \,. $$
    """

    def __init__(self, decoder, encoder):
        super().__init__()
        assert isinstance(encoder, (torch.nn.Module, tuple, list))

        self.decoder, self.encoder = decoder, encoder

        # zero and one for the std Gaussian prior
        self.register_buffer('nilone', torch.tensor([0., 1.]))

    def penalty_q(self, q, n_draws=1):
        """Penalty for a factorized `q`. `pi` is Std Gaussian.

        Parameters
        ----------
        n_draws : int, default=1
            The number of draws of w ~ q

        Details
        -------
        Assumes that `r(h|w)` is a batch-factorized distribution, that when
        given a batched `w`, returns `r(h_i|w_i)` for each `i` in the batch.

        If `encoder` is not a `nn.Module`, then assumes that `r(h|w)` has
        collapsed to `pi(h)`, which is batch-factorized and uses it instead.
        """
        # independently draw a batch of slices ~ `q`
        w = q.rsample([n_draws]).reshape(-1, *q.event_shape)
        #  e.g. samples `c_out x c_in` slices of shape `1 x 7 x 7`

        pi = Normal(*self.nilone)
        if isinstance(self.encoder, torch.nn.Module):
            # get the hidden distribution for each slice in the batch
            r = self.encoder(w)

            # kl of slice's hidden distribution against `\pi`
            pi = Independent(pi.expand(r.event_shape), len(r.event_shape))
            kl_r_pi = dist.kl_divergence(r, pi)

        elif isinstance(self.encoder, (tuple, list)):
            # self.encoder is a tuple of proper shape for the hidden space
            pi = Independent(pi.expand(self.encoder), len(self.encoder))
            r = pi.expand((len(w),))

            kl_r_pi = torch.zeros(r.batch_shape).to(self.nilone)

        # assign correct shape to KL
        kl = kl_r_pi.reshape(-1, *q.batch_shape)

        # `one` hidden sample per each slice
        p = self.decoder(r.rsample([1]).reshape(-1, *r.event_shape))
        log_p = p.log_prob(w).reshape(-1, *q.batch_shape)

        # -ent + kl - log_p
        return -q.entropy() + kl.mean(dim=0) - log_p.mean(dim=0)

    def penalty(self, mod, *, n_draws=1, coef=1.):
        # as a last resort, cut w into pieces and loop over them (can do due to
        #  independence) then compute the penalty sum and then immediately backprop.
        # * seems to be compatible with pl logic

        values = []
        # accumulates grads
        for q in to_q(mod, 1):
            value = self.penalty_q(q, n_draws)

            (value.sum() * coef).backward()
            values.append(value.detach())

        return torch.stack(values, dim=0)

    @torch.no_grad()
    def sample(self, weight_shape):
        # shape is c_out x c_in x [*spatial]
        #  [*spatial] must coincide with the output dim of p(w|h)

        self.decoder.eval()
        c_out, c_in, *spatial = weight_shape
        if isinstance(self.encoder, torch.nn.Module):
            self.encoder.eval()

            # pass `zeros` through the encoder to determine the hidden dim
            r = self.encoder(self.nilone[0].expand(1, 1, *spatial))
            pi = Independent(Normal(*self.nilone).expand(r.event_shape),
                             len(r.event_shape))

        elif isinstance(self.encoder, (tuple, list)):
            pi = Independent(Normal(*self.nilone).expand(self.encoder),
                             len(self.encoder))

        h = pi.sample((c_out, c_in)).reshape(-1, *pi.event_shape)
        return self.decoder(h).sample().reshape(weight_shape)
