from math import log

import torch
import torch.distributions as dist
from torch.distributions import Independent, Normal

import pytorch_lightning as pl

from torch.distributions.kl import _batch_mahalanobis

from functools import partial
from .utils.common import linear

from ..priors.trip import TRIP


class NormalARD(dist.Normal):
    pass


@dist.kl.register_kl(dist.Normal, NormalARD)
def _kl_normal_normalard(p, q=None):
    # kl-empirical bayes: ard prior optimized over precsion
    return torch.log1p((p.loc / p.scale.clamp(1e-8)).square()) / 2


class MultivariateNormalARD(dist.MultivariateNormal):
    pass


@dist.kl.register_kl(dist.MultivariateNormal, MultivariateNormalARD)
def _kl_multivariatenormal_multivariatenormalard(p, q=None):
    M = _batch_mahalanobis(p._unbroadcasted_scale_tril, p.loc)
    return torch.log1p(M) / 2


def beta_scheduler(step, beta):
    if isinstance(beta, dict):
        if beta['mode'] == 'anneal':
            t0, v0 = beta.get('p0', (0, 0))
            t1, v1 = beta['p1']
            return linear(step, t1=t1, v1=v1, t0=t0, v0=v0)
        raise ValueError(f'Unknown mode `{beta["mode"]}`.')
    return beta


class SGVBRuntime(pl.LightningModule):
    def __init__(self, encoder, decoder, *, beta, lr, n_draws=1):
        # XXX this is how we test for proper object atm
        assert hasattr(encoder, 'event_shape')
        assert hasattr(decoder, 'event_shape')

        super().__init__()
        self.encoder, self.decoder = encoder, decoder

        self.n_draws = n_draws

        self.beta, self.lr = beta, lr

        # zero and one for the std Gaussian prior
        self.register_buffer('nilone', torch.tensor([0., 1.]))

    @property
    def prior(self):
        # kl-std normal: factorized std gaussian prior
        event_shape = self.encoder.event_shape
        return Independent(Normal(*self.nilone).expand(event_shape),
                           len(event_shape))

    def forward(self, input):
        # default single-draw forward pass, see Kingma and Welling (2019)
        q = self.encoder(input)
        p = self.decoder(q.rsample())

        return p, q

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch, *dontcare = batch  # classic vae: only input matters
        X = batch.unsqueeze(1)

        if self.n_draws == 1:
            # use default single-draw forward pass
            p, q = self(X)
            log_p = p.log_prob(X)

        else:
            # X is `batch x [*p.event_shape]`, q.batch_shape is `batch`
            q = self.encoder(X)

            # (sgvb)_k = E_x E_{S~q^k(z|x)} E_{z~S} log p(x|z) pi(z) / q(z|x)
            # log_p is `self.k x batch`
            log_p = torch.stack([
                self.decoder(z).log_prob(X) for z in q.rsample([self.n_draws])
            ], dim=0).mean(dim=0)

        # the components are 1d with shape `batch`
        return {
            'sgvb/loglik': log_p,
            'sgvb/kl-div': dist.kl_divergence(q, self.prior)
        }

    def training_step_end(self, outputs):
        beta = beta_scheduler(self.current_epoch, self.beta)

        ll = outputs['sgvb/loglik'].mean()
        kl = outputs['sgvb/kl-div'].mean()
        elbo = ll - beta * kl

        return {'loss': -elbo,
                'log': {
                    'sgvb/elbo': elbo,
                    'sgvb/loglik': ll,
                    'sgvb/kl-div': kl,
                    'sgvb/beta': beta,
                }}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)

        sched = torch.optim.lr_scheduler.LambdaLR(
            optim, partial(linear, t0=0, t1=self.trainer.max_epochs))

        return [optim], [{'scheduler': sched, 'monitor': 'loss'}]


class TRIPRuntime(SGVBRuntime):
    """[work-in-progress] runtime for priors that habe log-prob and are
    sampleable, but without exact closed form expression for the Kulback-
    Leibler divergence of the encoder from the prior.
    """
    def __init__(self, encoder, decoder, *, beta, lr, n_draws,
                 shape, ranks):
        super().__init__(encoder, decoder, beta=beta, lr=lr, n_draws=n_draws)
        self.trip_ = TRIP(shape, ranks)

    @property
    def prior(self):
        return self.trip_

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch, *dontcare = batch  # classic vae: only input matters
        X = batch.unsqueeze(1)

        # X is `batch x [*p.event_shape]`, q.batch_shape is `batch`
        q = self.encoder(X)

        # (sgvb)_k = E_x E_{S~q^k(z|x)} E_{z~S} log p(x|z) pi(z) / q(z|x)
        # log_p is `self.k x batch`
        sample = q.rsample([self.n_draws])
        log_p = torch.stack([
            self.decoder(z).log_prob(X) for z in sample
        ], dim=0)

        # log_p and log_prior are both `self.n_draws x batch`
        sample = sample.flatten(2, -1).flatten(0, 1)
        log_prior = self.prior.log_prob(sample).reshape_as(log_p)

        # the components are 1d with shape `batch`
        return {
            'sgvb/loglik': log_p.mean(dim=0),
            'sgvb/kl-div': - q.entropy() - log_prior.mean(dim=0)
        }


class IWAERuntime(SGVBRuntime):
    def __init__(self, encoder, decoder, *, beta, lr, n_draws,
                 naive_grad=True):
        assert n_draws > 1
        super().__init__(encoder, decoder, beta=beta, lr=lr, n_draws=n_draws)
        self.naive_grad = naive_grad

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch, *dontcare = batch  # vae: only input matters
        X = batch.unsqueeze(1)

        # X is `batch x [*p.event_shape]`, q.batch_shape is `batch`
        q = self.encoder(X)

        # (iwae)_k = E_x E_{S~q^k(z|x)} log E_{z~S} p(x|z) pi(z) / q(z|x)
        #  * like (sgvb)_k but E_z and log are interchanged
        log_p, kldiv = [], []
        for z in q.rsample([self.n_draws]):
            log_p.append(self.decoder(z).log_prob(X))
            kldiv.append(q.log_prob(z) - self.prior.log_prob(z))

        # log_p and kldiv are both `self.n_draws x batch`
        log_p, kldiv = torch.stack(log_p, dim=0), torch.stack(kldiv, dim=0)
        log_iw = log_p - kldiv

        # ASSUMPTION: `q` is reparameterizable and `z` are continuous rv
        # naïve or reverse engineered from the gradient estimator
        outputs = {}
        if self.naive_grad:
            outputs['iwae'] = torch.logsumexp(log_iw, dim=0) - log(self.n_draws)

        else:
            # `1 / n_draws` normalization is handled by softmax here
            iw = log_iw.detach().softmax(dim=0)
            outputs['iwae'] = (log_iw * iw).sum(dim=0)

        # compute sgvb for diagnostics and comparison
        with torch.no_grad():
            outputs['sgvb/loglik'] = log_p.mean(dim=0)
            outputs['sgvb/kl-div'] = dist.kl_divergence(q, self.prior)

        return outputs

    def training_step_end(self, outputs):
        sgvb = super().training_step_end(outputs)

        iwae = outputs['iwae'].mean()
        return {'loss': -iwae,
                'log': {
                    'iwae': iwae,
                    **sgvb['log']
                }}
