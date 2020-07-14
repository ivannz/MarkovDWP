from math import log

import torch
import torch.distributions as dist

import pytorch_lightning as pl

from torch.distributions.kl import _batch_mahalanobis

from functools import partial
from ..source.utils.common import linear

from ..utils.vendor.pytorch_lightning import GradInformation


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


class VAERuntime(GradInformation, pl.LightningModule):
    def __init__(self, encoder, decoder, *, beta, lr):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.beta, self.lr = beta, lr
        self.k = 1  # default k to one in VAE, see Kingma and Welling (2019)

        self.register_buffer('nil', torch.tensor(0.))
        self.register_buffer('one', torch.tensor(1.))

    def forward(self, input):
        q = self.encoder(input)
        p = self.decoder(q.rsample())

        # kl-std normal: factorized std gaussian prior
        pi = dist.Independent(
            dist.Normal(self.nil, self.one).expand(q.event_shape), 3)
        return p, pi, q

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch, *dontcare = batch  # vae: only input matters
        X = batch.unsqueeze(1)
        p, pi, q = self(X)

        return {
            'sgvb/loglik': p.log_prob(X),
            'sgvb/kl': dist.kl_divergence(q, pi)
        }

    def training_step_end(self, outputs):
        beta = beta_scheduler(self.current_epoch, self.beta)

        ll = outputs['sgvb/loglik'].mean()
        kl = outputs['sgvb/kl'].mean()
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


class SGVBRuntime(VAERuntime):
    def __init__(self, encoder, decoder, *, k, beta, lr):
        assert k > 1

        super().__init__(encoder=encoder, decoder=decoder, beta=beta, lr=lr)
        self.k = k

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch, *dontcare = batch  # vae: only input matters
        X = batch.unsqueeze(1)

        q = self.encoder(X)
        pi = dist.Independent(
            dist.Normal(self.nil, self.one).expand(q.event_shape), 3)

        # (sgvb)_k = E_x E_{S~q^k(z|x)} E_{z~S} log p(x|z) pi(z) / q(z|x)
        ll = []
        for k, z in enumerate(q.rsample((self.k,))):
            ll.append(self.decoder(z).log_prob(X))

        return {
            'sgvb/loglik': sum(ll) / self.k,
            'sgvb/kl': dist.kl_divergence(q, pi)
        }

    def training_step_end(self, outputs):
        beta = beta_scheduler(self.current_epoch, self.beta)

        ll = outputs['sgvb/loglik'].mean()
        kl = outputs['sgvb/kl'].mean()
        elbo = ll - beta * kl

        return {'loss': -elbo,
                'log': {
                    'sgvb/elbo': elbo,
                    'sgvb/loglik': ll,
                    'sgvb/kl-div': kl,
                    'sgvb/beta': beta,
                }}


class IWAERuntime(VAERuntime):
    def __init__(self, encoder, decoder, *, k, beta, lr, naive_grad=True):
        assert k > 1

        super().__init__(encoder=encoder, decoder=decoder, beta=beta, lr=lr)
        self.k, self.naive_grad = k, naive_grad

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch, *dontcare = batch  # vae: only input matters
        X = batch.unsqueeze(1)

        q = self.encoder(X)
        pi = dist.Independent(
            dist.Normal(self.nil, self.one).expand(q.event_shape), 3)

        # (iwae)_k = E_x E_{S~q^k(z|x)} log E_{z~S} p(x|z) pi(z) / q(z|x)
        #  * like (sgvb)_k but E_z and log are interchanged
        ll, kl = [], []
        for k, z in enumerate(q.rsample((self.k,))):
            p = self.decoder(z)
            ll.append(p.log_prob(X))
            kl.append(pi.log_prob(z) - q.log_prob(z))

        log_iw = torch.stack([l + k for l, k in zip(ll, kl)], dim=0)

        # naïve or reverse engineered from the gradient estimator
        if self.naive_grad:
            output = {'iwae/elbo': torch.logsumexp(log_iw, dim=0) - log(self.k)}

        else:
            iw = log_iw.detach().softmax(dim=0)
            output = {'iwae/elbo': (log_iw * iw).sum(dim=0)}

        # compute sgvb for diagnostics and comparison
        with torch.no_grad():
            output['sgvb/loglik'] = sum(ll) / self.k
            output['sgvb/kl'] = dist.kl_divergence(q, pi)

        return output

    def training_step_end(self, outputs):
        beta = beta_scheduler(self.current_epoch, self.beta)

        ll = outputs['sgvb/loglik'].mean()
        kl = outputs['sgvb/kl'].mean()
        elbo = outputs['iwae/elbo'].mean()

        return {'loss': -elbo,
                'log': {
                    'iwae/elbo': elbo,
                    'sgvb/elbo': ll - beta * kl,
                    'sgvb/loglik': ll,
                    'sgvb/kl-div': kl,
                    'sgvb/beta': beta,
                }}
