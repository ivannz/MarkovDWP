from math import log

import torch
import torch.distributions as dist
from torch.distributions import Independent, Normal

import pytorch_lightning as pl

from torch.distributions.kl import _batch_mahalanobis

from functools import partial
from .utils.common import linear

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

        # zero and one for the std Gaussian prior
        self.register_buffer('nilone', torch.tensor([0., 1.]))

    def forward(self, input):
        q = self.encoder(input)
        p = self.decoder(q.rsample())

        # kl-std normal: factorized std gaussian prior
        pi = Independent(
            Normal(*self.nilone).expand(q.event_shape), len(q.event_shape))

        return p, pi, q

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch, *dontcare = batch  # vae: only input matters
        X = batch.unsqueeze(1)

        p, pi, q = self(X)

        # the components are 1d with shape `batch`
        return {
            'sgvb/loglik': p.log_prob(X),
            'sgvb/kl-div': dist.kl_divergence(q, pi)
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


class SGVBRuntime(VAERuntime):
    def __init__(self, encoder, decoder, *, k, beta, lr):
        assert k > 1

        super().__init__(encoder=encoder, decoder=decoder, beta=beta, lr=lr)
        self.k = k

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch, *dontcare = batch  # vae: only input matters
        X = batch.unsqueeze(1)

        # X is `batch x [*p.event_shape]`, q.batch_shape is `batch`
        q = self.encoder(X)
        pi = Independent(
            Normal(*self.nilone).expand(q.event_shape), len(q.event_shape))

        # (sgvb)_k = E_x E_{S~q^k(z|x)} E_{z~S} log p(x|z) pi(z) / q(z|x)
        # log_p is `self.k x batch`
        log_p = torch.stack([
            self.decoder(z).log_prob(X) for z in q.rsample([self.k])
        ], dim=0)

        return {
            'sgvb/loglik': log_p.mean(dim=0),
            'sgvb/kl-div': dist.kl_divergence(q, pi)
        }


class IWAERuntime(VAERuntime):
    def __init__(self, encoder, decoder, *, k, beta, lr, naive_grad=True):
        assert k > 1

        super().__init__(encoder=encoder, decoder=decoder, beta=beta, lr=lr)
        self.k, self.naive_grad = k, naive_grad

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch, *dontcare = batch  # vae: only input matters
        X = batch.unsqueeze(1)

        # X is `batch x [*p.event_shape]`, q.batch_shape is `batch`
        q = self.encoder(X)
        pi = Independent(
            Normal(*self.nilone).expand(q.event_shape), len(q.event_shape))

        # (iwae)_k = E_x E_{S~q^k(z|x)} log E_{z~S} p(x|z) pi(z) / q(z|x)
        #  * like (sgvb)_k but E_z and log are interchanged
        log_p, kldiv = [], []
        for z in q.rsample([self.k]):
            log_p.append(self.decoder(z).log_prob(X))
            kldiv.append(pi.log_prob(z) - q.log_prob(z))

        # log_p and kldiv are both `self.k x batch`
        log_p, kldiv = torch.stack(log_p, dim=0), torch.stack(kldiv, dim=0)
        log_iw = log_p + kldiv

        # naïve or reverse engineered from the gradient estimator
        if self.naive_grad:
            output = {'iwae': torch.logsumexp(log_iw, dim=0) - log(self.k)}

        else:
            # 1/k normalization is handled by softmax here
            iw = log_iw.detach().softmax(dim=0)
            output = {'iwae': (log_iw * iw).sum(dim=0)}

        # compute sgvb for diagnostics and comparison
        with torch.no_grad():
            output['sgvb/loglik'] = log_p.mean(dim=0)
            output['sgvb/kl-div'] = dist.kl_divergence(q, pi)

        return output

    def training_step_end(self, outputs):
        sgvb = super().training_step_end(self, outputs)

        iwae = outputs['iwae'].mean()
        return {'loss': -iwae,
                'log': {
                    'iwae': iwae,
                    **sgvb['log']
                }}
