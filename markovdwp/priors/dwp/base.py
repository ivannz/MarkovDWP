from math import log

import torch
import torch.distributions as dist

import pytorch_lightning as pl

from torch.distributions.kl import _batch_mahalanobis

from functools import partial
from ...source.base import linear

import wandb
import matplotlib.pyplot as plt
from ...source.utils.plotting import plot_slices

from ...utils.vendor.pytorch_lightning import GradInformation


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


def as_tuple(o):
    return tuple(o if isinstance(o, (list, tuple)) else (o,))


def get_range(data, r=0.05, a=0.001):
    lo, hi = float(data.min()), float(data.max())
    return lo - abs(lo) * r - a, hi + abs(hi) * r + a


def beta_scheduler(step, beta):
    if isinstance(beta, dict):
        if beta['mode'] == 'anneal':
            t1, v1 = beta['p1']
            return linear(step, t1=t1, v1=v1, t0=0, v0=0.)
        raise ValueError(f'Unknown mode `{beta["mode"]}`.')
    return beta


def scatter(data, **kwargs):
    """I dont like this. Viewing code should not reside in model logic."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(*data.cpu().numpy().T[:2], **kwargs)
    plt.close()
    return fig


class VAERuntime(GradInformation, pl.LightningModule):
    def __init__(self, encoder, decoder, *, beta, lr, ref_x=None):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.beta, self.lr = beta, lr
        self.k = 1  # default k to one in VAE, see Kingma and Welling (2019)

        self.register_buffer('nil', torch.tensor(0.))
        self.register_buffer('one', torch.tensor(1.))
        if isinstance(ref_x, torch.Tensor):
            self.register_buffer('ref_x', ref_x)

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

    # callbacks related to filter plotting
    def setup(self, stage='fit'):
        if not hasattr(self, 'ref_x'):
            sample, *_ = as_tuple(next(iter(self.train_dataloader())))
            self.register_buffer('ref_x', sample.unsqueeze(1))

        lo, hi = get_range(self.ref_x)
        self.kw_imshow = dict(vmax=hi, vmin=lo, cmap=plt.cm.coolwarm)

        if not hasattr(self, 'ref_z'):
            with torch.no_grad():
                _, pi, _ = self(self.ref_x)
            self.register_buffer('ref_z', pi.sample((len(self.ref_x),)))

    def on_epoch_end(self):
        self.zero_grad()
        self.eval()
        with torch.no_grad():
            # E_{z ~ q(z|x_0)} \log p(x_0|z)
            p, _, q = self(self.ref_x)

            # V_{x ~ data} E_{z ~ q(z|x) z}
            activity = q.mean.squeeze().std(0).flatten()

            # E_{x ~ p(x|z_0)} \log q(z_0|x)
            r = self.decoder(self.ref_z)
            e = self.encoder(r.sample())

        if self.current_epoch == 0:
            # commit source slices only once
            wandb.log({
                'task/src': plot_slices(self.ref_x[:, 0], **self.kw_imshow),
            }, commit=False)

        wandb.log({
            'diag/lat_z': scatter(q.mean.squeeze())
        }, commit=False)

        wandb.log({
            'task/rec': plot_slices(p.sample()[:, 0], **self.kw_imshow),
            'task/gen': plot_slices(r.sample()[:, 0], **self.kw_imshow),
            **{f'diag/A_u{i}': a for i, a in enumerate(activity.cpu())},
            'diag/ll_x': p.log_prob(self.ref_x).mean().cpu(),
            'diag/ll_z': e.log_prob(self.ref_z).mean().cpu(),
        }, commit=False)  # commit with the next call to pl's logger


class SGVBRuntime(VAERuntime):
    def __init__(self, encoder, decoder, *, k, beta, lr, ref_x=None):
        assert k > 1

        super().__init__(encoder=encoder, decoder=decoder,
                         beta=beta, lr=lr, ref_x=ref_x)
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
    def __init__(self, encoder, decoder, *, k, beta, lr, ref_x=None,
                 naive_grad=True):
        assert k > 1

        super().__init__(encoder=encoder, decoder=decoder,
                         beta=beta, lr=lr, ref_x=ref_x)
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
