import torch
import torch.distributions as dist

import pytorch_lightning as pl

from torch.distributions.kl import _batch_mahalanobis

from functools import partial
from ...source.base import linear

import wandb
import matplotlib.pyplot as plt
from ...source.utils.plotting import plot_slices


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


class VAERuntime(pl.LightningModule):
    def __init__(self, encoder, decoder, *, beta, lr, ref_x=None):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.beta, self.lr = beta, lr

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

        return {'loglik': p.log_prob(X), 'kl': dist.kl_divergence(q, pi)}

    def training_step_end(self, outputs):
        ll = outputs['loglik'].mean()
        kld = outputs['kl'].mean()

        return {'loss': -ll + self.beta * kld,
                'log': {'loglik': ll, 'kl-div': kld}}

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
            activity = q.mean.squeeze().std(0)

            # E_{x ~ p(x|z_0)} \log q(z_0|x)
            r = self.decoder(self.ref_z)
            e = self.encoder(r.sample())

        if self.current_epoch == 0:
            # commit source slices only once
            wandb.log({
                'vae_src': plot_slices(self.ref_x[:, 0], **self.kw_imshow),
            }, commit=False)

        wandb.log({
            'vae_rec': plot_slices(p.sample()[:, 0], **self.kw_imshow),
            'vae_gen': plot_slices(r.sample()[:, 0], **self.kw_imshow),
            **{f'A_u{i}': a for i, a in enumerate(activity.cpu())},
            'll_x': p.log_prob(self.ref_x).mean().cpu(),
            'll_z': e.log_prob(self.ref_z).mean().cpu(),
        }, commit=False)  # commit with the next call to pl's logger
