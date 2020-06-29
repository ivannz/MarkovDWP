import torch
import torch.distributions as dist

import pytorch_lightning as pl

from torch.distributions.kl import _batch_mahalanobis

from functools import partial
from ...source.base import linear


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


class VAERuntime(pl.LightningModule):
    def __init__(self, encoder, decoder, *, beta, lr):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.beta, self.lr = beta, lr

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
