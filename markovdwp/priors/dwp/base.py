import torch

import pytorch_lightning as pl

from functools import partial
from .utils.common import collate
from ..source.base import linear


import torch.distributions as dist
from torch.distributions.kl import _batch_mahalanobis


class NormalARD(dist.Normal):
    pass


@dist.kl.register_kl(dist.Normal, NormalARD)
def _kl_normal_normalard(p, q=None):
    return torch.log1p((p.loc / p.scale).square())


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

    def forward(self, input):
        q = self.encoder(input)
        p = self.decoder(q.rsample())
        return q, p

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch, *dontcare = batch  # vae: only input matters
        X = batch.unsqueeze(1)
        q, p = self(X)

        # kl-std normal: factorized std gaussian prior
        nil, one = torch.tensor(0.).to(X), torch.tensor(1.).to(X)
        pi = dist.Independent(dist.Normal(nil, one).expand(q.event_shape), 3)
#         # kl-empirical bayes ard: prior optimized over precsion
#         pi = dist.Independent(NormalARD(nil, one).expand(q.event_shape), 3)
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
