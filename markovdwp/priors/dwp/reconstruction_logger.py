import torch
import wandb

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from ...source.utils.plotting import plot_slices


def as_tuple(o):
    return tuple(o if isinstance(o, (list, tuple)) else (o,))


def get_range(data, r=0.05, a=0.001):
    lo, hi = float(data.min()), float(data.max())
    return lo - abs(lo) * r - a, hi + abs(hi) * r + a


def scatter(data, **kwargs):
    """I dont like this. Viewing code should not reside in model logic."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    ax.scatter(*data.cpu().numpy().T[:2], **kwargs)

    plt.close()
    return fig


def forward(pl_module, x, sample=True):
    # make sure to re-enable train mode and grads!
    assert isinstance(pl_module, pl.LightningModule)

    pl_module.eval()
    with torch.no_grad():
        # E_{z ~ q(z|x_0)} \log p(x_0|z)
        q = pl_module.encoder(x.to(pl_module.device))
        p = pl_module.decoder(q.sample() if sample else q.mean)
    pl_module.train()

    return p, q


def reverse(pl_module, z, sample=True):
    # make sure to re-enable train mode and grads!
    assert isinstance(pl_module, pl.LightningModule)

    pl_module.eval()
    with torch.no_grad():
        # E_{x ~ p(x|z_0)} \log q(z_0|x)
        p = pl_module.decoder(z.to(pl_module.device))
        q = pl_module.encoder(p.sample() if sample else p.mean)
    pl_module.train()

    return p, q


class SliceReconstructionLogger(Callback):
    def __init__(self, ref_x, ref_z=None, scatter=True, sample=True, **imshow):
        if ref_x.dim() == 3:
            ref_x = ref_x.unsqueeze(1)
        assert ref_x.dim() == 4 and ref_x.shape[1] == 1

        lo, hi = get_range(ref_x)
        self.imshow = dict(vmax=hi, vmin=lo, **imshow)

        self.ref_x, self.ref_z = ref_x, ref_z
        self.scatter, self.sample = scatter, sample

    # callbacks related to filter plotting
    def on_train_start(self, trainer, pl_module):
        # commit source slices only once
        wandb.log({
            'task/src': plot_slices(self.ref_x[:, 0], **self.imshow),
        }, commit=False)

    def on_epoch_end(self, trainer, pl_module):
        # A training epoch has just finished
        p, q = forward(pl_module, self.ref_x, sample=self.sample)
        rec = p.sample() if self.sample else p.mean

        # V_{x ~ data} E_{z ~ q(z|x) z}
        activity = q.mean.squeeze().std(0).flatten().cpu()
        output = {
            'task/rec': plot_slices(rec[:, 0], **self.imshow),
            'diag/ll_x': p.log_prob(self.ref_x).mean().cpu(),
            **{f'diag/A_u{i}': a for i, a in enumerate(activity)},
        }
        if self.scatter:
            output['diag/lat_z'] = scatter(q.mean.squeeze())

        if self.ref_z is not None:
            r, e = reverse(pl_module, self.ref_z, sample=self.sample)
            gen = r.sample() if self.sample else r.mean

            output.update({
                'task/gen': plot_slices(gen[:, 0], **self.imshow),
                'diag/ll_z': e.log_prob(self.ref_z).mean().cpu(),
            })

        # commit with the next call to pl's logger
        wandb.log(output, commit=False)
