import torch

import pytorch_lightning as pl
import torch.nn.functional as F

from functools import partial
from .utils.common import weighted_sum, prepare_log, collate, linear

from ..utils.vendor.pytorch_lightning import GradInformation


class ClassificationRuntime(GradInformation, pl.LightningModule):
    def __init__(self, core, *, coef, lr):
        super().__init__()
        self.core = core
        self.coef, self.lr = coef, lr

    def forward(self, input):
        return self.core(input)

    def training_step(self, batch, batch_idx):
        X, y = batch
        return {'y': y, 'out': self(X)}

    def training_step_end(self, outputs):
        """Step for  training w. kl-div and task (mse, hankel, acpr)."""
        y, out = outputs['y'], outputs['out']

        # 1 task related losses
        task = {
            'task.nll': F.cross_entropy(out, y, reduction='mean'),
        }

        # 2. Data independent (model) losses, e.g. KL of parameter distrib from prior
        model = {
            'model.l2_norm': sum(
                torch.sum(par * par)
                for name, par in self.core.features.named_parameters()
                if 'weight' in name
            ),
            # dwp loss here, maybe
            'model.dwp': 0.
        }
        # should be at training_step_end so that it is computed only once!

        # 3. return loss components as floats grouping stats for brevity
        value, terms = weighted_sum({**model, **task}, **self.coef)
        return {'loss': value, 'log': prepare_log(terms)}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)

        sched = torch.optim.lr_scheduler.LambdaLR(
            optim, partial(linear, t0=0, t1=self.trainer.max_epochs))

        return [optim], [{'scheduler': sched, 'monitor': 'loss'}]

    def validation_step(self, batch, batch_idx):
        X, y = batch
        return {'y': y, 'log_p': F.log_softmax(self(X), dim=-1)}

    def validation_epoch_end(self, outputs):
        return self.report_epoch_end(outputs, prefix='val')

    def report_epoch_end(self, outputs, prefix='test'):
        subprefix = prefix + ('/' if prefix else '')

        outputs = {k: torch.cat(v, dim=0) for k, v in collate(outputs).items()}

        y, log_p = outputs['y'], outputs['log_p']
        y_pred = log_p.argmax(-1)

        metrics = {
            f'{subprefix}task.nll':
                -log_p[torch.arange(len(y)), y].mean(),

            f'{subprefix}task.accuracy':
                float(sum(y_pred == y)) / len(y),
        }

        return {f'{prefix}_loss': metrics[f'{subprefix}task.nll'],
                'log': metrics}
