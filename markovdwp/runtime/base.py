import torch

import pytorch_lightning as pl
import torch.nn.functional as F

from .utils.common import weighted_sum, prepare_log, collate

from ..nn.base import named_penalties


def named_l2_norm_penalties(module, reduction='sum', prefix=''):
    # dict with many keys sharing the same value (physically).
    names = [name for name, mod in module.named_modules(prefix=prefix)
             if hasattr(mod, 'weight')]
    overrides = dict.fromkeys(names, lambda m: m.weight * m.weight)

    yield from named_penalties(module, reduction=reduction,
                               prefix=prefix, penalties=overrides)


class BaseRuntime(pl.LightningModule):
    def __init__(self, core, *, coef, lr):
        super().__init__()
        self.core, self.coef, self.lr = core, coef, lr

    def forward(self, input):
        return self.core(input)

    def training_penalty(self, outputs, prefix=''):
        # subprefix = prefix + ('.' if prefix else '')
        raise NotImplementedError


class BaseClassificationRuntime(BaseRuntime):
    def training_step(self, batch, batch_idx):
        X, y = batch
        return {'y': y, 'log_p': F.log_softmax(self(X), dim=-1)}

    def training_step_end(self, outputs):
        """Step for  training w. kl-div and task (mse, hankel, acpr)."""
        y, log_p = outputs['y'], outputs['log_p']

        # 1. task related losses
        task = {
            'task.nll': F.nll_loss(log_p, y, reduction='mean'),
        }

        # 2. Data independent (model) losses, e.g. KL of parameter distrib from prior
        # should be at training_step_end so that it is computed only once!
        model = self.training_penalty(outputs, prefix='model')

        # 3. return loss components as floats grouping stats for brevity
        value, terms = weighted_sum({**model, **task}, **self.coef)

        self.log_dict(prepare_log(terms))
        return value

    validation_step = training_step

    def validation_epoch_end(self, outputs):
        self.log_dict(self.report_epoch_end(outputs, prefix='val'))

    def report_epoch_end(self, outputs, prefix='test'):
        prefixslash = prefix + ('/' if prefix else '')
        prefixscore = prefix + ('_' if prefix else '')

        outputs = {k: torch.cat(v, dim=0) for k, v in collate(outputs).items()}
        y, log_p = outputs['y'], outputs['log_p']

        metrics = {
            f'{prefixslash}task.nll':
                F.nll_loss(log_p, y, reduction='mean'),

            f'{prefixslash}task.accuracy':
                float(sum(log_p.argmax(dim=-1) == y)) / len(y)
        }

        return {f'{prefixscore}loss': metrics[f'{prefixslash}task.nll'],
                'log': metrics}
