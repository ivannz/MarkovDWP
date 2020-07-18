import os
import json
import gzip
import time
import tempfile

import torch

from functools import partial
from pytorch_lightning.loggers import WandbLogger

from cplxmodule.nn.utils import named_sparsity

from markovdwp.runtime.base import BaseClassificationRuntime
from markovdwp.runtime.dwp import BaseDWPRuntime
from markovdwp.runtime.trainer import get_trainer
from markovdwp.runtime.utils.common import linear

from markovdwp.source import KernelDataset

from markovdwp.utils.runtime import get_instance
from markovdwp.utils.runtime import get_datasets, get_dataloaders

from markovdwp.utils.dicttools import flatten, unflatten, override
from markovdwp.utils.vendor.pytorch_lightning import GradInformation


def sparsity(module, threshold=-0.5, prefix=''):
    # sparsity metrics
    n_sparsity = dict(named_sparsity(module, threshold=threshold, hard=True))
    n_zer, n_par = map(sum, zip(*n_sparsity.values()))
    return {
        f'{prefix}n_par_effective': n_par - n_zer,
        f'{prefix}n_par_total': n_par,
        f'{prefix}compression': n_par / (n_par - n_zer),
        f'{prefix}sparsity': n_zer / max(n_par, 1),
    }


def reset_variational_parameters(self):
    if hasattr(self, 'log_sigma2'):
        self.log_sigma2.data.uniform_(-5, -5)


class Runtime(GradInformation, BaseDWPRuntime, BaseClassificationRuntime):
    def __init__(self, core, *, coef, lr, kind, priors, init,
                 reset_logsigma2=False):
        super().__init__(core, coef=coef, lr=lr,
                         kind=kind, priors=priors, init=init)

        if reset_logsigma2:
            self.core.apply(reset_variational_parameters)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)

        sched = torch.optim.lr_scheduler.LambdaLR(
            optim, partial(linear, t0=0, t1=self.trainer.max_epochs))

        return [optim], [{'scheduler': sched, 'monitor': 'loss'}]

    def report_epoch_end(self, outputs, prefix='test'):
        subprefix = prefix + ('/' if prefix else '')

        # sparsity metrics: (HARDCODE) fix tau at -1/2
        metrics = super().report_epoch_end(outputs, prefix=prefix)
        metrics['log'].update(sparsity(self.core, threshold=-0.5,
                                       prefix=subprefix))

        return metrics


def train(gpus, config, logger=None):
    datasets = get_datasets(config['dataset'])

    # handle automatic kl-div setting
    coef = config['options']['coef']
    kl_div = coef.get('model.kl_div')
    if kl_div == 'dataset':
        coef['model.kl_div'] = 1. / len(datasets['train'])

    # proceed
    pl_module = Runtime(
        get_instance(**config['model']), **config['options'])

    pl_trainer = get_trainer(gpus=gpus, logger=logger,
                             **config['trainer'])

    feeds = get_dataloaders(datasets, config['feeds'])

    # last-minute update of the config before training
    logger.experiment.config.update(
        flatten(config, delim='__'),
        allow_val_change=True)

    try:
        logger.watch(pl_module)

        pl_trainer.fit(pl_module,
                       train_dataloader=feeds.get('train'),
                       val_dataloaders=feeds.get('test'))

    except ValueError as e:
        # it would be better is pl raised special exceptions from
        # `detect_nan_tensors`, rather than the same ValueError...
        if 'nan or inf' in str(e) or 'nan and/or inf' in str(e):
            raise FloatingPointError(str(e)) from e

        raise

    return pl_module.core.cpu().eval()


def main(manifest, target=None, gpus=[0], debug=False, tags=None,
         priors='fixed', init='default'):
    breakpoint() if debug else None

    assert tags is None or isinstance(tags, str)
    tags = tags.split(',') if tags is not None else []

    # the manifest must be complete and ready to go
    parameters = json.load(open(manifest, 'rt'))

    # process priors and init fast args
    assert priors in ('collapsed', 'fixed', 'trainable')
    config = parameters['options']['priors']
    for name, prior in config.items():
        if isinstance(prior, dict):
            prior['kind'] = priors

    # try opening the kernel dataset and
    config = parameters['options']['init']
    if init not in ('default', 'prior'):
        missing = config.keys() - KernelDataset.info(init).keys()
        if missing:
            raise ValueError(f'Cannot find source '
                             f'distribution for `{missing}`.')
        init = {'root': init}

    # config is ready!
    if target is None:
        # do not save anything
        pass

    elif os.path.isdir(target):
        # consume the first tag
        tag, *tags = tags if tags else [None]

        # save under a random name if target is a directory
        fid, target = tempfile.mkstemp(dir=target, suffix='.gz',
                                       prefix=tag + '__' if tag else tag)
        os.close(fid)

    elif os.path.exists(target):
        raise ValueError(f'`{target}` already exists! Refusing to proceed.')

    # pl's Wandb logger uses reinit=true!
    # wandb.init(project='DWP Slice Replication Machine', reinit=False)
    logger = WandbLogger(project='DWP Slice Replication Machine', tags=tags)

    # sync with wandb's agent's arguments and rebuild the config
    logger.experiment.config.setdefaults(flatten(parameters, delim='__'))
    config = unflatten({**logger.experiment.config}, delim='__')

    # train the model
    model = train(gpus, config, logger)

    # store the model next to the manifest
    if target is not None:
        with gzip.open(target, 'wb', compresslevel=9) as fout:
            torch.save({
                '__dttm__': time.strftime('%Y%m%d %H%M%S'),
                'config': config,
                'state': model.state_dict(),
            }, fout)

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run a A classification experiments with Deep Weight '
                    'Prior.',
        add_help=True)

    parser.add_argument(
        '--manifest', type=str, required=True,
        help='The manifest of the experiment.')

    parser.add_argument(
        '--target', type=str, required=False, default=None,
        help='The path where to store the trained model under a '
             'unique name, or the name of the snapshot wherein the '
             'model will be saved, or omitted, if model snapshot is '
             'not needed.')

    parser.add_argument(
        '--gpus', type=int, nargs='+', required=False, default=None,
        help='Device ids to occupy by the experiment.')

    parser.add_argument(
        '--tags', type=str, required=False,
        help='Optional comma-separated tags for the experiment being run. '
             'If `target` is a directory, then the first tag is used as a '
             'prefix to in model names.')

    parser.add_argument(
        '--debug', required=False, dest='debug', action='store_true',
        help='Enter trace mode.')

    parser.add_argument(
        '--init', type=str, required=False, default='default',
        help='Specify how the network is initialized: `default`, '
             '`prior`, or a path to a compatible kernel dataset.')

    parser.add_argument(
        '--priors', type=str, required=False, default='fixed',
        help='Specify whether priors are `trainable`, `fixed`, '
             'or `collapsed` to Standard Factorized Gaussian.')

    parser.set_defaults(
        target=None, gpus=[0], init='default',
        priors='fixed', tags=None, debug=False)

    args, _ = parser.parse_known_args()
    main(**vars(args))
