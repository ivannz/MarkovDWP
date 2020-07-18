import os
import json
import gzip
import time
import tempfile

import torch

from functools import partial

from markovdwp.source.datasets import utils

from markovdwp.runtime.trainer import get_trainer
from markovdwp.runtime.utils.common import linear
from markovdwp.runtime.utils.reconstruction_logger \
    import SliceReconstructionLogger

from markovdwp.utils.dicttools import flatten, unflatten
from markovdwp.utils.runtime import get_datasets, get_dataloaders
from markovdwp.utils.runtime import register, get_class, get_instance

from pytorch_lightning.loggers import WandbLogger
from markovdwp.utils.vendor.pytorch_lightning import GradInformation


def get_vae(encoder, decoder, options):
    return {
        'encoder': get_instance(**encoder, **options),
        'decoder': get_instance(**decoder, **options)
    }


def apply_resampling(datasets, resampling):
    """Apply the specified resampling stratefy."""
    resampled = {}
    for name, spec in resampling.items():
        if isinstance(spec, dict):
            dataset, kind = datasets[spec['source']], spec['kind']
            if kind not in ('subset', 'undersample', 'stratified'):
                raise ValueError(f'Unrecognized resampling `{kind}`')

            if kind == 'stratified':
                dataset, _ = utils.stratified_split(
                    dataset, spec['size'], spec.get('random_state'))

            elif kind == 'undersample':
                dataset, _ = utils.undersampled_split(
                    dataset, spec['size'], spec.get('random_state'))

            elif kind == 'subset':
                dataset = utils.subset(
                    dataset, spec['size'], spec.get('random_state'))

        else:
            dataset = datasets[spec]

        resampled[name] = dataset
    return resampled


def configure_optimizers(pl_module):
    optim = torch.optim.Adam(pl_module.parameters(), lr=pl_module.lr)

    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, partial(linear, t0=0, t1=pl_module.trainer.max_epochs))

    return [optim], [{'scheduler': sched, 'monitor': 'loss'}]


def train(gpus, config, logger=None):
    # select the specified runtime and attach the dynamic methods
    Runtime = register(
        '_VAERuntime', GradInformation, get_class(config['runtime']),
        configure_optimizers=configure_optimizers)

    vae = get_vae(**config['vae'])
    pl_module = Runtime(**vae, **config['options'])

    pl_trainer = get_trainer(gpus=gpus, logger=logger, **config['trainer'])

    datasets = get_datasets(config['dataset'])
    if 'resampling' in config:
        datasets = apply_resampling(datasets, config['resampling'])
    feeds = get_dataloaders(datasets, config['feeds'])

    # if 'reference' dataset exists, then we create a reconstruction logger
    if 'reference' in datasets:
        X, y, *rest = zip(*datasets['reference'])
        order = sorted(range(len(y)), key=y.__getitem__)

        ref_x = torch.stack([X[i] for i in order], dim=0)
        ref_z = pl_module.prior.sample([len(ref_x)])

        pl_trainer.callbacks.append(
            SliceReconstructionLogger(
                ref_x.unsqueeze(1), ref_z, scatter=False))

    # do the training
    try:
        pl_trainer.fit(pl_module, train_dataloader=feeds.get('train'))

    except ValueError as e:
        # it would be better is pl raised special exceptions from
        # `detect_nan_tensors`, rather than the same ValueError...
        if 'nan or inf' in str(e) or 'nan and/or inf' in str(e):
            raise FloatingPointError(str(e)) from e

        raise

    pl_module.core.cpu().eval()
    return pl_module.encoder, pl_module.decoder


def pack(module):
    from inspect import signature

    parameters = {}
    # adopt sklearn's convention: save arguments in instance's `__dict__`
    for name, p in signature(type(module)).parameters.items():
        # ignore pure var-kwargs
        if p.kind == p.VAR_KEYWORD:
            continue

        parameters[name] = getattr(module, p.name)

    return {
        'model': {'cls': str(type(module)), **parameters},
        'state': module.state_dict(),
    }


def main(manifest, target=None, gpus=[0], debug=False, tags=None):
    breakpoint() if debug else None

    assert tags is None or isinstance(tags, str)
    tags = tags.split(',') if tags is not None else []

    # the manifest must be complete and ready to go
    parameters = json.load(open(manifest, 'rt'))

    if target is None:
        # do not save anything
        pass

    elif os.path.isdir(target):
        # consume the first tag
        tag, *tags = tags if tags else [None]

        # save under a random name if target is a directory
        fid, target = tempfile.mkstemp(dir=target, suffix='.gz',
                                       prefix=tag + ('__' if tag else ''))
        os.close(fid)

    elif os.path.exists(target):
        raise ValueError(f'`{target}` already exists! Refusing to proceed.')

    # pl's Wandb logger uses reinit=true!
    # wandb.init(project='DWP Slice Replication Machine', reinit=False)
    logger = WandbLogger(project='DWP VAE Production Line', tags=tags)

    # sync with wandb's agent's arguments and rebuild the config
    logger.experiment.config.setdefaults(flatten(parameters, delim='__'))
    config = unflatten({**logger.experiment.config}, delim='__')

    # train the model
    encoder, decoder = train(gpus, config, logger)
    if target is not None:
        with gzip.open(target, 'wb', compresslevel=9) as fout:
            torch.save({
                '__dttm__': time.strftime('%Y%m%d %H%M%S'),
                'config': config,
                'encoder': pack(encoder),
                'decoder': pack(decoder),
                'prior': {},
            }, fout)

    return encoder, decoder


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train a single VAE from .models on the specified dataset.',
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

    parser.set_defaults(
        target=None, gpus=[0], tags=None, debug=False)

    args, _ = parser.parse_known_args()
    main(**vars(args))
