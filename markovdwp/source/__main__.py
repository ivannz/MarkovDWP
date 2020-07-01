import random  # import before anything else

import os
import gzip
import time

import json
import argparse
import tempfile  # cleverly uses a separate instance of random's PRGN

import torch  # has its own PRNGs, but torchvision also uses `random`
import numpy as np  # preemptively import to seed it's default PRNG

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.lr_logger import LearningRateLogger

from torch.utils.data import DataLoader
from .base import ClassificationRuntime
from .utils.runtime import get_instance, get_class


def get_datasets(datasets):
    return {
        name: get_instance(**klass)
        for name, klass in datasets.items()
    }


def get_dataloaders(datasets, feeds):
    return {
        feed: DataLoader(datasets[feed], **settings)
        for feed, settings in feeds.items()
    }


def get_trainer(*, gpus, logger, max_epochs=0, **kwargs):
    callbacks, checkpoint_callback = None, None
    if isinstance(kwargs.get('resume_from_checkpoint'), str) \
       and max_epochs < 1:
        # disable logging for reloaded models with no training
        logger = None

    if logger is not None:
        callbacks = [LearningRateLogger()]
        if max_epochs >= 1:
            # will inherit dirpath from logger
            checkpoint_callback = ModelCheckpoint()

    kwargs = {
        'track_grad_norm': 1,
        'val_check_interval': 1.0,
        **kwargs,
    }
    return pl.Trainer(
        gpus=gpus,
        weights_summary=None,
        terminate_on_nan=True,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
        max_epochs=max_epochs,
        **kwargs
    )


# coef=..., lr=options['lr'], max_epochs=options['lr_max_epochs']
def train(gpus, parameters, logger=None):
    pl_module = ClassificationRuntime(
        get_instance(**parameters['model']), **parameters['options'])

    feeds = get_dataloaders(
        get_datasets(parameters['dataset']), parameters['feeds'])

    pl_trainer = get_trainer(
        gpus=gpus, logger=logger, **parameters['trainer'])
    try:
        pl_trainer.fit(pl_module,
                       train_dataloader=feeds.get('train'),
                       val_dataloaders=feeds.get('test'))

    except ValueError as e:
        # it would be bettter is pl raised special excections from
        # `detect_nan_tensors`, rather than the same ValueError...
        if 'nan or inf' in str(e) or 'nan and/or inf' in str(e):
            raise FloatingPointError(str(e)) from e

        raise

    return pl_module.core.cpu().eval()


def fix_randomness(seed):
    """Fix randomness in as many relevant sources as we can.

    Details
    -------
    `torchvision` uses two sources of randomness:
        * stdlib's `random`, e.g. `RandomCrop`
        * `torch.rand()` as in `RandomHorizontalFlip`

    `torch` itself uses global ganarator and device-specific generators
    """
    if seed is None:
        return None

    # `random` is still in non-deterministic mode, so we can safely draw
    from sys import maxsize
    deterministic_seed = random.randrange(0, maxsize)

    try:
        deterministic_seed = int(seed)

    except ValueError:
        if seed != 'deterministic':
            raise TypeError('Seed can be either `None`, and int,'
                            f' or `deterministic`. Got `{seed}`.')

        # use deterministic_seed generated by `random`

    # `random.seed` accepts strings too (hashes them to use full entropy)
    random.seed(deterministic_seed)

    # seed everything else from `random`

    # torch: `torch.cuda.manual_seed_all` is called anyway (1.1+)
    torch.manual_seed(random.randrange(0, maxsize))

    # numpy's default prng seed must be between 0 and 2**32 - 1 (uint32)
    np.random.seed(random.randrange(np.iinfo(np.uint32).min,
                                    np.iinfo(np.uint32).max))

    return deterministic_seed


def generate_tags(config):
    """Make a tag from dataset and model class names."""
    dataset = get_class(config['dataset']['train']['cls'])
    model = get_class(config['model']['cls'])
    return [dataset.__name__, model.__name__]


# python -m 'markovdwp.source' <manifest> --gpus 2 3 
def main(manifest, target, gpus=[3], tag=None, seed=None, debug=False):
    breakpoint() if debug else None

    seed = fix_randomness(seed)

    manifest = os.path.abspath(os.path.normpath(manifest))
    experiment, ext = os.path.splitext(os.path.basename(manifest))
    if not (os.path.isfile(manifest) and ext == '.json'):
        raise TypeError(manifest)

    # open the config
    config = json.load(open(manifest, 'tr'))

    # save under a random name if target is a directory
    if os.path.isdir(target):
        # > Caller is responsible for deleting the [temporary] file
        # > when done with it.
        # -- We close it, but never delete it
        fid, target = tempfile.mkstemp(
            dir=target, suffix='.gz', prefix=f'{tag}__' if tag else tag)
        os.close(fid)

    elif os.path.exists(target):
        raise ValueError(f'`{target}` already exists! Refusing to proceed.')

    # train the model
    model = train(gpus, config, WandbLogger(
        tags=[*generate_tags(config)]
    ))

    # store the model next to the manifest
    with gzip.open(target, 'wb', compresslevel=9) as fout:
        torch.save({
            '__dttm__': time.strftime('%Y%m%d %H%M%S'),
            '__seed__': seed,
            'config': config,
            'state': model.state_dict(),
        }, fout)


parser = argparse.ArgumentParser(
    description='Run one experiment.',
    add_help=True)

parser.add_argument(
    'manifest', type=str,
    help='The manifest of the experiment.')

parser.add_argument(
    'target', type=str,
    help='The path where to store the trained model under a unique name.')

parser.add_argument(
    '--seed', required=False, default=None,
    help='PRNG seed to use in the experiments.')

parser.add_argument(
    '--gpus', type=int, nargs='+', required=False, default=None,
    help='Device ids to occupy by the experiment.')

parser.add_argument(
    '--tag', type=str, required=False,
    help='Optional prefix for the experiment being run.')

parser.add_argument(
    '--debug', required=False, dest='debug', action='store_true',
    help='Enter trace mode.')

# parser.add_argument('--no-save-optim', dest='save_optim', action='store_false')
parser.set_defaults(debug=False, seed=None, tag=None, gpus=None)

main(**vars(parser.parse_args()))
