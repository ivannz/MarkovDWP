import os
import gzip
import time
import torch

import json
import argparse
import tempfile

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.lr_logger import LearningRateLogger

from torch.utils.data import DataLoader
from .base import ClassificationRuntime
from .utils.runtime import get_instance


def get_datasets(name, root, augmentation=False,
                 train_size=None, random_state=None):
    assert name in ('MNIST', 'CIFAR100', 'CIFAR10')

    if name == 'MNIST':
        from .mnist.dataset import MNIST_Train as Train
        from .mnist.dataset import MNIST_Test as Test

    if name == 'CIFAR10':
        from .cifar.dataset import CIFAR10_Train as Train
        from .cifar.dataset import CIFAR10_Test as Test

    if name == 'CIFAR100':
        from .cifar.dataset import CIFAR100_Train as Train
        from .cifar.dataset import CIFAR100_Test as Test

    train = Train(root, augmentation=augmentation,
                  train_size=train_size,
                  random_state=random_state)

    return {'train': train, 'test': Test(root)}


def get_dataloaders(datasets, **feeds):
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
        get_datasets(**parameters['dataset']), **parameters['feeds'])

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


def sanitize_seed(seed):
    if seed is None:
        return seed

    try:
        return int(seed)

    except ValueError:
        if seed != 'deterministic':
            raise TypeError('Seed can be either `None`, and int,'
                            f' or `deterministic`. Got `{seed}`.')

        from sys import maxsize
        from random import randrange

        return randrange(0, maxsize)


# python -m 'markovdwp.source' <manifest> --gpus 2 3 
def main(manifest, target, gpus=[3], tag=None, seed=None, debug=False):
    breakpoint() if debug else None

    seed = sanitize_seed(seed)
    if seed is not None:
        # seed torch's prngs
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
            dir=target, prefix=tag, suffix='.gz')
        os.close(fid)

    elif os.path.exists(target):
        raise ValueError(f'`{target}` already exists! Refusing to proceed.')

    # train the model
    model = train(gpus, config, WandbLogger())

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
