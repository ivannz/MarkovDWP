import os
import gzip
import torch

import json
import logging
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.lr_logger import LearningRateLogger

from torch.utils.data import DataLoader
from .base import ClassificationRuntime
from .utils.runtime import get_instance


def get_datasets(name, root, augmentation=False,
                 train_size=None, random_state=None):
    assert name in ('MNIST', 'CIFAR100')

    if name == 'MNIST':
        from .mnist.dataset import MNIST_Train as Train
        from .mnist.dataset import MNIST_Test as Test

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
    # set torch's seed
    if parameters.get('__seed__') is not None:
        seed = int(parameters['__seed__'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.manual_seed(seed)

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


# python -m 'markovdwp.source' <manifest> --gpus 2 3 
def main(manifest, gpus, tag=None, debug=False):
    breakpoint() if debug else None

    manifest = os.path.abspath(os.path.normpath(manifest))
    experiment, ext = os.path.splitext(os.path.basename(manifest))
    if not (os.path.isfile(manifest) and ext == '.json'):
        raise TypeError(manifest)

    if tag is not None:
        experiment = f'{tag}__{experiment}'

    root = os.path.dirname(manifest)

    # setup file reporting
    logging.basicConfig(
        filename=os.path.join(root, 'report.log'), style='{',
        format='{asctime:24} {levelname:12} ' +
               experiment + ' {name} {message}',
        level=logging.INFO
    )

    # read the config
    config = json.load(open(manifest, 'tr'))

    target = os.path.join(root, 'result')
    os.makedirs(target, exist_ok=True)  # once created it is kept

    # train the model
    model = train(gpus, config, TensorBoardLogger(
        os.path.join(target, 'tb-logs'), name=experiment))

    # store the model next to the manifest
    filename = os.path.join(target, experiment + '.gz')
    with gzip.open(filename, 'wb', compresslevel=9) as fout:
        torch.save({
            'state': model.state_dict(),
            'config': config
        }, fout)


parser = argparse.ArgumentParser(
    description='Run one experiment.',
    add_help=True)

parser.add_argument(
    'manifest', type=str,
    help='The manifest of the experiment.')

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
parser.set_defaults(debug=False, tag=None, gpus=None)

try:
    main(**vars(parser.parse_args()))

except Exception as e:
    logging.exception(repr(e))
    raise

else:
    logging.info('complete')
