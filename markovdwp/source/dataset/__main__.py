import os
import warnings

import time
import tqdm
import json
import tempfile
import argparse

from torch.nn import Conv2d
from torch.nn.modules.conv import _ConvNd

from ..utils.runtime import get_instance, get_qualname
from ...utils.io import load, write_file


def enumerate_snapshots(path, ext='.gz'):
    path, _, snapshots = next(os.walk(os.path.abspath(path)))
    for basename, extension in map(os.path.splitext, snapshots):
        if extension != ext:
            continue

        # check file size
        snapshot = os.path.join(path, basename + ext)
        if os.stat(snapshot).st_size < 1:
            continue

        yield snapshot


def conv_filters(module, cls=Conv2d, prefix=''):
    """Collect filters from the convolutional layers of the specified model."""
    assert issubclass(cls, _ConvNd)

    for name, mod in module.named_modules(prefix=prefix):
        if not isinstance(mod, cls):
            continue

        yield name, mod.weight.cpu().detach().clone()


def main(target, root, force=False, debug=False):
    breakpoint() if debug else None

    # the first snapshot is processed separately
    primary_filename, *filenames = enumerate_snapshots(root)
    snapshot = load(primary_filename)

    # determine the primary config
    master = snapshot['config']

    # load the model
    model = get_instance(**master['model'])
    model.load_state_dict(snapshot['state'], strict=True)

    # create the dataset pack based on model's qualifying name
    model_name = get_qualname(master['model']['cls'])  # use `model-cls`

    target = os.path.abspath(os.path.join(target, model_name))
    os.makedirs(target, exist_ok=force)

    # create tensor storage vault and commit the master model
    vault, sources = {}, [primary_filename]
    for name, weight in conv_filters(model):
        fid, vault[name] = tempfile.mkstemp(
            dir=target, prefix='v', suffix='.bin')
        write_file(weight, fid, save_size=False)
        os.close(fid)

    # check that all models have exactly the same config and commit them
    for filename in tqdm.tqdm(filenames, desc='fetching datasets'):
        snapshot = load(filename)
        if snapshot['config'] != master:
            # does not match the primary model
            warnings.warn(f'Config of `{filename}` does not match the'
                          f' the primary model `{primary_filename}`.',
                          RuntimeWarning)
            continue

        model.load_state_dict(snapshot['state'], strict=True)
        for name, weight in conv_filters(model):
            write_file(weight, open(vault[name], 'ab'), save_size=False)

        sources.append(filename)

    # make vault paths relative and compile meta info (from the last model)
    meta, filters = {}, list(conv_filters(model))
    common = os.path.commonpath([target, *vault.values()])
    for name, weight in tqdm.tqdm(filters, desc='saving datasets'):
        meta[name] = {
            'shape': (len(sources), *weight.shape),
            'dtype': repr(weight.dtype),
            'vault': os.path.normpath(vault[name].replace(common, '.'))
        }

    # the meta.json defines the parameters of the dataset
    with open(os.path.join(target, 'meta.json'), 'tw') as fout:
        json.dump({
            '__dttm__': time.strftime('%Y%m%d %H%M%S'),
            'dataset': meta,
            'config': master,
            'sources': sources,
        }, fout, indent=2)


parser = argparse.ArgumentParser(
    description='Preprocess model snapshots into kernel dataset.',
    add_help=True)

parser.add_argument(
    'root', type=str,
    help='The directory with stored model snapshots.')

parser.add_argument(
    'target', type=str,
    help='The path where to put preprocessed kernel dataset.')

parser.add_argument(
    '--force', required=False, dest='force', action='store_true',
    help='Overwrite experiment report if it already exists.')

parser.add_argument(
    '--debug', required=False, dest='debug', action='store_true',
    help='Enter trace mode.')

# parser.add_argument('--no-save-optim', dest='save_optim', action='store_false')
parser.set_defaults(force=False, debug=False)

main(**vars(parser.parse_args()))
