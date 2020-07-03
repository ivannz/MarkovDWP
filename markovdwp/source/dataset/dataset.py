import os
import json

import torch
import torch.nn.functional as F

from bisect import bisect_right
from torch.utils.data import Dataset


def get_dim_shape(dim):
    if not isinstance(dim, (list, tuple, str)):
        raise TypeError(f'`dim` is either `str`, a tuple, '
                        f'or `None`. Got `{dim}`.')
    assert all(d in 'moi' for d in dim)

    # 'm' is always added
    shape = [0, (1 if 'o' in dim else None), (2 if 'i' in dim else None)]
    return tuple([s for s in shape if s is not None])


def get_pad(shape, *, to):
    """Pad the shape to target size.

    Details
    -------
    Tries to pad symmetrically, but leans towards the origin
    if odd padding is required.
    """
    excess = [t - s for s, t in zip(shape, to)]
    assert all(e >= 0 for e in excess)  # safegurad
    pad = []
    # start from the last dimension and move forward (see `F.pad`)
    for exc in reversed(excess):
        div, mod = divmod(exc, 2)
        pad.extend((div, div + mod))
    # divmod guarantees that div + (div + mod) == exc, so no safeguard here
    return pad


class KernelDataset(Dataset):
    """Source convolutional kernel dataset.

    Parameters
    ----------
    root : str
        Path to preprocessed convolutional kernels from trained models.

    source : str
        Specifies convolutions of which layer to load.

    dim : str, or None, default='mio'
        Specifies dimensions along which to assume the independence:
        'i' -- input channels, 'o' -- output channels, 'm' -- trained models.
        Independence along 'm' is implicit and always enforced. For example,
        for `mo` each element in the dataset will be 'in_channels x *kernel'
        where `*kernel` is the shape of the spatial dimensions.

    min_norm : float, default=1e-2
        Filter the source dataset by the specified minimal ell_2 norm.

    Details
    -------
    See `this <https://bochang.me/blog/posts/pytorch-distributions/>`_ post.
    a thorough investigation into the semantics of these attributes. Briefly,
        * `sample_shape` independent and identically distributed
        * `batch_shape` -- independent, not necessarily identically distributed
        * `event_shape` relates to dimensionality of a **single draw**
    """

    @staticmethod
    def info(root, full=False):
        """Return basic information about the contents of the source dataset.

        Parameters
        ----------
        root : str
            Path to preprocessed convolutional kernels from trained models.

        full : bool, default=False
            Whether to return full or partial meta information. If `full`
            is `False`, then on the part related to the specifications of
            the stored dataset is returned.

        Returns
        -------
        info : dict
            The meta information dictionary.
        """
        metafile = os.path.join(root, 'meta.json')
        if not os.path.isfile(metafile):
            raise TypeError(f'Could not find `meta.json` in `{root}`.')

        meta = json.load(open(metafile, 'rt'))

        for name, dataset in meta['dataset'].items():
            vault = os.path.join(root, dataset['vault'])
            if not os.path.isfile(vault):
                raise ValueError(f'Missing vault file for dataset `{name}`.')

        return meta if full else meta['dataset']

    def __init__(self, root, source, dim='mio', min_norm=1e-2):

        root = os.path.abspath(root)
        assert os.path.isdir(root)
        self.dim = get_dim_shape('mio' if dim is None else dim)

        self.source, self.root = source, root

        # open the vault
        self.meta = self.info(root, full=True)
        dataset = self.meta['dataset'][source]

        shape = torch.Size(dataset['shape'])
        assert len(self.dim) < len(shape)

        # Sources x Out x In x *spatial
        self.kernel_size = shape[3:]
        assert self.kernel_size  # require at least 1-d convolution

        # lookup the storage type and shape
        dtype = getattr(torch, dataset['dtype'].split('.', 1)[1])
        storage_type = torch.tensor([], dtype=dtype).storage()

        # map the reshaped tensor into RAM (not shared to keep file read-only)
        storage = storage_type.from_file(os.path.join(root, dataset['vault']),
                                         shared=False, size=shape.numel())
        self.tensor = torch.Tensor(storage).reshape(*shape)

        # similar interface and semantics as in `torch.distributions`
        event_dim = tuple(i for i in range(len(shape)) if i not in self.dim)

        # kernels are independent, and, hopefully, identically distributed
        # * tensor.shape = sample_shape + batch_shape + event_shape
        self.sample_shape = shape[0],
        self.batch_shape = tuple(shape[i] for i in self.dim if i > 0)
        self.event_shape = tuple(shape[i] for i in event_dim)

        # drop `events` with small ell-2 norm
        norms = self.tensor.norm(dim=event_dim, p=2)
        self.indices = (norms >= min_norm).nonzero()

    def __getitem__(self, index):
        index = len(self) + index if index < 0 else index

        i = self.indices[index]
        source = self.tensor[i[0]]
        if self.dim == (0, 1, 2):
            return source[i[1], i[2]]

        elif self.dim == (0, 1):
            return source[i[1], :]

        elif self.dim == (0, 2):
            return source[:, i[1]]

        elif self.dim == (0,):
            return source

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        pieces = [f'source=`{self.source}`']
        pieces.append(f'kernel={tuple(self.kernel_size)}')
        pieces.append(f'dim={self.dim}')
        pieces.append(f'n_models={len(self.meta["sources"])}')
        pieces.append(f'root="{self.root}"')

        return type(self).__name__ + '(\n  ' + '\n  '.join(pieces) + '\n)'


class MutliTaskKernelDataset(KernelDataset):
    """Multitask source convolutional kernel dataset.

    Parameters
    ----------
    root : str
        Path to preprocessed convolutional kernels from trained models.

    min_norm : float, default=1e-2
        Filter the source dataset by the specified minimal ell_2 norm.

    dim : str, or None, default='mio'
        Independence assumption used to slice the dataset. Currently
        only 'mio' is supported. See `markovdwp.source.KernelDataset`
        for details.
    """

    def __init__(self, root, source=None, *, min_norm=1e-2, dim=None):
        self.dim = get_dim_shape('mio' if dim is None else dim)
        if self.dim != (0, 1, 2):
            raise ValueError(f'Multi task Source Kenrel dataset can '
                             f'work only in `mio` mode. Got `{dim}`.')

        root = os.path.abspath(root)
        assert os.path.isdir(root)
        self.root = root

        # open the vault and select the specified sources
        self.meta = self.info(root, full=True)
        if source is None:
            source = list(self.meta['dataset'].keys())

        elif isinstance(source, str):
            source = [source]

        dataset = self.meta['dataset'] = {
            src: self.meta['dataset'][src] for src in source
        }

        # require at least 1-d convolution
        shapes = [torch.Size(spec['shape']) for spec in dataset.values()]
        assert all(len(shapes[0]) == len(shape) for shape in shapes)
        assert all(max(3, len(self.dim)) < len(shape) for shape in shapes)
        assert all(shapes[0][0] == shape[0] for shape in shapes)

        # similar interface and semantics as in `torch.distributions`
        event_dim = tuple(i for i in range(len(shapes[0]))
                          if i not in self.dim)

        # kernels are independent, and, hopefully, identically distributed
        # * tensor.shape = sample_shape + batch_shape + event_shape
        self.sample_shape = shapes[0][0]
        # self.batch_shape = {k: tuple(shape[i] for i in self.dim if i > 0)
        #                     for k, shape in zip(dataset, shapes)}
        self.batch_shape = None

        # get padding based on maximal shape
        self.kernel_size = {k: shape[3:] for k, shape in zip(dataset, shapes)}
        self.event_shape = torch.Size(map(max, zip(*self.kernel_size.values())))
        self.padding = {k: get_pad(ker, to=self.event_shape)
                        for k, ker in self.kernel_size.items()}

        indptr, tensors, indices = [0], [], []
        for source, spec in dataset.items():
            # look up the storage type and shape
            dtype = getattr(torch, spec['dtype'].split('.', 1)[1])
            storage_type = torch.tensor([], dtype=dtype).storage()

            # Sources x Out x In x *spatial
            shape = torch.Size(spec['shape'])

            # map the reshaped tensor into RAM (not shared to keep file read-only)
            storage = storage_type.from_file(os.path.join(root, spec['vault']),
                                             shared=False, size=shape.numel())
            tensor = torch.Tensor(storage).reshape(*shape)

            # drop `events` with small ell-2 norm
            norms = tensor.norm(dim=event_dim, p=2)

            tensors.append(tensor)
            indices.append((norms >= min_norm).nonzero())
            indptr.append(indptr[-1] + len(indices[-1]))

        self.tensors, self.indptr = tuple(tensors), tuple(indptr)
        self.sources, self.indices = tuple(dataset), tuple(indices)

    def __getitem__(self, index):
        index = len(self) + index if index < 0 else index

        key = bisect_right(self.indptr, index) - 1  # a[:k] <= v < a[k:]
        if not (0 <= key < len(self.sources)):
            raise KeyError

        i = self.indices[key][index - self.indptr[key]]
        tensor = self.tensors[key][i[0], i[1], i[2]]

        return F.pad(tensor, self.padding[self.sources[key]]), key

    def __len__(self):
        return self.indptr[-1]

    def __repr__(self):
        pieces = [f'source=`{self.sources}`']
        pieces.append(f'kernel={tuple(self.event_shape)}')
        pieces.append(f'n_models={len(self.meta["sources"])}')
        pieces.append(f'root="{self.root}"')

        return type(self).__name__ + '(\n  ' + '\n  '.join(pieces) + '\n)'
