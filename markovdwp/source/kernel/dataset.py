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

    def __init__(self, root, source, *, dim='mio', min_norm=1e-2):

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

        # figure out the label associated to this source
        labels = {k: i for i, k in enumerate(self.meta['dataset'])}
        self.label = labels[self.source]

    def __getitem__(self, index):
        index = len(self) + index if index < 0 else index

        i = self.indices[index]
        source = self.tensor[i[0]]
        if self.dim == (0, 1, 2):
            return source[i[1], i[2]], self.label

        elif self.dim == (0, 1):
            return source[i[1], :], self.label

        elif self.dim == (0, 2):
            return source[:, i[1]], self.label

        elif self.dim == (0,):
            return source, self.label

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        pieces = [f'source=`{self.source}`']
        pieces.append(f'kernel={tuple(self.kernel_size)}')
        pieces.append(f'dim={self.dim}')
        pieces.append(f'n_models={len(self.meta["sources"])}')
        pieces.append(f'root="{self.root}"')

        return type(self).__name__ + '(\n  ' + '\n  '.join(pieces) + '\n)'


class LabelledKernelDataset(KernelDataset):
    """Labelled source convolutional kernel dataset.

    Parameters
    ----------
    root : str
        Path to preprocessed convolutional kernels from trained models.

    sources : str, list, or None
        Specifies convolutions of which layers to load. `str` loads
        convolutions from a single layer, `list` -- from several lists,
        `None` loads from all available layers. Class labels of each slice
        are consistent and determined uniquely byt the metainfo of the
        dataset under `root`.

    min_norm : dict of floats, or float, default=1e-2
        Filter the source datasets by the specified minimal ell_2 norm.
        Different thershold can be specified for each source.

    dim : str, or None, default='mio'
        Independence assumption used to slice the dataset. Currently
        only 'mio' is supported. See `markovdwp.source.KernelDataset`
        for details.

    pad : 'full' or 'selection'
        If 'full', then pad kernels to the spatially largest convolution kernel
        in the __whole__ dataset. If 'selection', then kernels are padded only
        up to the spatially largest in the selected subset. This parameter
        affects `event_shape`.
    """

    def __init__(self, root, sources=None, *, min_norm=1e-2, dim=None,
                 pad='selection'):
        assert pad in ('full', 'selection')

        self.dim = get_dim_shape('mio' if dim is None else dim)
        if self.dim != (0, 1, 2):
            raise ValueError(f'Labelled Source Kenrel dataset can '
                             f'work only in `mio` mode. Got `{dim}`.')

        root = os.path.abspath(root)
        assert os.path.isdir(root)
        self.root = root

        # open the vault and compute the parameters for all sources
        self.meta = self.info(root, full=True)

        # figure out the min_norm setting
        if isinstance(min_norm, float):
            min_norm = dict.fromkeys(self.meta['dataset'], min_norm)
        elif not isinstance(min_norm, dict):
            raise TypeError(f'`min_norm` must either be a dict of floats '
                            f'or a float. Got `{type(min_norm).__name__}`.')

        missing = min_norm.keys() - self.meta['dataset'].keys()
        if missing:
            raise ValueError(f'`min_norm` contains the following '
                             f'unrecognized sources `{missing}`.')

        # select the specified sources
        if sources is None:
            sources = list(self.meta['dataset'].keys())

        elif isinstance(sources, str):
            sources = [sources]

        assert all(src in min_norm for src in sources)
        self.min_norm = min_norm

        # prepare the dataset details
        dataset = self.meta['dataset']
        shapes = {k: torch.Size(spec['shape']) for k, spec in dataset.items()}

        # check the source dataset shapes for consistency
        first, *_ = shapes.values()
        assert all(len(first) == len(shape) for shape in shapes.values())
        self.event_dim = tuple(i for i in range(len(first))
                               if i not in self.dim)

        # require at least 1-d convolution
        self.kernel_size = {k: shape[3:] for k, shape in shapes.items()}
        assert all(self.kernel_size.values())

        # kernels are independent, and, hopefully, identically distributed
        # * tensor.shape = sample_shape + batch_shape + event_shape
        sample_dim, *batch_dim = self.dim
        assert all(first[0] == shape[0] for shape in shapes.values())

        # similar interface and semantics as in `torch.distributions`
        self.sample_shape = first[sample_dim]
        self.batch_shape = {k: tuple(first[i] for i in batch_dim)
                            for k, shape in shapes.items()}

        # Determine which kernels sizes should be considered
        if pad == 'selection':
            kernels = {k: size for k, size in self.kernel_size.items()
                       if k in sources}

        elif pad == 'full':
            kernels = self.kernel_size

        # get padding based on the spatially largest kernel
        self.event_shape = torch.Size(map(max, zip(*kernels.values())))
        self.padding = {k: get_pad(ker, to=self.event_shape)
                        for k, ker in kernels.items()}

        indptr, tensors, indices = [0], [], []
        for source in sources:
            spec = self.meta['dataset'][source]

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
            norms = tensor.norm(dim=self.event_dim, p=2)

            tensors.append(tensor)
            indices.append((norms >= self.min_norm[source]).nonzero())
            indptr.append(indptr[-1] + len(indices[-1]))

        # index to source interval mapping in split key-val form
        self.sources, self.indptr = tuple(sources), tuple(indptr)

        # other dataset attributes are dicts
        self.tensors = dict(zip(self.sources, tensors))
        self.indices = dict(zip(self.sources, indices))

        # map source names to class labels in order of enumeration in `meta`
        self.labels = {k: i for i, k in enumerate(self.meta['dataset'])}

        # create a `targets` array for compatibility with startified_split
        self.targets = torch.full((self.indptr[-1],), 0, dtype=torch.uint8)
        for src, m, n in zip(self.sources, self.indptr[1:], self.indptr):
            self.targets[n:m] = self.labels[src]

    def __getitem__(self, index):
        index = len(self) + index if index < 0 else index

        key = bisect_right(self.indptr, index) - 1  # a[:k] <= v < a[k:]
        if not (0 <= key < len(self.sources)):
            raise IndexError

        source = self.sources[key]
        ix = self.indices[source][index - self.indptr[key]]
        tensor = self.tensors[source][ix[0], ix[1], ix[2]]
        return F.pad(tensor, self.padding[source]), self.labels[source]

    def __len__(self):
        return self.indptr[-1]

    def __repr__(self):
        pieces = [f'source=`{self.sources}`']
        pieces.append(f'kernel={tuple(self.event_shape)}')
        pieces.append(f'n_models={len(self.meta["sources"])}')
        pieces.append(f'root="{self.root}"')

        return type(self).__name__ + '(\n  ' + '\n  '.join(pieces) + '\n)'
