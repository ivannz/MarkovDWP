import os
import json

import torch
from torch.utils.data import Dataset


def get_dim_shape(dim):
    if not isinstance(dim, (list, tuple, str)):
        raise TypeError(f'`dim` is either `str`, a tuple, '
                        f'or `None`. Got `{dim}`.')
    assert all(d in 'moi' for d in dim)

    # 'm' is always added
    shape = [0, (1 if 'o' in dim else None), (2 if 'i' in dim else None)]
    return tuple([s for s in shape if s is not None])


class KernelDataset(Dataset):
    """Source convolutional kernel dataset.

    Parameters
    ----------
    root : str
        Path to preprocessed convolutional kernels from trained models.

    source : str
        Specifies convolutions of which layer to load.

    dim : str, or None, default='sio'
        Specifies dimensions alog which to assume the independence:
        input channels 'i', output channels 'o', and trained models 'm'.
        Independence along 'm' is implicit and always enforced.

    min_norm : float, default=1e-2
        Filter the source dataset by the specified minimal ell_2 norm.

    min_norm

    Details
    -------
    See `this <https://bochang.me/blog/posts/pytorch-distributions/>`_ post.
    a thorought investigation into the semantics of these attributes. Briefly,
        * `sample_shape` independent and identically distributed
        * `batch_shape` -- independent, not necessarily identically distributed
        * `event_shape` relates to dimensionality of a **single draw**
    """

    def __init__(self, root, source, dim='sio', min_norm=1e-2):

        root = os.path.abspath(root)
        assert os.path.isdir(root)
        self.dim = get_dim_shape('sio' if dim is None else dim)

        self.source, self.root = source, root

        # open the vault
        self.meta = json.load(open(os.path.join(root, 'meta.json'), 'rt'))
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
        pieces = [self.source]
        pieces.append(f'kernel={"x".join(map(str, self.kernel_size))}')
        pieces.append(f'dim={self.dim}')
        pieces.append(f'n_models={len(self.meta["sources"])}')

        return f'{type(self).__name__}({", ".join(pieces)})'
