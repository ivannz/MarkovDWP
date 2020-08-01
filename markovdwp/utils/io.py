import os
import gzip
import torch

import tempfile

from functools import partial


def load(snapshot):
    with gzip.open(snapshot, 'rb') as fin:
        return torch.load(fin, map_location=torch.device('cpu'))


def write_file(tensor, file, *, save_size=False, is_real_file=None,
               chunk_size=1048576):
    """Write tensor storage to a file-like object.

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor, the storage of which to write to a file.

    file : file-like, file descriptor
        The file-ilke object to write to.

    save_size : bool, default=False
        Whether to write the size of the storage before the binary data.

    chunk_size : int, default=1Mb
        The buffer size for the the fallback method.

    Details
    -------
    Reverse engineered logic from
        `/torch/csrc/generic/StorageMethods.cpp#L220-L238`

    See Also
    --------
    torch/csrc/generic/StorageMethods.cpp#L202
        * .from_buffer(buf, byte_order=('native', 'little', 'big'), count=-1, offset=0)
        * .from_file(filename, shared=False, size=0)
        * ._write_file(file, is_real_file=False, save_size=False)

    """
    if not isinstance(is_real_file, bool):
        try:
            is_real_file = file.fileno() >= 0

        except (OSError, AttributeError):
            is_real_file = isinstance(file, int)

    if save_size or torch.__version__ >= '1.5.1':
        tensor.storage()._write_file(file, is_real_file, save_size)

    else:  # save-size is false and torch earlier than 1.5
        if isinstance(file, int):
            file = os.fdopen(file)

        # assumes that we can concatenate storage! see unit test
        with tempfile.NamedTemporaryFile('rb') as fout:
            # use the temp file as storage for torch tensor
            storage = tensor.storage().from_file(
                fout.name, shared=True, size=tensor.numel())

            torch.Tensor(storage).copy_(tensor.flatten())
            for chunk in iter(partial(fout.read, chunk_size), b''):
                file.write(chunk)
