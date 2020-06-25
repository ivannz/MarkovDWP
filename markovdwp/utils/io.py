import gzip
import torch
import tempfile


def load(snapshot):
    with gzip.open(snapshot, 'rb') as fin:
        return torch.load(fin, map_location=torch.device('cpu'))


def as_buffer(tensor):
    # assumes that we can concatenate storage! see unit test
    with tempfile.NamedTemporaryFile('rb') as fout:
        # use the temp file as storage for torch tensor
        storage = tensor.storage().from_file(
            fout.name, shared=True, size=tensor.numel())

        torch.Tensor(storage).copy_(tensor.flatten())
        return fout.read()
