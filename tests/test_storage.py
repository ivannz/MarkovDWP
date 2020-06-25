import pytest

import torch

from markovdwp.utils.io import as_buffer


@pytest.mark.parametrize('n_tensors', [
    1, 10, 100
])
def test_buffer(n_tensors):
    pairs = []
    for _ in range(n_tensors):
        t = torch.randn(10, 10)
        pairs.append((t, as_buffer(t)))

    tensors, buffers = zip(*pairs)

    tensor = torch.stack(tensors, dim=0)

    storage = tensor.storage().from_buffer(b''.join(buffers), 'little')

    concat = torch.Tensor(storage).reshape_as(tensor)
    assert torch.allclose(concat, tensor)
