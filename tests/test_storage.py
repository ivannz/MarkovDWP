import io
import tempfile
import pytest

import torch

from markovdwp.utils.io import write_file


@pytest.mark.parametrize('n_tensors', [
    1, 10, 100
])
def test_buffer(n_tensors):
    tensors, buf = [], io.BytesIO()
    for _ in range(n_tensors):
        tensors.append(torch.randn(10, 10, n_tensors))
        write_file(tensors[-1], buf)

    tensor = torch.stack(tensors, dim=0)

    storage = tensor.storage().from_buffer(buf.getbuffer(), 'native')
    concat = torch.Tensor(storage).reshape_as(tensor)

    assert torch.allclose(concat, tensor)


def test_write_file():
    tensor = torch.ones(16)

    # 1. test in-memory buffer
    buf = io.BytesIO()
    write_file(tensor, buf, save_size=False)

    recon = torch.Tensor(tensor.storage().from_buffer(
        buf.getbuffer(), 'native'))
    print(recon)
    assert torch.allclose(recon, tensor)

    # 2. test in-memory buffer
    buf = io.BytesIO()
    write_file(tensor, buf, save_size=True)

    recon = torch.Tensor(tensor.storage().from_buffer(
        buf.getbuffer(), 'native', offset=8))
    assert torch.allclose(recon, tensor)

    # 3. test file buffer, forget to flush
    with pytest.raises(RuntimeError):
        with tempfile.NamedTemporaryFile() as file:
            write_file(tensor, file, save_size=False)

            # file.flush()

            assert torch.allclose(torch.Tensor(tensor.storage().from_file(
                file.name, False, tensor.numel())), tensor)

    # 4. test file buffer
    with tempfile.NamedTemporaryFile() as file:
        write_file(tensor, file, save_size=False)

        file.flush()

        assert torch.allclose(torch.Tensor(tensor.storage().from_file(
            file.name, False, tensor.numel())), tensor)
