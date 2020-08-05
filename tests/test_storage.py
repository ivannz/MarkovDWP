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
    with tempfile.NamedTemporaryFile() as file:
        file.write(bytes(range(64)))

        # file.flush()
        with pytest.raises(RuntimeError):
            assert torch.allclose(torch.Tensor(tensor.storage().from_file(
                file.name, False, tensor.numel())), tensor)

        file.flush()
        with pytest.raises(AssertionError):
            assert torch.allclose(torch.Tensor(tensor.storage().from_file(
                file.name, False, tensor.numel())), tensor)

    # 4. test file buffer, but first forget to flush
    with tempfile.NamedTemporaryFile() as file:
        # force python file object interface
        write_file(tensor, file, save_size=False, is_real_file=False)

        # file.flush()
        with pytest.raises(RuntimeError):
            assert torch.allclose(torch.Tensor(tensor.storage().from_file(
                file.name, False, tensor.numel())), tensor)

        file.flush()

        assert torch.allclose(torch.Tensor(tensor.storage().from_file(
            file.name, False, tensor.numel())), tensor)


@pytest.mark.parametrize('n_tensors', [
    1, 10, 100
])
@pytest.mark.parametrize('shape', [
    [7, 3, 5],
    [17],
])
def test_storage_concat(n_tensors, shape):
    # test appending to a file
    with tempfile.NamedTemporaryFile('wb') as file:
        tensors = []
        for _ in range(n_tensors):
            tensors.append(torch.randn(*shape))
            write_file(tensors[-1], file)

        shape = torch.Size([len(tensors), *tensors[-1].shape])
        dtype = tensors[-1].dtype

        file.flush()

        # rebuild the tensor from binary data
        storage_type = torch.tensor([], dtype=dtype).storage()
        storage = storage_type.from_file(file.name, False, shape.numel())
        stored = torch.Tensor(storage).reshape(shape)

        assert torch.allclose(stored, torch.stack(tensors, dim=0))
