import warnings
from typing import Any, List, Optional, Tuple, Union
import mindspore as ms
from mindspore import Tensor, ops, COOTensor, CSRTensor, mint

from .coalesce import coalesce
from .functions import cumsum


def is_sparse_tensor(src: Any) -> bool:
    r"""Returns :obj:`True` if the input :obj:`src` is a
    :class:`ms.sparse.Tensor` (in any sparse layout).

    Args:
        src (Any): The input object to be checked.
    """
    if isinstance(src, COOTensor):
        return True
    elif isinstance(src, CSRTensor):
        return True
    return False


def ptr2index(ptr: Tensor) -> Tensor:
    index = mint.arange(ptr.numel() - 1, dtype=ptr.dtype)
    return index.repeat(ptr.diff().tolist())


def index2ptr(index: Tensor, shape: Optional[int] = None) -> Tensor:
    count = index.bincount().astype(index.dtype)
    if shape is not None:
        ptr = mint.zeros(shape).astype(index.dtype)
        ptr[:len(count)] = count
    else:
        ptr = count
    return cumsum(ptr)

