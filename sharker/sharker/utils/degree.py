from typing import Optional

import mindspore as ms
from mindspore import Tensor, ops, nn, mint
from .num_nodes import maybe_num_nodes


def degree(
    index: Tensor, num_nodes: Optional[int] = None, dtype: Optional[ms.Type] = None
) -> Tensor:
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (Tensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`ms.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`

    Example:
        >>> row = Tensor([0, 1, 0, 2, 0])
        >>> degree(row, dtype=ms.int64)
        tensor([3, 1, 1])
    """
    N = maybe_num_nodes(index, num_nodes)
    one = mint.ones((index.shape[0],), dtype=dtype)
    out = ops.unsorted_segment_sum(one, index, N)
    return out
