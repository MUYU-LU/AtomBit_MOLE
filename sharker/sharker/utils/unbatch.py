from typing import List, Optional
from mindspore import Tensor

from .functions import cumsum
from .degree import degree


def unbatch(
    src: Tensor,
    batch: Tensor,
    axis: int = 0,
    batch_size: Optional[int] = None,
) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        axis (int, optional): The dimension along which to split the :obj:`src`
             tensor. (default: :obj:`0`)
        batch_size (int, optional): The batch size. (default: :obj:`None`)

    :rtype: :class:`List[Tensor]`

    Example:
        >>> src = ops.arange(7)
        >>> batch = Tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    """
    sizes = degree(batch, batch_size).long()
    return src.split(sizes.tolist(), axis)


def unbatch_edge_index(
    edge_index: Tensor,
    batch: Tensor,
    batch_size: Optional[int] = None,
) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.
        batch_size (int, optional): The batch size. (default: :obj:`None`)

    :rtype: :class:`List[Tensor]`

    Example:
        >>> edge_index = Tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        >>> batch = Tensor([0, 0, 0, 0, 1, 1, 1])
        >>> unbatch_edge_index(edge_index, batch)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]))
    """
    deg = degree(batch, batch_size).long()
    ptr = cumsum(deg)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, batch_size).long()
    return edge_index.split(sizes.tolist(), axis=1)
