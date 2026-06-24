from typing import Optional, Tuple

import mindspore as ms
from mindspore import Tensor, ops, nn, mint
from .coalesce import coalesce


def grid(
    height: int,
    width: int,
    dtype: Optional[ms.Type] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Returns the edge indices of a two-dimensional grid graph with height
    :attr:`height` and width :attr:`width` and its node positions.

    Args:
        height (int): The height of the grid.
        width (int): The width of the grid.
        dtype (ms.Type, optional): The desired data type of the returned
            position tensor. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Example:
        >>> (row, col), pos = grid(height=2, width=2)
        >>> row
        tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        >>> col
        tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
        >>> pos
        tensor([[0., 1.],
                [1., 1.],
                [0., 0.],
                [1., 0.]])
    """
    edge_index = grid_index(height, width)
    pos = grid_pos(height, width, dtype)
    return edge_index, pos


def grid_index(
    height: int,
    width: int,
) -> Tensor:

    w = width
    kernel = ops.Tensor([-w - 1, -1, w - 1, -w, 0, w, -w + 1, 1, w + 1])

    row = mint.arange(height * width).long()
    row = row.view(-1, 1).tile((1, kernel.shape[0]))
    col = row + kernel.view(1, -1)
    row, col = row.view(height, -1), col.view(height, -1)
    index = mint.arange(3, row.shape[-1] - 3).long()
    row, col = row[:, index].view(-1), col[:, index].view(-1)

    mask = mint.logical_and((col >= 0), (col < height * width))
    row, col = row[mask], col[mask]

    edge_index = mint.stack([row, col], dim=0)
    edge_index = coalesce(edge_index, num_nodes=height * width)
    return edge_index


def grid_pos(height: int, width: int, dtype: Optional[ms.Type] = None) -> Tensor:

    dtype = ms.float32 if dtype is None else dtype
    x = mint.arange(width, dtype=dtype)
    y = (height - 1) - mint.arange(height, dtype=dtype)

    x = x.tile((height,))
    y = y.unsqueeze(-1).tile((1, width)).view(-1)

    return mint.stack([x, y], dim=-1)

