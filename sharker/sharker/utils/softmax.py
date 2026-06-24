from typing import Optional

from mindspore import Tensor, mint
from .num_nodes import maybe_num_nodes
from . import scatter
from . import segment


def softmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    axis: int = 0,
) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`

    Examples:
        >>> src = Tensor([1., 1., 1., 1.])
        >>> index = Tensor([0, 0, 1, 2])
        >>> ptr = Tensor([0, 2, 3, 4])
        >>> softmax(src, index)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> softmax(src, None, ptr)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> src = ops.randn(4, 4)
        >>> ptr = Tensor([0, 4])
        >>> softmax(src, index, dim=-1)
        tensor([[0.7404, 0.2596, 1.0000, 1.0000],
                [0.1702, 0.8298, 1.0000, 1.0000],
                [0.7607, 0.2393, 1.0000, 1.0000],
                [0.8062, 0.1938, 1.0000, 1.0000]])
    """
    if ptr is not None and (ptr.dim() == 1 or (ptr.dim() > 1 and index is None)):
        axis = axis + src.dim() if axis < 0 else axis
        count = ptr[1:] - ptr[:-1]
        src_max = segment(src, ptr, dim=axis, reduce='max')
        src_max = src_max.repeat(count.tolist(), axis=axis)
        out = (src - src_max).exp()
        out_sum = segment(out, ptr, dim=axis, reduce='sum') + 1e-16
        out_sum = out_sum.repeat(count.tolist(), axis=axis)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src, index, axis, dim_size=N, reduce="max")
        out = src - mint.index_select(src_max, axis, index.astype("int32"))
        out = out.exp()
        out_sum = scatter(out, index, axis, dim_size=N, reduce="sum") + 1e-16
        out_sum = mint.index_select(out_sum, axis, index.astype("int32"))
    else:
        raise NotImplementedError("'softmax' requires 'index' to be specified")

    return out / out_sum   