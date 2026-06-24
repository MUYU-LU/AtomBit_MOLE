import mindspore as ms

from typing import Optional
from mindspore import Tensor, ops, mint

from .functions import cumsum
from . import scatter


def to_dense_adj(
    edge_index: Tensor,
    batch: Optional[Tensor] = None,
    edge_attr: Optional[Tensor] = None,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features.
            If :obj:`edge_index` contains duplicated edges, the dense adjacency
            matrix output holds the summed up entries of :obj:`edge_attr` for
            duplicated edges. (default: :obj:`None`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)
        batch_size (int, optional): The batch size. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:
        >>> edge_index = Tensor([[0, 0, 1, 2, 3],
        ...                            [0, 1, 0, 3, 0]])
        >>> batch = Tensor([0, 0, 1, 1])
        >>> to_dense_adj(edge_index, batch)
        tensor([[[1., 1.],
                [1., 0.]],
                [[0., 1.],
                [1., 0.]]])

        >>> to_dense_adj(edge_index, batch, max_num_nodes=4)
        tensor([[[1., 1., 0., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 1., 0., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]]])

        >>> edge_attr = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> to_dense_adj(edge_index, batch, edge_attr)
        tensor([[[1., 2.],
                [3., 0.]],
                [[0., 4.],
                [5., 0.]]])
    """
    edge_index = edge_index.astype("int32")
    if batch is None:
        max_index = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        # batch = edge_index.new_zeros(max_index, dtype=ms.int32)
        batch = ops.zeros(max_index, edge_index.dtype)

    if batch_size is None:
        batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1

    one = batch.new_ones(batch.shape[0])
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="sum")
    cum_nodes = cumsum(num_nodes)

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    elif (idx1.numel() > 0 and idx1.max() >= max_num_nodes) or (
        idx2.numel() > 0 and idx2.max() >= max_num_nodes
    ):
        mask = mint.logical_and((idx1 < max_num_nodes), (idx2 < max_num_nodes))

        idx0 = ops.masked_select(idx0, mask)
        idx1 = ops.masked_select(idx1, mask)
        idx2 = ops.masked_select(idx2, mask)
        edge_attr = None if edge_attr is None else ops.masked_select(edge_attr, mask)
        

    if edge_attr is None:
        edge_attr = mint.ones(idx0.numel())

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.shape)[1:]
    flattened_size = batch_size * max_num_nodes * max_num_nodes

    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    adj = scatter(edge_attr, idx, dim=0, dim_size=flattened_size, reduce="sum")
    adj = adj.reshape(size)

    return adj
