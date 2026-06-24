from typing import Optional, Tuple, Union

import mindspore as ms
from mindspore import Tensor, ops, mint, Generator
from .functions import cumsum
from . import scatter
from .negative_sampling import negative_sampling


def shuffle_node(
    x: Tensor,
    batch: Optional[Tensor] = None,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Randomly shuffle the feature matrix :obj:`x` along the
    first dimmension.

    The method returns (1) the shuffled :obj:`x`, (2) the permutation
    indicating the orders of original nodes after shuffling.

    Args:
        x (FloatTensor): The feature matrix.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`FloatTensor`, :class:`LongTensor`)

    Example:
        >>> # Standard case
        >>> x = Tensor([[0, 1, 2],
        ...                   [3, 4, 5],
        ...                   [6, 7, 8],
        ...                   [9, 10, 11]], dtype=ms.float32)
        >>> x, node_perm = shuffle_node(x)
        >>> x
        tensor([[ 3.,  4.,  5.],
                [ 9., 10., 11.],
                [ 0.,  1.,  2.],
                [ 6.,  7.,  8.]])
        >>> node_perm
        tensor([1, 3, 0, 2])

        >>> # For batched graphs as inputs
        >>> batch = Tensor([0, 0, 1, 1])
        >>> x, node_perm = shuffle_node(x, batch)
        >>> x
        tensor([[ 3.,  4.,  5.],
                [ 0.,  1.,  2.],
                [ 9., 10., 11.],
                [ 6.,  7.,  8.]])
        >>> node_perm
        tensor([1, 0, 3, 2])
    """
    perm = ops.arange(x.shape[0]).int()
    if not training:
        return x, perm
    if batch is None:
        perm = ops.shuffle(perm.expand_dims(0).transpose()).transpose()[0]
        return x[perm], perm
    num_nodes = scatter(batch.new_ones(x.shape[0]), batch, dim=0, reduce="sum")
    ptr = cumsum(num_nodes)
    perm = mint.cat(
    [
        ops.shuffle(ops.arange(n).int().expand_dims(0).transpose()).transpose()[0] + offset
        for offset, n in zip(ptr[:-1], num_nodes)
    ]
    ).int()
    return x[perm], perm


def mask_feature(
    x: Tensor,
    p: float = 0.5,
    mode: str = "col",
    fill_value: float = 0.0,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Randomly masks feature from the feature matrix
    :obj:`x` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`x`, (2) the feature
    mask broadcastable with :obj:`x` (:obj:`mode='row'` and :obj:`mode='col'`)
    or with the same shape as :obj:`x` (:obj:`mode='all'`),
    indicating where features are retained.

    Args:
        x (FloatTensor): The feature matrix.
        p (float, optional): The masking ratio. (default: :obj:`0.5`)
        mode (str, optional): The masked scheme to use for feature masking.
            (:obj:`"row"`, :obj:`"col"` or :obj:`"all"`).
            If :obj:`mode='col'`, will mask entire features of all nodes
            from the feature matrix. If :obj:`mode='row'`, will mask entire
            nodes from the feature matrix. If :obj:`mode='all'`, will mask
            individual features across all nodes. (default: :obj:`'col'`)
        fill_value (float, optional): The value for masked features in the
            output tensor. (default: :obj:`0`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`FloatTensor`, :class:`BoolTensor`)

    Examples:
        >>> # Masked features are column-wise sampled
        >>> x = Tensor([[1, 2, 3],
        ...                   [4, 5, 6],
        ...                   [7, 8, 9]], dtype=ms.float32)
        >>> x, feat_mask = mask_feature(x)
        >>> x
        tensor([[1., 0., 3.],
                [4., 0., 6.],
                [7., 0., 9.]]),
        >>> feat_mask
        tensor([[True, False, True]])

        >>> # Masked features are row-wise sampled
        >>> x, feat_mask = mask_feature(x, mode='row')
        >>> x
        tensor([[1., 2., 3.],
                [0., 0., 0.],
                [7., 8., 9.]]),
        >>> feat_mask
        tensor([[True], [False], [True]])

        >>> # Masked features are uniformly sampled
        >>> x, feat_mask = mask_feature(x, mode='all')
        >>> x
        tensor([[0., 0., 0.],
                [4., 0., 6.],
                [0., 0., 9.]])
        >>> feat_mask
        tensor([[False, False, False],
                [True, False,  True],
                [False, False,  True]])
    """
    y = x.copy()
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Masking ratio has to be between 0 and 1 " f"(got {p}")
    if not training or p == 0.0:
        return y, mint.ones_like(y, dtype=ms.bool_)
    assert mode in ["row", "col", "all"]

    if mode == "row":
        mask = ops.rand(y.shape[0]) >= p
        mask = mask.view(-1, 1)
    elif mode == "col":
        mask = ops.rand(y.shape[1]) >= p
        mask = mask.view(1, -1)
        
    else:
        mask = ops.rand_like(y) >= p
        
    y[~mask] = fill_value
    return y, mask


def add_random_edge(
    edge_index: Tensor,
    p: float = 0.5,
    force_undirected: bool = False,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Randomly adds edges to :obj:`edge_index`.

    The method returns (1) the retained :obj:`edge_index`, (2) the added
    edge indices.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float): Ratio of added edges to the existing edges.
            (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`,
            added edges will be undirected.
            (default: :obj:`False`)
        num_nodes (int, Tuple[int], optional): The overall number of nodes,
            *i.e.* :obj:`max_val + 1`, or the number of source and
            destination nodes, *i.e.* :obj:`(max_src_val + 1, max_dst_val + 1)`
            of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`)

    Examples:
        >>> # Standard case
        >>> edge_index = Tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5)
        >>> edge_index
        tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3],
                [1, 0, 2, 1, 3, 2, 0, 2, 1]])
        >>> added_edges
        tensor([[2, 1, 3],
                [0, 2, 1]])

        >>> # The returned graph is kept undirected
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
        ...                                           force_undirected=True)
        >>> edge_index
        tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3, 0, 2, 1],
                [1, 0, 2, 1, 3, 2, 0, 2, 1, 2, 1, 3]])
        >>> added_edges
        tensor([[2, 1, 3, 0, 2, 1],
                [0, 2, 1, 2, 1, 3]])

        >>> # For bipartite graphs
        >>> edge_index = Tensor([[0, 1, 2, 3, 4, 5],
        ...                            [2, 3, 1, 4, 2, 1]])
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
        ...                                           num_nodes=(6, 5))
        >>> edge_index
        tensor([[0, 1, 2, 3, 4, 5, 3, 4, 1],
                [2, 3, 1, 4, 2, 1, 1, 3, 2]])
        >>> added_edges
        tensor([[3, 4, 1],
                [1, 3, 2]])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(
            f"Ratio of added edges has to be between 0 and 1 " f"(got '{p}')"
        )
    if force_undirected and isinstance(num_nodes, (tuple, list)):
        raise RuntimeError("`force_undirected` is not supported for `bipartite graphs`")

    if not training or p == 0.0:
        edge_index_to_add = mint.zeros([2, 0])
        return edge_index, edge_index_to_add

    edge_index_to_add = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=round(edge_index.shape[1] * p),
        force_undirected=force_undirected,
    )

    edge_index = mint.cat([edge_index, edge_index_to_add], dim=1)

    return edge_index, edge_index_to_add
