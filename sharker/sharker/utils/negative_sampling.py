import random
from typing import Optional, Tuple, Union

import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, nn, mint

from .coalesce import coalesce
from .functions import cumsum
from .degree import degree
from .loop import remove_self_loops
from .num_nodes import maybe_num_nodes


def negative_sampling(
    edge_index: Tensor,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
) -> Tensor:
    r"""Samples random negative edges of a graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int or Tuple[int, int], optional): The number of nodes,
            *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)
        num_neg_samples (int, optional): The (approximate) number of negative
            samples to return.
            If set to :obj:`None`, will try to return a negative edge for every
            positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:
        >>> # Standard usage
        >>> edge_index = Tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> negative_sampling(edge_index)
        tensor([[3, 0, 0, 3],
                [2, 3, 2, 1]])

        >>> # For bipartite graph
        >>> negative_sampling(edge_index, num_nodes=(3, 4))
        tensor([[0, 2, 2, 1],
                [2, 2, 1, 3]])
    """
    assert method in ["sparse", "dense"]

    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if isinstance(num_nodes, int):
        size = (num_nodes, num_nodes)
        bipartite = False
    else:
        size = num_nodes
        bipartite = True
        force_undirected = False

    idx, population = edge_index_to_vector(
        edge_index, size, bipartite, force_undirected
    )

    if idx.numel() >= population:
        return -ops.ones((2, 0), dtype=edge_index.dtype)

    if num_neg_samples is None:
        num_neg_samples = edge_index.shape[1]
    if force_undirected:
        num_neg_samples = num_neg_samples // 2

    prob = 1.0 - idx.numel() / population  # Probability to sample a negative.
    sample_size = int(1.1 * num_neg_samples / prob)  # (Over)-sample size.

    neg_idx: Optional[Tensor] = None
    if method == "dense":
        # The dense version creates a mask of shape `population` to check for
        # invalid samples.
        mask = ops.ones(population).bool()
        mask[idx] = False
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size)
            rnd = rnd[((mask.astype("int32"))[rnd]).astype(ms.bool_)]  # Filter true negatives.
            neg_idx = rnd if neg_idx is None else mint.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
            mask[neg_idx] = False

    else:  # 'sparse'
        # The sparse version checks for invalid samples via `np.isin`.
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size)
            mask = np.isin(rnd.numpy(), idx.numpy())  # type: ignore
            if neg_idx is not None:
                mask |= np.isin(rnd, neg_idx)
            mask = Tensor.from_numpy(mask).bool()
            rnd = rnd[~mask]
            neg_idx = rnd if neg_idx is None else mint.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break

    assert neg_idx is not None
    return vector_to_edge_index(neg_idx, size, bipartite, force_undirected)


def batched_negative_sampling(
    edge_index: Tensor,
    batch: Union[Tensor, Tuple[Tensor, Tensor]],
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
) -> Tensor:
    r"""Samples random negative edges of multiple graphs given by
    :attr:`edge_index` and :attr:`batch`.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor or Tuple[LongTensor, LongTensor]): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph connecting two different node types.
        num_neg_samples (int, optional): The number of negative samples to
            return. If set to :obj:`None`, will try to return a negative edge
            for every positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:
        >>> # Standard usage
        >>> edge_index = Tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
        >>> edge_index = ops.cat([edge_index, edge_index + 4], dim=1)
        >>> edge_index
        tensor([[0, 0, 1, 2, 4, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7]])
        >>> batch = Tensor([0, 0, 0, 0, 1, 1, 1, 1])
        >>> batched_negative_sampling(edge_index, batch)
        tensor([[3, 1, 3, 2, 7, 7, 6, 5],
                [2, 0, 1, 1, 5, 6, 4, 4]])

        >>> # For bipartite graph
        >>> edge_index1 = Tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
        >>> edge_index2 = edge_index1 + Tensor([[2], [4]])
        >>> edge_index3 = edge_index2 + Tensor([[2], [4]])
        >>> edge_index = ops.cat([edge_index1, edge_index2,
        ...                         edge_index3], dim=1)
        >>> edge_index
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]])
        >>> src_batch = Tensor([0, 0, 1, 1, 2, 2])
        >>> dst_batch = Tensor, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        >>> batched_negative_sampling(edge_index,
        ...                           (src_batch, dst_batch))
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9]])
    """
    if isinstance(batch, Tensor):
        src_batch, dst_batch = batch, batch
    else:
        src_batch, dst_batch = batch[0], batch[1]

    split = degree(src_batch[edge_index[0]], dtype=ms.int64).tolist()
    edge_indices = mint.split(edge_index, split, dim=1)

    num_src = degree(src_batch, dtype=ms.int64)
    cum_src = cumsum(num_src)[:-1]

    if isinstance(batch, Tensor):
        num_nodes = num_src.tolist()
        ptr = cum_src
    else:
        num_dst = degree(dst_batch, dtype=ms.int64)
        cum_dst = cumsum(num_dst)[:-1]

        num_nodes = mint.stack([num_src, num_dst], dim=1).tolist()
        ptr = mint.stack([cum_src, cum_dst], dim=1).unsqueeze(-1)

    neg_edge_indices = []
    for i, edge_index in enumerate(edge_indices):
        edge_index = edge_index - ptr[i]
        neg_edge_index = negative_sampling(
            edge_index, num_nodes[i], num_neg_samples, method, force_undirected
        )
        neg_edge_index += ptr[i]
        neg_edge_indices.append(neg_edge_index)

    return mint.cat(neg_edge_indices, dim=1)


def structured_negative_sampling(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
    contains_neg_self_loops: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: (LongTensor, LongTensor, LongTensor)

    Example:
        >>> edge_index = Tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> structured_negative_sampling(edge_index)
        (tensor([0, 0, 1, 2]), tensor([0, 1, 2, 3]), tensor([2, 3, 0, 2]))

    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    pos_idx = row * num_nodes + col
    if not contains_neg_self_loops:
        loop_idx = mint.arange(num_nodes) * (num_nodes + 1)
        pos_idx = mint.cat([pos_idx, loop_idx], dim=0)

    rand = ops.randint(0, num_nodes, (row.shape[0],)).long()
    neg_idx = row * num_nodes + rand

    mask = Tensor.from_numpy(np.isin(neg_idx, pos_idx)).bool()
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = ops.randint(0, num_nodes, (rest.shape[0],)).long()
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + tmp

        mask = Tensor.from_numpy(np.isin(neg_idx, pos_idx)).bool()
        rest = rest[mask]

    return edge_index[0], edge_index[1], rand


def structured_negative_sampling_feasible(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
    contains_neg_self_loops: bool = True,
) -> bool:
    r"""Returns :obj:`True` if
    :meth:`~sharker.utils.structured_negative_sampling` is feasible
    on the graph given by :obj:`edge_index`.
    :meth:`~sharker.utils.structured_negative_sampling` is infeasible
    if atleast one node is connected to all other nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: bool

    Examples:
        >>> edge_index = ms.Tensor([[0, 0, 1, 1, 2, 2, 2],
        ...                                [1, 2, 0, 2, 0, 1, 1]])
        >>> structured_negative_sampling_feasible(edge_index, 3, False)
        False

        >>> structured_negative_sampling_feasible(edge_index, 3, True)
        True
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    max_num_neighbors = num_nodes

    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    if not contains_neg_self_loops:
        edge_index, _ = remove_self_loops(edge_index)
        max_num_neighbors -= 1  # Reduce number of valid neighbors

    deg = degree(edge_index[0], num_nodes)
    # True if there exists no node that is connected to all other nodes.
    return bool(mint.all(deg < max_num_neighbors))


###############################################################################


def sample(
    population: int,
    k: int,
) -> Tensor:
    if population <= k:
        return mint.arange(population)
    else:
        return Tensor(random.sample(range(population), k))


def edge_index_to_vector(
    edge_index: Tensor,
    size: Tuple[int, int],
    bipartite: bool,
    force_undirected: bool = False,
) -> Tuple[Tensor, int]:

    row, col = edge_index.copy()

    if bipartite:  # No need to account for self-loops.
        idx = (row * size[1]) + col
        population = size[0] * size[1]
        return idx, population

    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]

        # We only operate on the upper triangular matrix:
        mask = row < col
        row, col = row[mask], col[mask]
        offset = mint.cumsum(mint.arange(1, num_nodes), dim=0)[row]
        idx = row * num_nodes + col - offset
        population = (num_nodes * (num_nodes + 1)) // 2 - num_nodes
        return idx, population

    else:
        assert size[0] == size[1]
        num_nodes = size[0]

        # We remove self-loops as we do not want to take them into account
        # when sampling negative values.
        mask = row != col
        row, col = row[mask], col[mask]
        col[row < col] -= 1
        idx = row * (num_nodes - 1) + col
        population = num_nodes * num_nodes - num_nodes
        return idx, population


def vector_to_edge_index(
    idx: Tensor,
    size: Tuple[int, int],
    bipartite: bool,
    force_undirected: bool = False,
) -> Tensor:

    if bipartite:  # No need to account for self-loops.
        row = idx.div(size[1], rounding_mode="floor")
        col = idx % size[1]
        return mint.stack([row, col], dim=0)

    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]

        offset = mint.cumsum(mint.arange(1, num_nodes), dim=0)
        end = mint.arange(num_nodes, num_nodes * num_nodes, num_nodes)
        row = ops.bucketize(idx, (end - offset).tolist(), right=True).astype(idx.dtype)
        col = (offset[row] + idx) % num_nodes
        return mint.stack([mint.cat([row, col]), mint.cat([col, row])], 0)

    else:
        assert size[0] == size[1]
        num_nodes = size[0]

        row = idx.div(num_nodes - 1, rounding_mode="floor")
        col = idx % (num_nodes - 1)
        col[row <= col] += 1
        return mint.stack([row, col], dim=0)
