import mindspore as ms
from mindspore import Tensor, ops, mint
from mindspore import ops
from .coalesce import coalesce
from .degree import degree
from .to_dense_adj import to_dense_adj


def assortativity(edge_index: Tensor) -> float:
    r"""The degree assortativity coefficient from the
    `"Mixing patterns in networks"
    <https://arxiv.org/abs/cond-mat/0209450>`_ paper.
    Assortativity in a network refers to the tendency of nodes to
    connect with other similar nodes over dissimilar nodes.
    It is computed from Pearson correlation coefficient of the node degrees.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.

    Returns:
        The value of the degree assortativity coefficient for the input
        graph :math:`\in [-1, 1]`

    Example:
        >>> edge_index = Tensor([[0, 1, 2, 3, 2],
        ...                            [1, 2, 0, 1, 3]])
        >>> assortativity(edge_index)
        -0.666667640209198
    """
    assert isinstance(edge_index, Tensor)
    row, col = edge_index

    out_deg = degree(row, dtype=ms.int64)
    in_deg = degree(col, dtype=ms.int64)
    degrees = mint.unique(mint.cat([out_deg, in_deg]))
    mapping = row.new_zeros(degrees.max().item() + 1)
    mapping[degrees] = mint.arange(degrees.shape[0])

    # Compute degree mixing matrix (joint probability distribution) `M`
    num_degrees = degrees.shape[0]
    src_deg = mapping[out_deg[row]]
    dst_deg = mapping[in_deg[col]]

    pairs = mint.stack([src_deg, dst_deg], dim=0)
    occurrence = mint.ones(pairs.shape[1])
    pairs, occurrence = coalesce(pairs, occurrence)
    M = to_dense_adj(pairs, edge_attr=occurrence, max_num_nodes=num_degrees)[0]
    # normalization
    M /= M.sum()

    # numeric assortativity coefficient, computed by
    # Pearson correlation coefficient of the node degrees
    x = y = degrees.float()
    a, b = M.sum(0), M.sum(1)

    vara = (a * x**2).sum() - ((a * x).sum()) ** 2
    varb = (b * x**2).sum() - ((b * x).sum()) ** 2
    xy = ops.outer(x, y)
    ab = ops.outer(a, b)
    out = (xy * (M - ab)).sum() / (vara * varb).sqrt()
    return out.item()
