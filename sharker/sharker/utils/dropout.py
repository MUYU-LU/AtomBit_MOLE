from typing import Optional, Tuple

import mindspore as ms
from mindspore import Tensor, ops, nn, mint, Generator

from .subgraph import subgraph
from .num_nodes import maybe_num_nodes


def filter_adj(
    row: Tensor, col: Tensor, edge_attr: Optional[Tensor], mask: Tensor
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """_summary_

    Args:
        row (Tensor): _description_
        col (Tensor): _description_
        edge_attr (Optional[Tensor]): _description_
        mask (Tensor): _description_

    Returns:
        Tuple[Tensor, Tensor, Optional[Tensor]]: _description_
    """
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dropout_node(
    edge_index: Tensor,
    p: float = 0.5,
    num_nodes: Optional[int] = None,
    training: bool = True,
    relabel_nodes: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)
        relabel_nodes (bool, optional): If set to `True`, the resulting
            `edge_index` will be relabeled to hold consecutive indices
            starting from zero.

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

    Examples:
        >>> edge_index = Tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability has to be between 0 and 1 " f"(got {p}")

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if not training or p == 0.0:
        node_mask = mint.ones(num_nodes).bool()
        edge_mask = mint.ones(edge_index.shape[1]).bool()
        return edge_index, edge_mask, node_mask

    prob = ops.rand(num_nodes)
    node_mask = prob > p
    edge_index, _, edge_mask = subgraph(
        node_mask,
        edge_index,
        relabel_nodes=relabel_nodes,
        num_nodes=num_nodes,
        return_edge_mask=True,
    )
    return edge_index, edge_mask, node_mask


def dropout_edge(
    edge_index: Tensor,
    p: float = 0.5,
    force_undirected: bool = False,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Randomly drops edges from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    or index indicating which edges were retained, depending on the argument
    :obj:`force_undirected`.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor` or :class:`LongTensor`)

    Examples:
        >>> edge_index = Tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask = dropout_edge(edge_index)
        >>> edge_index
        tensor([[0, 1, 2, 2],
                [1, 2, 1, 3]])
        >>> edge_mask # masks indicating which edges are retained
        tensor([ True, False,  True,  True,  True, False])

        >>> edge_index, edge_id = dropout_edge(edge_index,
        ...                                    force_undirected=True)
        >>> edge_index
        tensor([[0, 1, 2, 1, 2, 3],
                [1, 2, 3, 0, 1, 2]])
        >>> edge_id # indices indicating which edges are retained
        tensor([0, 2, 4, 0, 2, 4])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability has to be between 0 and 1 " f"(got {p}")

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.shape[1], dtype=ms.bool_)
        return edge_index, edge_mask

    row, col = edge_index
    edge_mask = ops.rand(row.shape[0]) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = mint.cat([edge_index, edge_index.flip((0, ))], dim=1)
        edge_mask = edge_mask.nonzero().tile((2, 1)).squeeze()

    return edge_index, edge_mask
