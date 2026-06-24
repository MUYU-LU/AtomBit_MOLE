import warnings
from typing import List, Union

import mindspore as ms
from mindspore import Tensor, ops, mint, Generator

from .loop import remove_self_loops
from .undirected import to_undirected



def erdos_renyi_graph(
    num_nodes: int,
    edge_prob: float,
    directed: bool = False,
) -> Tensor:
    r"""Returns the :obj:`edge_index` of a random Erdos-Renyi graph.

    Args:
        num_nodes (int): The number of nodes.
        edge_prob (float): Probability of an edge.
        directed (bool, optional): If set to :obj:`True`, will return a
            directed graph. (default: :obj:`False`)

    Examples:
        >>> erdos_renyi_graph(5, 0.2, directed=False)
        tensor([[0, 1, 1, 4],
                [1, 0, 4, 1]])

        >>> erdos_renyi_graph(5, 0.2, directed=True)
        tensor([[0, 1, 3, 3, 4, 4],
                [4, 3, 1, 2, 1, 3]])
    """
    if directed:
        idx = mint.arange((num_nodes - 1) * num_nodes)
        idx = idx.view(num_nodes - 1, num_nodes)
        idx = idx + mint.arange(1, num_nodes).view(-1, 1)
        idx = idx.view(-1)
    else:
        warnings.filterwarnings("ignore", ".*pass the indexing argument.*")
        idx = ops.combinations(mint.arange(num_nodes), r=2)

    # Filter edges.
    mask = ops.rand(idx.shape[0]) < edge_prob

    if not mask.any():
        return idx.T[:, mask.T]
    idx = idx[mask]

    if directed:
        row = idx.div(num_nodes, rounding_mode="floor")
        col = idx % num_nodes
        edge_index = mint.stack([row, col], dim=0)
    else:
        edge_index = to_undirected(idx.t(), num_nodes=num_nodes)
    print(" erdos_renyi_grapht edge index-----: ", edge_index)
    return edge_index


def barabasi_albert_graph(num_nodes: int, num_edges: int) -> Tensor:
    r"""Returns the :obj:`edge_index` of a Barabasi-Albert preferential
    attachment model, where a graph of :obj:`num_nodes` nodes grows by
    attaching new nodes with :obj:`num_edges` edges that are preferentially
    attached to existing nodes with high degree.

    Args:
        num_nodes (int): The number of nodes.
        num_edges (int): The number of edges from a new node to existing nodes.

    Example:
        >>> barabasi_albert_graph(num_nodes=4, num_edges=3)
        tensor([[0, 0, 0, 1, 1, 2, 2, 3],
                [1, 2, 3, 0, 2, 0, 1, 0]])
    """
    assert num_edges > 0 and num_edges < num_nodes

    row, col = mint.arange(num_edges), ops.shuffle(mint.arange(num_edges))

    for i in range(num_edges, num_nodes):
        row = mint.cat([row, ops.full((num_edges,), i).long()])
        choice = ops.shuffle(mint.cat([row, col]))[:num_edges]
        col = mint.cat([col, choice])

    edge_index = mint.stack([row, col], dim=0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    return edge_index
