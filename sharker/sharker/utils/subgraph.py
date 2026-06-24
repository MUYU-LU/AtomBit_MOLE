from typing import List, Optional, Tuple, Union
import numpy as np
import mindspore as ms
import pdb
from mindspore import Tensor, ops, nn, mint
from .map import map_index, map_index_np
from .mask import index_to_mask, index_to_mask_np
from .num_nodes import maybe_num_nodes


def get_num_hops(model: nn.Cell) -> int:
    r"""Returns the number of hops the model is aggregating information
    from.

    .. note::

        This function counts the number of message passing layers as an
        approximation of the total number of hops covered by the model.
        Its output may not necessarily be correct in case message passing
        layers perform multi-hop aggregation, *e.g.*, as in
        :class:`~sharker.nn.conv.ChebConv`.

    Example:
        >>> class GNN(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = GCNConv(3, 16)
        ...         self.conv2 = GCNConv(16, 16)
        ...         self.lin = nn.Dense16, 2)
        ...
        ...     def construct(self, x, edge_index):
        ...         x = ops.relu(self.conv1(x, edge_index))
        ...         x = ops.relu(self.conv2(x, edge_index))
        ...         return self.lin(x)
        >>> get_num_hops(GNN())
        2
    """
    from sharker.nn.conv import MessagePassing

    num_hops = 0
    for cell in model.cells():
        if isinstance(cell, MessagePassing):
            num_hops += 1
    return num_hops


def subgraph(
    subset: Union[Tensor, np.ndarray, List[int]],
    edge_index: Union[Tensor, np.ndarray],
    edge_attr: Optional[Union[Tensor, np.ndarray]] = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    *,
    return_edge_mask: bool = False,
) -> Union[
    Tuple[Tensor, Optional[Tensor]],
    Tuple[Tensor, Optional[Tensor], Tensor],
]:
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
    ## Not support BoolTensor at the moment
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index) + 1`. (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> edge_index = Tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]])
        >>> edge_attr = Tensor, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        >>> subset = Tensor([3, 4, 5])
        >>> subgraph(subset, edge_index, edge_attr)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]))

        >>> subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]),
        tensor([False, False, False, False, False, False,  True,
                True,  True,  True,  False, False]))
    """
    if isinstance(subset, (list, tuple)):
        subset = np.array(subset, dtype=np.int64)
    elif isinstance(subset, Tensor):
        subset = subset.asnumpy()
    
    if isinstance(edge_index, Tensor):
        edge_index = edge_index.asnumpy()

    if subset.dtype != np.bool_:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask = index_to_mask_np(subset, size=num_nodes)
    else:
        num_nodes = subset.shape[0]
        node_mask = subset
        subset = np.nonzero(node_mask)[0]

    src, dst = edge_index
    edge_mask = node_mask[src] & node_mask[dst]
    edge_index = edge_index[:,edge_mask]
    
    if edge_attr is not None:
        if isinstance(edge_attr, Tensor):
            edge_attr = edge_attr.asnumpy()
        edge_attr = edge_attr[edge_mask]
    else:
        None

    if relabel_nodes:
        edge_index, _ = map_index_np(
            edge_index.reshape(-1),
            subset,
            max_index=num_nodes,
            inclusive=True,
        )
        edge_index = edge_index.reshape(2, -1)

    if return_edge_mask == True:
        return [edge_index, edge_attr, edge_mask]
    else:
        return edge_index, edge_attr


def bipartite_subgraph(
    subset: Union[Tuple[Tensor, Tensor], Tuple[List[int], List[int]]],
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    relabel_nodes: bool = False,
    size: Optional[Tuple[int, int]] = None,
    return_edge_mask: bool = False,
) -> Union[
    Tuple[Tensor, Optional[Tensor]],
    Tuple[Tensor, Optional[Tensor], Optional[Tensor]],
]:
    r"""Returns the induced subgraph of the bipartite graph
    :obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`.

    Args:
        subset (Tuple[Tensor, Tensor] or tuple([int],[int])): The nodes
            to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        size (tuple, optional): The number of nodes.
            (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> edge_index = Tensor([[0, 5, 2, 3, 3, 4, 4, 3, 5, 5, 6],
        ...                            [0, 0, 3, 2, 0, 0, 2, 1, 2, 3, 1]])
        >>> edge_attr = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        >>> subset = (Tensor([2, 3, 5]), Tensor
        >>> bipartite_subgraph(subset, edge_index, edge_attr)
        (tensor([[2, 3, 5, 5],
                [3, 2, 2, 3]]),
        tensor([ 3,  4,  9, 10]))

        >>> bipartite_subgraph(subset, edge_index, edge_attr,
        ...                    return_edge_mask=True)
        (tensor([[2, 3, 5, 5],
                [3, 2, 2, 3]]),
        tensor([ 3,  4,  9, 10]),
        tensor([False, False,  True,  True, False, False, False, False,
                True,  True,  False]))
    """
    src_subset, dst_subset = subset
    if not isinstance(src_subset, Tensor):
        src_subset = Tensor(src_subset, dtype=ms.int64)
    if not isinstance(dst_subset, Tensor):
        dst_subset = Tensor(dst_subset, dtype=ms.int64)

    src, dst = edge_index
    if src_subset.dtype != ms.bool_:
        src_size = int(src.max()) + 1 if size is None else size[0]
        src_node_mask = index_to_mask(src_subset, size=src_size)
    else:
        src_size = src_subset.shape[0]
        src_node_mask = src_subset
        src_subset = mint.nonzero(src_subset).view(-1)

    if dst_subset.dtype != ms.bool_:
        dst_size = int(dst.max()) + 1 if size is None else size[1]
        dst_node_mask = index_to_mask(dst_subset, size=dst_size)
    else:
        dst_size = dst_subset.shape[0]
        dst_node_mask = dst_subset
        dst_subset = mint.nonzero(dst_subset).view(-1)

    edge_mask = mint.logical_and(mint.index_select(src_node_mask, 0, src), mint.index_select(dst_node_mask, 0, dst))
    edge_index = (ops.masked_select(edge_index, edge_mask)).view(2, -1)
    
    if edge_attr is not None:
        edge_attr = ops.masked_select(edge_attr, edge_mask)
    else:
        None

    if relabel_nodes:
        src_index, _ = map_index(edge_index[0], src_subset, max_index=src_size, inclusive=True)
        dst_index, _ = map_index(edge_index[1], dst_subset, max_index=dst_size, inclusive=True)
        edge_index = mint.stack([src_index, dst_index], dim=0)

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


def k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = "src_to_dst",
    directed: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops.

    The :attr:`flow` argument denotes the direction of edges for finding
    :math:`k`-hop neighbors. If set to :obj:`"src_to_dst"`, then the
    method will find all neighbors that point to the initial set of seed nodes
    in :attr:`node_idx.`
    This mimics the natural flow of message passing in Graph Neural Networks.

    The method returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`Tensor`): The central seed
            node(s).
        num_hops (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (str, optional): The flow direction of :math:`k`-hop aggregation
            (:obj:`"src_to_trg"` or :obj:`"trg_to_src"`).
            (default: :obj:`"src_to_trg"`)
        directed (bool, optional): If set to :obj:`True`, will only include
            directed edges to the seed nodes :obj:`node_idx`.
            (default: :obj:`False`)

    :rtype: (:class:`Tensor`)

    Examples:
        >>> edge_index = Tensor([[0, 1, 2, 3, 4, 5],
        ...                         [2, 2, 4, 4, 6, 6]])

        >>> # Center node 6, 2-hops
        >>> subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        ...     6, 2, edge_index, relabel_nodes=True)
        >>> subset
        Tensor([2, 3, 4, 5, 6])
        >>> edge_index
        Tensor([[0, 1, 2, 3],
                [2, 2, 4, 4]])
        >>> mapping
        Tensor([4])
        >>> edge_mask
        Tensor([False, False,  True,  True,  True,  True])
        >>> subset[mapping]
        Tensor([6])

        >>> edge_index = Tensor([[1, 2, 4, 5],
        ...                         [0, 1, 5, 6]])
        >>> (subset, edge_index,
        ...  mapping, edge_mask) = k_hop_subgraph([0, 6], 2,
        ...                                       edge_index,
        ...                                       relabel_nodes=True)
        >>> subset
        tensor([0, 1, 2, 4, 5, 6])
        >>> edge_index
        tensor([[1, 2, 3, 4],
                [0, 1, 4, 5]])
        >>> mapping
        tensor([0, 5])
        >>> edge_mask
        tensor([True, True, True, True])
        >>> subset[mapping]
        tensor([0, 6])
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ["src_to_dst", "dst_to_src"]
    if flow == "dst_to_src":
        dst, src = edge_index
    else:
        src, dst = edge_index

    node_mask = mint.zeros(num_nodes, dtype=ms.bool_)
    edge_mask = mint.zeros(dst.shape[0], dtype=ms.bool_)


    if isinstance(node_idx, int):
        node_idx = Tensor([node_idx])
    elif isinstance(node_idx, (list, tuple)):
        node_idx = Tensor(node_idx)
    subsets = [node_idx]

    for _ in range(num_hops):

        node_mask = node_mask.fill(False)
        node_mask[subsets[-1]] = True
        edge_mask = mint.index_select(node_mask, 0, dst)
        subsets.append(ops.masked_select(src, edge_mask))
    
    subset, inv = mint.unique(mint.cat(subsets), return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask = node_mask.fill(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = mint.logical_and(mint.index_select(node_mask, 0, dst), mint.index_select(node_mask, 0, src))

    edge_index = (ops.masked_select(edge_index, edge_mask)).view(2, -1)

    if relabel_nodes:
        mapping = -mint.ones(num_nodes, dtype=dst.dtype)
        mapping[subset] = mint.arange(subset.shape[0])
        edge_index = mapping[edge_index]

    return subset, edge_index, inv, edge_mask


def hyper_subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    return_edge_mask: bool = False,
) -> Union[
    Tuple[Tensor, Optional[Tensor]],
    Tuple[Tensor, Optional[Tensor], Tensor],
]:
    r"""Returns the induced subgraph of the hyper graph of
    :obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`.

    Args:
        subset (Tensor or [int]): The nodes to keep.
        edge_index (LongTensor): Hyperedge tensor
            with shape :obj:`[2, num_edges*num_nodes_per_edge]`, where
            :obj:`edge_index[1]` denotes the hyperedge index and
            :obj:`edge_index[0]` denotes the node indices that are connected
            by the hyperedge.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features of shape :obj:`[num_edges, *]`.
            (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the
            resulting :obj:`edge_index` will be relabeled to hold
            consecutive indices starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index[0]) + 1`. (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> edge_index = Tensor([[0, 1, 2, 1, 2, 3, 0, 2, 3],
        ...                            [0, 0, 0, 1, 1, 1, 2, 2, 2]])
        >>> edge_attr = Tensor([3, 2, 6])
        >>> subset = Tensor
        >>> subgraph(subset, edge_index, edge_attr)
        (tensor([[0, 3],
                [0, 0]]),
        tensor([ 6.]))

        >>> subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
        (tensor([[0, 3],
                [0, 0]]),
        tensor([ 6.]))
        tensor([False, False, True])
    """
    if isinstance(subset, (list, tuple)):
        subset = Tensor(subset, dtype=ms.int64)

    if subset.dtype != ms.bool_:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask = index_to_mask(subset, size=num_nodes)
    else:
        num_nodes = subset.shape[0]
        node_mask = subset

    src, dst = edge_index
    # Mask all connections that contain a node not in the subset
    hyper_edge_mask = mint.index_select(node_mask, 0, src)


    # Mask hyperedges that contain one or less nodes from the subset
    edge_mask = ops.unsorted_segment_sum(
        hyper_edge_mask.astype(ms.int64),
        dst, dst.max() + 1) > 1

    # Mask connections if hyperedge contains one or less nodes from the subset
    # or is connected to a node not in the subset
    hyper_edge_mask = mint.logical_and(hyper_edge_mask, mint.index_select(edge_mask, 0, dst))

    src = ops.masked_select(src, hyper_edge_mask)
    dst = ops.masked_select(dst, hyper_edge_mask)
    if edge_attr is not None:
        edge_attr = ops.masked_select(edge_attr, edge_mask)
    else:
        None
    

    # Relabel edges
    edge_idx = mint.zeros(edge_mask.shape[0], dtype=ms.int64)
    edge_mask_idx = ops.argwhere(edge_mask)
    edge_mask_idx = edge_mask_idx.view(int(edge_mask.sum()),)
    edge_idx[edge_mask_idx] = mint.arange(edge_mask.sum().item())
    src, dst = mint.stack([src, edge_idx[dst]])

    if relabel_nodes:
        node_idx = mint.zeros(node_mask.shape[0], dtype=ms.int64)
        node_idx[subset] = mint.arange(node_mask.sum().item())
        src, dst = mint.stack([node_idx[src], dst])
    edge_index = mint.stack([src, dst])
    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr