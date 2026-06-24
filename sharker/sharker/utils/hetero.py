from typing import Dict, List, Optional, Set, Tuple, Union

import mindspore as ms
from mindspore import Tensor, ops, mint

from .num_nodes import maybe_num_nodes_dict


def group_hetero_graph(
    edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    num_nodes_dict: Optional[Dict[str, int]] = None,
) -> Tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Dict[Union[str, int], Tensor],
    Dict[Union[str, Tuple[str, str, str]], int],
]:
    num_nodes_dict = maybe_num_nodes_dict(edge_index_dict, num_nodes_dict)

    tmp = list(edge_index_dict.values())[0]

    key2int: Dict[Union[str, Tuple[str, str, str]], int] = {}

    cumsum, offset = 0, {}  # Helper data.
    node_types, local_node_indices = [], []
    local2global: Dict[Union[str, int], Tensor] = {}
    for i, (key, N) in enumerate(num_nodes_dict.items()):
        key2int[key] = i
        node_types.append(ops.full((N,), i), dtype=tmp.dtype)
        local_node_indices.append(mint.arange(N))
        offset[key] = cumsum
        local2global[key] = local_node_indices[-1] + cumsum
        local2global[i] = local2global[key]
        cumsum += N

    node_type = mint.cat(node_types, dim=0)
    local_node_idx = mint.cat(local_node_indices, dim=0)

    edge_indices, edge_types = [], []
    for i, (keys, edge_index) in enumerate(edge_index_dict.items()):
        key2int[keys] = i
        inc = ms.Tensor([offset[keys[0]], offset[keys[-1]]]).view(2, 1)
        edge_indices.append(edge_index + inc)
        edge_types.append(ops.full((edge_index.shape[1],), i), dtype=tmp.dtype)

    edge_index = mint.cat(edge_indices, dim=-1)
    edge_type = mint.cat(edge_types, dim=0)

    return (
        edge_index,
        edge_type,
        node_type,
        local_node_idx,
        local2global,
        key2int,
    )


def get_unused_node_types(
    node_types: List[str], edge_types: List[Tuple[str, str, str]]
) -> Set[str]:
    dst_node_types = set(edge_type[-1] for edge_type in edge_types)
    return set(node_types) - set(dst_node_types)


def check_add_self_loops(
    module: ms.nn.Cell,
    edge_types: List[Tuple[str, str, str]],
) -> None:
    is_bipartite = any([key[0] != key[-1] for key in edge_types])
    if is_bipartite and getattr(module, "add_self_loops", False):
        raise ValueError(
            f"'add_self_loops' attribute set to 'True' on module '{module}' "
            f"for use with edge type(s) '{edge_types}'. This will lead to "
            f"incorrect message passing results."
        )


def construct_bipartite_edge_index(
    edge_index_dict: Dict[Tuple[str, str, str], Union[Tensor,]],
    src_offset_dict: Dict[Tuple[str, str, str], int],
    dst_offset_dict: Dict[str, int],
    edge_attr_dict: Optional[Dict[Tuple[str, str, str], Tensor]] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Union[Tensor,], Optional[Tensor]]:
    """Constructs a tensor of edge indices by concatenating edge indices
    for each edge type. The edge indices are increased by the offset of the
    source and destination nodes.

    Args:
        edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A
            dictionary holding graph connectivity information for each
            individual edge type, either as a :class:`Tensor` of
            shape :obj:`[2, num_edges]` or a
            :class:`mindGeometric_sparse.SparseTensor`.
        src_offset_dict (Dict[Tuple[str, str, str], int]): A dictionary of
            offsets to apply to the source node type for each edge type.
        dst_offset_dict (Dict[str, int]): A dictionary of offsets to apply for
            destination node types.
        edge_attr_dict (Dict[Tuple[str, str, str], Tensor]): A
            dictionary holding edge features for each individual edge type.
            (default: :obj:`None`)
        num_nodes (int, optional): The final number of nodes in the bipartite
            adjacency matrix. (default: :obj:`None`)
    """
    edge_indices: List[Tensor] = []
    edge_attrs: List[Tensor] = []
    for edge_type, src_offset in src_offset_dict.items():
        edge_index = edge_index_dict[edge_type]
        dst_offset = dst_offset_dict[edge_type[-1]]

        edge_index = edge_index.copy()

        edge_index[0] += src_offset
        edge_index[1] += dst_offset
        edge_indices.append(edge_index)

        if edge_attr_dict is not None:
            value = edge_attr_dict[edge_type]
            if value.shape[0] != edge_index.shape[1]:
                value = value.broadcast_to((edge_index.shape[1], -1))
            edge_attrs.append(value)

    edge_index = mint.cat(edge_indices, dim=1)

    edge_attr: Optional[Tensor] = None
    if edge_attr_dict is not None:
        edge_attr = mint.cat(edge_attrs, dim=0)

    return edge_index, edge_attr
