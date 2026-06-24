from typing import Dict, List, Optional, Tuple, Union
from mindspore import Tensor, nn


def trim_to_layer(
    layer: int,
    num_sampled_nodes_per_hop: Union[List[int], Dict[str, List[int]]],
    num_sampled_edges_per_hop: Union[List[int], Dict[Tuple[str, str, str], List[int]]],
    x: Union[Tensor, Dict[str, Tensor]],
    edge_index: Union[Tensor, Dict[Tuple[str, str, str], Tensor]],
    edge_attr: Optional[Union[Tensor, Dict[Tuple[str, str, str], Tensor]]] = None,
) -> Tuple[
    Union[Tensor, Dict[str, Tensor]],
    Union[Tensor, Dict[Tuple[str, str, str], Union[Tensor,]]],
    Optional[Union[Tensor, Dict[Tuple[str, str, str], Tensor]]],
]:
    r"""Trims the :obj:`edge_index` representation, node features :obj:`x` and
    edge features :obj:`edge_attr` to a minimal-sized representation for the
    current GNN layer :obj:`layer` in directed
    :class:`~sharker.loader.NeighborLoader` scenarios.

    This ensures that no computation is performed for nodes and edges that are
    not included in the current GNN layer, thus avoiding unnecessary
    computation within the GNN when performing neighborhood sampling.

    Args:
        layer (int): The current GNN layer.
        num_sampled_nodes_per_hop (List[int] or Dict[str, List[int]]): The
            number of sampled nodes per hop.
        num_sampled_edges_per_hop (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of sampled edges per hop.
        x (Tensor or Dict[str, Tensor]): The homogeneous or
            heterogeneous (hidden) node features.
        edge_index (Tensor or Dict[Tuple[str, str, str], Tensor]): The
            homogeneous or heterogeneous edge indices.
        edge_attr (Tensor or Dict[Tuple[str, str, str], Tensor], optional): The
            homogeneous or heterogeneous (hidden) edge features.
    """
    if layer <= 0:
        return x, edge_index, edge_attr

    if isinstance(num_sampled_edges_per_hop, dict):
        assert isinstance(num_sampled_nodes_per_hop, dict)

        assert isinstance(x, dict)
        x = {k: trim_feat(v, layer, num_sampled_nodes_per_hop[k]) for k, v in x.items()}

        assert isinstance(edge_index, dict)
        edge_index = {
            k: trim_adj(
                v,
                layer,
                num_sampled_nodes_per_hop[k[0]],
                num_sampled_nodes_per_hop[k[-1]],
                num_sampled_edges_per_hop[k],
            )
            for k, v in edge_index.items()
        }

        if edge_attr is not None:
            assert isinstance(edge_attr, dict)
            edge_attr = {
                k: trim_feat(v, layer, num_sampled_edges_per_hop[k])
                for k, v in edge_attr.items()
            }

        return x, edge_index, edge_attr

    assert isinstance(num_sampled_nodes_per_hop, list)

    assert isinstance(x, Tensor)
    x = trim_feat(x, layer, num_sampled_nodes_per_hop)

    assert isinstance(edge_index, Tensor)
    edge_index = trim_adj(
        edge_index,
        layer,
        num_sampled_nodes_per_hop,
        num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop,
    )

    if edge_attr is not None:
        assert isinstance(edge_attr, Tensor)
        edge_attr = trim_feat(edge_attr, layer, num_sampled_edges_per_hop)

    return x, edge_index, edge_attr


class TrimToLayer(nn.Cell):
    def construct(
        self,
        layer: int,
        num_sampled_nodes_per_hop: Optional[List[int]],
        num_sampled_edges_per_hop: Optional[List[int]],
        x: Tensor,
        edge_index: Union[Tensor, ],
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Union[Tensor,], Optional[Tensor]]:

        if not isinstance(num_sampled_nodes_per_hop, list) and isinstance(
            num_sampled_edges_per_hop, list
        ):
            raise ValueError("'num_sampled_nodes_per_hop' needs to be given")
        if not isinstance(num_sampled_edges_per_hop, list) and isinstance(
            num_sampled_nodes_per_hop, list
        ):
            raise ValueError("'num_sampled_edges_per_hop' needs to be given")

        if num_sampled_nodes_per_hop is None:
            return x, edge_index, edge_attr
        if num_sampled_edges_per_hop is None:
            return x, edge_index, edge_attr

        return trim_to_layer(
            layer,
            num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop,
            x,
            edge_index,
            edge_attr,
        )


# Helper functions ############################################################


def trim_feat(x: Tensor, layer: int, num_samples_per_hop: List[int]) -> Tensor:
    if layer <= 0:
        return x

    return x.narrow(
        axis=0,
        start=0,
        length=x.shape[0] - num_samples_per_hop[-layer],
    )


def trim_adj(
    edge_index: Union[Tensor, ],
    layer: int,
    num_sampled_src_nodes_per_hop: List[int],
    num_sampled_dst_nodes_per_hop: List[int],
    num_sampled_edges_per_hop: List[int],
) -> Union[Tensor,]:

    if layer <= 0:
        return edge_index

    if isinstance(edge_index, Tensor):
        edge_index = edge_index.narrow(
            axis=1,
            start=0,
            length=edge_index.shape[1] - num_sampled_edges_per_hop[-layer],
        )
        return edge_index
    raise ValueError(f"Unsupported 'edge_index' type '{type(edge_index)}'")
