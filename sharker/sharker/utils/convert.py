from collections import defaultdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import scipy.sparse
import mindspore as ms
from mindspore import Tensor, ops, mint

import sharker
from .num_nodes import maybe_num_nodes


def to_tensor(data):
    if isinstance(data, tuple):
        return (to_tensor(d) for d in data)
    elif isinstance(data, list):
        return [to_tensor(d) for d in data]
    elif isinstance(data, set):
        return set([to_tensor(d) for d in data])
    elif isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return Tensor.from_numpy(data)
    elif isinstance(data, Tensor):
        return data
    else:
        raise NotImplementedError("Datatype {} cannot be cast to tensor!")


def to_array(data):
    if isinstance(data, tuple):
        return (to_array(d) for d in data)
    elif isinstance(data, list):
        return [to_array(d) for d in data]
    elif isinstance(data, set):
        return set([to_array(d) for d in data])
    elif isinstance(data, dict):
        return {k: to_array(v) for k, v in data.items()}
    elif isinstance(data, Tensor):
        return data.asnumpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise NotImplementedError("Datatype {} cannot be cast to tensor!")


def to_scipy_sparse_matrix(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> scipy.sparse.coo_matrix:
    r"""Converts a graph given by edge indices and edge attributes to a scipy
    sparse matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    Examples:
        >>> edge_index = Tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> to_scipy_sparse_matrix(edge_index)
        <4x4 sparse matrix of type '<class 'numpy.float32'>'
            with 6 stored elements in COOrdinate format>
    """
    src, dst = edge_index.asnumpy()

    if edge_attr is None:
        edge_attr = mint.ones(src.shape[0])
    else:
        edge_attr = edge_attr.reshape(-1)
        assert edge_attr.shape[0] == edge_index.shape[1]

    N = maybe_num_nodes(edge_index, num_nodes)
    out = scipy.sparse.coo_matrix((edge_attr.asnumpy(), (src, dst)), (N, N))
    return out


def from_scipy_sparse_matrix(A: scipy.sparse.spmatrix) -> Tuple[Tensor, Tensor]:
    r"""Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.

    Examples:
        >>> edge_index = Tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> adj = to_scipy_sparse_matrix(edge_index)
        >>> # `edge_index` and `edge_weight` are both returned
        >>> from_scipy_sparse_matrix(adj)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([1., 1., 1., 1., 1., 1.]))
    """
    A = A.tocoo()
    src = Tensor.from_numpy(A.row).astype(ms.int64)
    dst = Tensor.from_numpy(A.col).astype(ms.int64)
    edge_index = mint.stack([src, dst], dim=0)
    edge_weight = Tensor.from_numpy(A.data)
    return edge_index, edge_weight


def to_networkx(
    graph: Union["sharker.data.Graph", "sharker.data.HeteroGraph"],
    node_attrs: Optional[Iterable[str]] = None,
    edge_attrs: Optional[Iterable[str]] = None,
    graph_attrs: Optional[Iterable[str]] = None,
    to_undirected: Optional[Union[bool, str]] = False,
    to_multi: bool = False,
    remove_self_loops: bool = False,
) -> Any:
    r"""Converts a :class:`sharker.data.Graph` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        graph (sharker.data.Graph or sharker.data.HeteroGraph): A
            homogeneous or heterogeneous data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        graph_attrs (iterable of str, optional): The graph attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool or str, optional): If set to :obj:`True`, will
            return a :class:`networkx.Graph` instead of a
            :class:`networkx.DiGraph`.
            By default, will include all edges and make them undirected.
            If set to :obj:`"upper"`, the undirected graph will only correspond
            to the upper triangle of the input adjacency matrix.
            If set to :obj:`"lower"`, the undirected graph will only correspond
            to the lower triangle of the input adjacency matrix.
            Only applicable in case the :obj:`data` object holds a homogeneous
            graph. (default: :obj:`False`)
        to_multi (bool, optional): if set to :obj:`True`, will return a
            :class:`networkx.MultiGraph` or a :class:`networkx:MultiDiGraph`
            (depending on the :obj:`to_undirected` option), which will not drop
            duplicated edges that may exist in :obj:`data`.
            (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self-loops in the resulting graph. (default: :obj:`False`)

    Examples:
        >>> edge_index = Tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Graph(edge_index=edge_index, num_nodes=4)
        >>> to_networkx(data)
        <networkx.classes.digraph.DiGraph at 0x2713fdb40d0>

    """
    import networkx as nx
    from ..data import HeteroGraph

    to_undirected_upper: bool = to_undirected == "upper"
    to_undirected_lower: bool = to_undirected == "lower"

    to_undirected = to_undirected is True
    to_undirected |= to_undirected_upper or to_undirected_lower
    assert isinstance(to_undirected, bool)

    if isinstance(graph, HeteroGraph) and to_undirected:
        raise ValueError(
            "'to_undirected' is not supported in "
            "'to_networkx' for heterogeneous graphs"
        )

    if to_undirected:
        G = nx.MultiGraph() if to_multi else nx.Graph()
    else:
        G = nx.MultiDiGraph() if to_multi else nx.DiGraph()

    def to_networkx_value(value: Any) -> Any:
        return value.tolist() if isinstance(value, Tensor) else value

    for key in graph_attrs or []:
        G.graph[key] = to_networkx_value(graph[key])

    node_offsets = graph.node_offsets
    for node_store in graph.node_stores:
        start = node_offsets[node_store._key]
        assert node_store.num_nodes is not None
        for i in range(node_store.num_nodes):
            node_kwargs: Dict[str, Any] = {}
            if isinstance(graph, HeteroGraph):
                node_kwargs["type"] = node_store._key
            for key in node_attrs or []:
                node_kwargs[key] = to_networkx_value(node_store[key][i])

            G.add_node(start + i, **node_kwargs)

    for edge_store in graph.edge_stores:
        for i, (v, w) in enumerate(edge_store.edge_index.t().tolist()):
            if to_undirected_upper and v > w:
                continue
            elif to_undirected_lower and v < w:
                continue
            elif remove_self_loops and v == w and not edge_store.is_bipartite():
                continue

            edge_kwargs: Dict[str, Any] = {}
            if isinstance(graph, HeteroGraph):
                v = v + node_offsets[edge_store._key[0]]
                w = w + node_offsets[edge_store._key[-1]]
                edge_kwargs["type"] = edge_store._key
            for key in edge_attrs or []:
                edge_kwargs[key] = to_networkx_value(edge_store[key][i])

            G.add_edge(v, w, **edge_kwargs)

    return G


def from_networkx(
    G: Any,
    group_node_attrs: Optional[Union[List[str], Literal["all"]]] = None,
    group_edge_attrs: Optional[Union[List[str], Literal["all"]]] = None,
) -> "sharker.data.Graph":
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`sharker.data.Graph` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or "all", optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or "all", optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.

    Examples:
        >>> edge_index = Tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Graph(edge_index=edge_index, num_nodes=4)
        >>> g = to_networkx(data)
        >>> # A `Graph` object is returned
        >>> from_networkx(g)
        Graph(edge_index=[2, 6], num_nodes=4)
    """
    import networkx as nx
    from ..data import Graph

    G = G.to_directed() if not nx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = -mint.ones((2, G.number_of_edges()), dtype=ms.int64)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data_dict: Dict[str, Any] = defaultdict(list)
    data_dict["edge_index"] = edge_index

    node_attrs: List[str] = []
    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())

    edge_attrs: List[str] = []
    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())

    if group_node_attrs is not None and not isinstance(group_node_attrs, list):
        group_node_attrs = node_attrs

    if group_edge_attrs is not None and not isinstance(group_edge_attrs, list):
        group_edge_attrs = edge_attrs

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError("Not all nodes contain the same attributes")
        for key, value in feat_dict.items():
            data_dict[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError("Not all edges contain the same attributes")
        for key, value in feat_dict.items():
            key = f"edge_{key}" if key in node_attrs else key
            data_dict[str(key)].append(value)

    for key, value in G.graph.items():
        if key == "node_default" or key == "edge_default":
            continue  # Do not load default attributes.
        key = f"graph_{key}" if key in node_attrs else key
        data_dict[str(key)] = value

    for key, value in data_dict.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data_dict[key] = mint.stack(value, dim=0)
        else:
            try:
                data_dict[key] = ms.Tensor(value)
            except Exception:
                pass

    data = Graph.from_dict(data_dict)

    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = mint.cat(xs, dim=-1)

    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f"edge_{key}" if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = mint.cat(xs, dim=-1)

    if data.x is None and data.crd is None:
        data.num_nodes = G.number_of_nodes()

    return data


def to_trimesh(data: "sharker.data.Graph") -> Any:
    r"""Converts a :class:`sharker.data.Graph` instance to a
    :obj:`trimesh.Trimesh`.

    Args:
        data (sharker.data.Graph): The data object.

    Example:
        >>> crd = Tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        ...                    dtype=ms.float32)
        >>> face = Tensor, 3]]).t()

        >>> data = Graph(crd=crd, face=face)
        >>> to_trimesh(data)
        <trimesh.Trimesh(vertices.shape=(4, 3), faces.shape=(2, 3))>
    """
    import trimesh

    assert data.crd is not None
    assert data.face is not None

    return trimesh.Trimesh(
        vertices=data.crd.asnumpy(),
        faces=data.face.T.asnumpy(),
        process=False,
    )


def from_trimesh(mesh: Any) -> "sharker.data.Graph":
    r"""Converts a :obj:`trimesh.Trimesh` to a
    :class:`sharker.data.Graph` instance.

    Args:
        mesh (trimesh.Trimesh): A :obj:`trimesh` mesh.

    Example:
        >>> crd = Tensor [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        ...                    dtype=ms.float32)
        >>> face = Tensor([[0, 1, 2], [1, 2, 3]]).t()

        >>> data = Graph(crd=crd, face=face)
        >>> mesh = to_trimesh(data)
        >>> from_trimesh(mesh)
        Graph(crd=[4, 3], face=[3, 2])
    """
    from ..data import Graph

    crd = Tensor.from_numpy(mesh.vertices).float()
    face = Tensor.from_numpy(mesh.faces).T

    return Graph(crd=crd, face=face)


