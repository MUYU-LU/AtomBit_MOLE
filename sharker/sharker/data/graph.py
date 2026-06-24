import copy
import warnings
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import mindspore as ms
from typing_extensions import Self
from mindspore import Tensor, ops, mint

from .storage import BaseStorage, EdgeStorage, GlobalStorage, NodeStorage
from ..utils import select, subgraph


class Data:
    def __getattr__(self, key: str) -> Any:
        raise NotImplementedError

    def __setattr__(self, key: str, value: Any):
        raise NotImplementedError

    def __delattr__(self, key: str):
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        raise NotImplementedError

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError

    def __delitem__(self, key: str):
        raise NotImplementedError

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self, memo):
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def stores_as(self, data: Self):
        raise NotImplementedError

    @property
    def stores(self) -> List[BaseStorage]:
        raise NotImplementedError

    @property
    def node_stores(self) -> List[NodeStorage]:
        raise NotImplementedError

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dictionary of stored key/value pairs."""
        raise NotImplementedError

    def to_namedtuple(self) -> NamedTuple:
        r"""Returns a :obj:`NamedTuple` of stored key/value pairs."""
        raise NotImplementedError

    def update(self, data: Self) -> Self:
        r"""Updates the data object with the elements from another data object.
        Added elements will override existing ones (in case of duplicates).
        """
        raise NotImplementedError

    def concat(self, data: Self, return_tensor: bool = True) -> Self:
        r"""Concatenates :obj:`self` with another :obj:`data` object.
        All values needs to have matching shapes at non-concat dimensions.
        """
        out = copy.copy(self)
        for store, other_store in zip(out.stores, data.stores):
            store.concat(other_store)
        if return_tensor == True:
            out.tensor()
        else:
            out.numpy()
        return out

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Returns the dimension for which the value :obj:`value` of the
        attribute :obj:`key` will get concatenated when creating mini-batches
        using :class:`sharker.loader.DataLoader`.

        .. note::

            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        raise NotImplementedError

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Returns the incremental count to cumulatively increase the value
        :obj:`value` of the attribute :obj:`key` when creating mini-batches
        using :class:`sharker.loader.DataLoader`.

        .. note::

            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        raise NotImplementedError

    ###########################################################################

    def keys(self) -> List[str]:
        r"""Returns a list of all graph attribute names."""
        out = []
        for store in self.stores:
            out += list(store.keys())
        return list(set(out))

    def __len__(self) -> int:
        r"""Returns the number of graph attributes."""
        return len(self.keys())

    def __contains__(self, key: str) -> bool:
        r"""Returns :obj:`True` if the attribute :obj:`key` is present in the
        data.
        """
        return key in self.keys()

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.__dict__[key] = value

    @property
    def num_nodes(self) -> Optional[int]:
        r"""Returns the number of nodes in the graph.

        .. note::
            The number of nodes in the data object is automatically inferred
            in case node-level attributes are present, *e.g.*, :obj:`data.x`.
            In some cases, however, a graph may only be given without any
            node-level attributes.
            :mindgeometric:`MindGeometric` then *guesses* the number of nodes according to
            :obj:`edge_index.max().item() + 1`.
            However, in case there exists isolated nodes, this number does not
            have to be correct which can result in unexpected behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        try:
            size = sum([v.num_nodes for v in self.node_stores])
            if isinstance(size, Tensor):
                size = size.item()
            return size
        except TypeError:
            return None

    @property
    def shape(self) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        r"""Returns the size of the adjacency matrix induced by the graph."""
        shape = (self.num_nodes, self.num_nodes)
        return shape

    @property
    def num_edges(self) -> int:
        r"""Returns the number of edges in the graph.
        For undirected graphs, this will return the number of bi-directional
        edges, which is double the amount of unique edges.
        """
        size = sum([v.num_edges for v in self.edge_stores])
        if isinstance(size, Tensor):
            size = size.item()
        return size

    def node_attrs(self) -> List[str]:
        r"""Returns all node-level tensor attribute names."""
        return list(set(chain(*[s.node_attrs() for s in self.node_stores])))

    def edge_attrs(self) -> List[str]:
        r"""Returns all edge-level tensor attribute names."""
        return list(set(chain(*[s.edge_attrs() for s in self.edge_stores])))

    @property
    def node_offsets(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        offset: int = 0
        for store in self.node_stores:
            out[store._key] = offset
            offset = offset + store.num_nodes
        return out

    def generate_ids(self):
        r"""Generates and sets :obj:`n_id` and :obj:`e_id` attributes to assign
        each node and edge to a continuously ascending and unique ID.
        """
        for store in self.node_stores:
            store.n_id = np.arange(store.num_nodes)
        for store in self.edge_stores:
            store.e_id = np.arange(store.num_edges)

    def is_sorted(self, sort_by_row: bool = True) -> bool:
        r"""Returns :obj:`True` if edge indices :obj:`edge_index` are sorted.

        Args:
            sort_by_row (bool, optional): If set to :obj:`False`, will require
                column-wise order/by destination node order of
                :obj:`edge_index`. (default: :obj:`True`)
        """
        input_graph = copy.copy(self).numpy()
        return all([store.is_sorted(sort_by_row) for store in input_graph.edge_stores])

    def sort(self, sort_by_row: bool = True, return_tensor: bool = True) -> Self:
        r"""Sorts edge indices :obj:`edge_index` and their corresponding edge
        features.

        Args:
            sort_by_row (bool, optional): If set to :obj:`False`, will sort
                :obj:`edge_index` in column-wise order/by destination node.
                (default: :obj:`True`)
        """
        out = copy.copy(self).numpy()
        for store in out.edge_stores:
            store.sort(sort_by_row)
        if return_tensor == True:
            out.tensor()
        else:
            out.numpy()
        return out

    def is_coalesced(self) -> bool:
        r"""Returns :obj:`True` if edge indices :obj:`edge_index` are sorted
        and do not contain duplicate entries.
        """
        input_graph = copy.copy(self).numpy()
        return all([store.is_coalesced() for store in input_graph.edge_stores])

    def coalesce(self, return_tensor: bool = True) -> Self:
        r"""Sorts and removes duplicated entries from edge indices
        :obj:`edge_index`.
        """
        out = copy.copy(self).numpy()
        for store in out.edge_stores:
            store.coalesce()
        if return_tensor == True:
            out.tensor()
        else:
            out.numpy()
        return out

    def is_sorted_by_time(self) -> bool:
        r"""Returns :obj:`True` if :obj:`time` is sorted."""
        input_graph = copy.copy(self).numpy()
        return all([store.is_sorted_by_time() for store in input_graph.stores])

    def sort_by_time(self, return_tensor: bool = True) -> Self:
        r"""Sorts data associated with :obj:`time` according to :obj:`time`."""
        out = copy.copy(self).numpy()
        for store in out.stores:
            store.sort_by_time()
        if return_tensor == True:
            out.tensor()
        else:
            out.numpy()
        return out

    def snapshot(
        self,
        start_time: Union[float, int],
        end_time: Union[float, int],
        return_tensor: bool = True
    ) -> Self:
        r"""Returns a snapshot of :obj:`data` to only hold events that occurred
        in period :obj:`[start_time, end_time]`.
        """
        out = copy.copy(self).numpy()
        for store in out.stores:
            store.snapshot(start_time, end_time)
        if return_tensor == True:
            out.tensor()
        else:
            out.numpy()
        return out

    def up_to(self, end_time: Union[float, int], return_tensor: bool = True) -> Self:
        r"""Returns a snapshot of :obj:`data` to only hold events that occurred
        up to :obj:`end_time` (inclusive of :obj:`edge_time`).
        """
        out = copy.copy(self).numpy()
        for store in out.stores:
            store.up_to(end_time)
        if return_tensor == True:
            out.tensor()
        else:
            out.numpy()
        return out

    def has_isolated_nodes(self) -> bool:
        r"""Returns :obj:`True` if the graph contains isolated nodes."""
        return any([store.has_isolated_nodes() for store in self.edge_stores])

    def has_self_loops(self) -> bool:
        """Returns :obj:`True` if the graph contains self-loops."""
        return any([store.has_self_loops() for store in self.edge_stores])

    def is_undirected(self) -> bool:
        r"""Returns :obj:`True` if graph edges are undirected."""
        return all([store.is_undirected() for store in self.edge_stores])

    def is_directed(self) -> bool:
        r"""Returns :obj:`True` if graph edges are directed."""
        return not self.is_undirected()

    def apply_(self, func: Callable, *args: str):
        r"""Applies the in-place function :obj:`func`, either to all attributes
        or only the ones given in :obj:`*args`.
        """
        for store in self.stores:
            store.apply_(func, *args)
        return self

    def apply(self, func: Callable, *args: str):
        r"""Applies the function :obj:`func`, either to all attributes or only
        the ones given in :obj:`*args`.
        """
        for store in self.stores:
            store.apply(func, *args)
        return self

    def numpy(self, *args: str):
        r"""Copies attributes to CPU memory, either for all attributes or only
        the ones given in :obj:`*args`.
        """
        return self.apply(lambda x: x.asnumpy() if isinstance(x, Tensor) else x, *args)

    def tensor(self, *args: str):
        r"""Copies attributes to CPU memory, either for all attributes or only
        the ones given in :obj:`*args`.
        """
        return self.apply(
            lambda x: Tensor.from_numpy(x) if isinstance(x, np.ndarray) else x, *args
        )

    def copy(self, *args: str):
        r"""Performs cloning of tensors, either for all attributes or only the
        ones given in :obj:`*args`.
        """
        return copy.copy(self).apply(lambda x: x.copy(), *args)
###############################################################################

@ms.jit_class
class Graph(Data):
    r"""A graph object describing a homogeneous graph.
    The data object can hold node-level, link-level and graph-level attributes.
    In general, :class:`~sharker.data.Graph` tries to mimic the
    behavior of a regular :python:`Python` dictionary.
    In addition, it provides useful functionality for analyzing graph
    structures, and provides basic MindSpore tensor functionalities.
    See `here <https://sharker.readthedocs.io/en/latest/get_started/
    introduction.html#data-handling-of-graphs>`__ for the accompanying
    tutorial.

    .. code-block:: python

        from sharker.data import Graph

        data = Graph(x=x, edge_index=edge_index, ...)

        # Add additional arguments to `data`:
        data.train_idx = Tensor([...], dtype=ms.int64)
        data.test_mask = Tensor([...], dtype=ms.bool_)

        # Analyzing the graph structure:
        data.num_nodes
        >>> 23

        data.is_directed()
        >>> False

        # MindSpore tensor functionality:
        data = data.pin_memory()

    Args:
        x (Tensor, optional): Node feature matrix with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph-level or node-level ground-truth
            labels with arbitrary shape. (default: :obj:`None`)
        crd (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        time (Tensor, optional): The timestamps for each event with shape
            :obj:`[num_edges]` or :obj:`[num_nodes]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """

    def __init__(
        self,
        x: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        y: Optional[Union[Tensor, int, float]] = None,
        crd: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
        **kwargs,
    ):
        self.__dict__["_store"] = GlobalStorage(_parent=self)

        if x is not None:
            self.x = x
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if y is not None:
            self.y = y
        if crd is not None:
            self.crd = crd
        if time is not None:
            self.time = time

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, key: str) -> Any:
        if "_store" not in self.__dict__:
            raise RuntimeError(
                "The 'data' object was created by an older version of MindeGometric. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again."
            )
        return getattr(self._store, key)

    def __setattr__(self, key: str, value: Any):
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, "fset", None) is not None:
            propobj.fset(self, value)
        else:
            setattr(self._store, key, value)

    def __delattr__(self, key: str):
        delattr(self._store, key)

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any):
        self._store[key] = value

    def __delitem__(self, key: str):
        if key in self._store:
            del self._store[key]

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__["_store"] = copy.copy(self._store)
        out._store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        return out

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        has_dict = any([isinstance(v, Mapping) for v in self._store.values()])

        if not has_dict:
            info = [size_repr(k, v) for k, v in self._store.items()]
            info = ", ".join(info)
            return f"{cls}({info})"
        else:
            info = [size_repr(k, v, indent=2) for k, v in self._store.items()]
            info = ",\n".join(info)
            return f"{cls}(\n{info}\n)"

    @property
    def num_nodes(self) -> Optional[int]:
        return super().num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: Optional[int]):
        self._store.num_nodes = num_nodes

    def stores_as(self, data: Self):
        return self

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def node_stores(self) -> List[NodeStorage]:
        return [self._store]

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return [self._store]

    def to_dict(self) -> Dict[str, Any]:
        return self._store.to_dict()

    def to_namedtuple(self) -> NamedTuple:
        return self._store.to_namedtuple()

    def update(self, data: Union[Self, Dict[str, Any]]) -> Self:
        for key, value in data.items():
            self[key] = value
        return self

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "adj" in key:
            return (0, 1)
        elif "index" in key or key == "face":
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "batch" in key and isinstance(value, (Tensor, np.ndarray)):
            return int(value.max()) + 1
        elif "index" in key or key == "face":
            return self.num_nodes
        else:
            return 0

    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the data."""
        cls_name = self.__class__.__name__
        status = True

        num_nodes = self.num_nodes
        if num_nodes is None:
            status = False
            warn_or_raise(f"'num_nodes' is undefined in '{cls_name}'", raise_on_error)

        if "edge_index" in self:
            edge_index_np = self.edge_index.asnumpy() if isinstance(self.edge_index, ms.Tensor) else self.edge_index
            if edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
                status = False
                warn_or_raise(
                    f"'edge_index' needs to be of shape [2, num_edges] in "
                    f"'{cls_name}' (found {edge_index_np.shape})",
                    raise_on_error,
                )

        if "edge_index" in self and self.edge_index.size > 0:
            if np.max(edge_index_np) < 0:
                status = False
                warn_or_raise(
                    f"'edge_index' contains negative indices in "
                    f"'{cls_name}' (found {int(edge_index_np.min())})",
                    raise_on_error,
                )

            if np.max(edge_index_np) >= num_nodes:
                status = False
                warn_or_raise(
                    f"'edge_index' contains larger indices than the number "
                    f"of nodes ({num_nodes}) in '{cls_name}' "
                    f"(found {int(edge_index_np.max())})",
                    raise_on_error,
                )

        return status

    def is_node_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at key :obj:`key` denotes a
        node-level tensor attribute.
        """
        return self._store.is_node_attr(key)

    def is_edge_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at key :obj:`key` denotes an
        edge-level tensor attribute.
        """
        return self._store.is_edge_attr(key)

    def subgraph(self, subset: Union[Tensor, np.ndarray], return_tensor: bool = True) -> Self:
        r"""Returns the induced subgraph given by the node indices
        :obj:`subset`.

        Args:
            subset (LongTensor or BoolTensor): The nodes to keep.
        """
        if isinstance(subset, Tensor):
            subset = subset.asnumpy()
        input_graph = copy.copy(self).numpy()

        if "edge_index" in input_graph:
            edge_index, _, edge_mask = subgraph(
                subset,
                input_graph.edge_index,
                relabel_nodes=True,
                num_nodes=input_graph.num_nodes,
                return_edge_mask=True,
            )
        else:
            edge_index = None
            edge_mask = np.ones(input_graph.num_edges, dtype=np.bool_)

        data = copy.copy(input_graph)

        for key, value in input_graph:
            if key == "edge_index":
                data.edge_index = edge_index
            elif key == "num_nodes":
                if subset.dtype == np.bool_:
                    data.num_nodes = int(subset.sum())
                else:
                    data.num_nodes = subset.shape[0]
            elif input_graph.is_node_attr(key):
                cat_dim = input_graph.__cat_dim__(key, value)
                data[key] = select(value, subset, axis=cat_dim)
            elif input_graph.is_edge_attr(key):
                cat_dim = input_graph.__cat_dim__(key, value)
                data[key] = select(value, edge_mask, axis=cat_dim)

        if return_tensor == True:
            data.tensor()
        else:
            data.numpy()
        return data

    def edge_subgraph(self, subset: Union[Tensor, np.ndarray], return_tensor: bool = True) -> Self:
        r"""Returns the induced subgraph given by the edge indices
        :obj:`subset`.
        Will currently preserve all the nodes in the graph, even if they are
        isolated after subgraph computation.

        Args:
            subset (LongTensor or BoolTensor): The edges to keep.
        """
        data = copy.copy(self).numpy()
        input_graph = copy.copy(self).numpy()
        if isinstance(subset, Tensor):
            subset = subset.asnumpy()

        for key, value in input_graph:
            if input_graph.is_edge_attr(key):
                cat_dim = input_graph.__cat_dim__(key, value)
                data[key] = select(value, subset, axis=cat_dim)
        if return_tensor == True:
            data.tensor()
        else:
            data.numpy()
        return data

    def to_hetero(
        self,
        node_type: Optional[Tensor] = None,
        edge_type: Optional[Tensor] = None,
        node_type_names: Optional[List[str]] = None,
        edge_type_names: Optional[List[Tuple[str, str, str]]] = None,
        return_tensor: bool = True
    ):
        r"""Converts a :class:`~sharker.data.Graph` object to a
        heterogeneous :class:`~sharker.data.HeteroGraph` object.
        For this, node and edge attributes are splitted according to the
        node-level and edge-level vectors :obj:`node_type` and
        :obj:`edge_type`, respectively.
        :obj:`node_type_names` and :obj:`edge_type_names` can be used to give
        meaningful node and edge type names, respectively.
        That is, the node_type :obj:`0` is given by :obj:`node_type_names[0]`.
        If the :class:`~sharker.data.Graph` object was constructed via
        :meth:`~sharker.data.HeteroGraph.to_homogeneous`, the object can
        be reconstructed without any need to pass in additional arguments.

        Args:
            node_type (Tensor, optional): A node-level vector denoting
                the type of each node. (default: :obj:`None`)
            edge_type (Tensor, optional): An edge-level vector denoting
                the type of each edge. (default: :obj:`None`)
            node_type_names (List[str], optional): The names of node types.
                (default: :obj:`None`)
            edge_type_names (List[Tuple[str, str, str]], optional): The names
                of edge types. (default: :obj:`None`)
        """
        from .heterograph import HeteroGraph

        input_graph = copy.copy(self).numpy()
        if node_type is not None and isinstance(node_type, Tensor):
            node_type = node_type.asnumpy()
        if edge_type is not None and isinstance(edge_type, Tensor):
            edge_type = edge_type.asnumpy()


        if node_type is None:
            node_type = input_graph._store.get("node_type", None)
        if node_type is None:
            node_type = np.zeros(input_graph.num_nodes, dtype=np.int64)

        if node_type_names is None:
            store = input_graph._store
            node_type_names = store.__dict__.get("_node_type_names", None)
        if node_type_names is None:
            node_type_names = [str(i) for i in np.unique(node_type)]

        if edge_type is None:
            edge_type = input_graph._store.get("edge_type", None)
        if edge_type is None:
            edge_type = np.zeros(input_graph.num_edges, dtype=np.int64)

        if edge_type_names is None:
            store = input_graph._store
            edge_type_names = store.__dict__.get("_edge_type_names", None)
        if edge_type_names is None:
            edge_type_names = []
            edge_index = input_graph.edge_index
            for i in np.unique(edge_type):
                src, dst = edge_index[:, edge_type == i]
                src_types = np.unique(node_type[src])
                dst_types = np.unique(node_type[dst])
                if len(src_types) != 1 and len(dst_types) != 1:
                    raise ValueError(
                        "Could not construct a 'HeteroGraph' object from the "
                        "'Graph' object because single edge types span over "
                        "multiple node types"
                    )
                edge_type_names.append(
                    (
                        node_type_names[src_types.item(0)],
                        str(i),
                        node_type_names[dst_types.item(0)],
                    )
                )

        # We iterate over node types to find the local node indices belonging
        # to each node type. Furthermore, we create a global `index_map` vector
        # that maps global node indices to local ones in the final
        # heterogeneous graph:
        node_ids, index_map = {}, np.zeros_like(node_type)
        for i, key in enumerate(node_type_names):
            node_ids[i] = np.nonzero((node_type == i).reshape(-1))[0]
            index_map[node_ids[i]] = np.arange(len(node_ids[i]))

        # We iterate over edge types to find the local edge indices:
        edge_ids = {}
        for i, key in enumerate(edge_type_names):
            edge_ids[i] = np.nonzero((edge_type == i).reshape(-1))[0]

        graph = HeteroGraph()

        for i, key in enumerate(node_type_names):
            for attr, value in input_graph.items():
                if attr in {"node_type", "edge_type", "ptr"}:
                    continue
                elif isinstance(value, np.ndarray) and input_graph.is_node_attr(attr):
                    cat_dim = input_graph.__cat_dim__(attr, value)
                    graph[key][attr] = np.take(value, node_ids[i], cat_dim)
            if len(graph[key]) == 0:
                graph[key].num_nodes = node_ids[i].shape[0]

        for i, key in enumerate(edge_type_names):
            src, _, dst = key
            for attr, value in input_graph.items():
                if attr in {"node_type", "edge_type", "ptr"}:
                    continue
                elif attr == "edge_index":
                    edge_index = value[:, edge_ids[i]]
                    edge_index[0] = index_map[edge_index[0]]
                    edge_index[1] = index_map[edge_index[1]]
                    graph[key].edge_index = edge_index
                elif isinstance(value, np.ndarray) and input_graph.is_edge_attr(attr):
                    cat_dim = input_graph.__cat_dim__(attr, value)
                    graph[key][attr] = np.take(value, edge_ids[i], cat_dim)

        # Add global attributes.
        exclude_keys = set(graph.keys()) | {
            "node_type",
            "edge_type",
            "edge_index",
            "num_nodes",
            "ptr",
        }
        for attr, value in input_graph.items():
            if attr in exclude_keys:
                continue
            graph[attr] = value

        if return_tensor == True:
            graph.tensor()
        else:
            graph.numpy()
        return graph

    ###########################################################################

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]) -> Self:
        r"""Creates a :class:`~sharker.data.Graph` object from a
        dictionary.
        """
        return cls(**mapping)

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the graph."""
        return self._store.num_node_features

    @property
    def num_features(self) -> int:
        r"""Returns the number of features per node in the graph.
        Alias for :py:attr:`~num_node_features`.
        """
        return self.num_node_features

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the graph."""
        return self._store.num_edge_features

    @property
    def num_node_types(self) -> int:
        r"""Returns the number of node types in the graph."""
        return int(self.node_type.max()) + 1 if "node_type" in self else 1

    @property
    def num_edge_types(self) -> int:
        r"""Returns the number of edge types in the graph."""
        return int(self.edge_type.max()) + 1 if "edge_type" in self else 1

    def __iter__(self) -> Iterable:
        r"""Iterates over all attributes in the data, yielding their attribute
        names and values.
        """
        for key, value in self._store.items():
            yield key, value

    @property
    def x(self) -> Optional[Tensor]:
        return self["x"] if "x" in self._store else None

    @x.setter
    def x(self, x: Optional[Tensor]):
        self._store.x = x

    @property
    def edge_index(self) -> Optional[Tensor]:
        return self["edge_index"] if "edge_index" in self._store else None

    @edge_index.setter
    def edge_index(self, edge_index: Optional[Tensor]):
        self._store.edge_index = edge_index

    @property
    def edge_weight(self) -> Optional[Tensor]:
        return self["edge_weight"] if "edge_weight" in self._store else None

    @edge_weight.setter
    def edge_weight(self, edge_weight: Optional[Tensor]):
        self._store.edge_weight = edge_weight

    @property
    def edge_attr(self) -> Optional[Tensor]:
        return self["edge_attr"] if "edge_attr" in self._store else None

    @edge_attr.setter
    def edge_attr(self, edge_attr: Optional[Tensor]):
        self._store.edge_attr = edge_attr

    @property
    def y(self) -> Optional[Union[Tensor, int, float]]:
        return self["y"] if "y" in self._store else None

    @y.setter
    def y(self, y: Optional[Tensor]):
        self._store.y = y

    @property
    def crd(self) -> Optional[Tensor]:
        return self["crd"] if "crd" in self._store else None

    @crd.setter
    def crd(self, crd: Optional[Tensor]):
        self._store.crd = crd

    @property
    def batch(self) -> Optional[Tensor]:
        return self["batch"] if "batch" in self._store else None

    @batch.setter
    def batch(self, batch: Optional[Tensor]):
        self._store.batch = batch

    @property
    def time(self) -> Optional[Tensor]:
        return self["time"] if "time" in self._store else None

    @time.setter
    def time(self, time: Optional[Tensor]):
        self._store.time = time

    @property
    def face(self) -> Optional[Tensor]:
        return self["face"] if "face" in self._store else None

    @face.setter
    def face(self, face: Optional[Tensor]):
        self._store.face = face


###############################################################################


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, Tensor) and value.dim() == 0:
        out = value.item()
    elif isinstance(value, Tensor) and getattr(value, "is_nested", False):
        out = str(list(value.to_padded_tensor(padding=0.0).shape))
    elif isinstance(value, Tensor):
        out = str(list(value.shape))
    elif isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 0:
        out = "{}"
    elif (
        isinstance(value, Mapping)
        and len(value) == 1
        and not isinstance(list(value.values())[0], Mapping)
    ):
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = "{ " + ", ".join(lines) + " }"
    elif isinstance(value, Mapping):
        lines = [size_repr(k, v, indent + 2) for k, v in value.items()]
        out = "{\n" + ",\n".join(lines) + ",\n" + pad + "}"
    else:
        out = str(value)

    key = str(key).replace("'", "")
    return f"{pad}{key}={out}"


def warn_or_raise(msg: str, raise_on_error: bool = True):
    if raise_on_error:
        raise ValueError(msg)
    else:
        warnings.warn(msg)
